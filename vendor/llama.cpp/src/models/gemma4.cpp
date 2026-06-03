#include "models.h"
#include "llama-infinitum-moe.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

static bool gemma4_infinitum_debug_enabled() {
    const char * value = std::getenv("LLAMA_INFINITUM_DEBUG");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static void gemma4_infinitum_debug_log(const char * fmt, ...) {
    if (!gemma4_infinitum_debug_enabled()) {
        return;
    }
    std::fprintf(stderr, "infinitum-gemma4: ");
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
}

static bool gemma4_infinitum_q4_slots_enabled() {
    const char * value = std::getenv("LLAMA_INFINITUM_GEMMA4_Q4_SLOTS");
    if (value != nullptr && value[0] != '\0' && value[0] != '0') {
        return true;
    }
    value = std::getenv("LLAMA_INFINITUM_GEMMA4_F16_SLOTS");
    if (value != nullptr && value[0] != '\0' && value[0] != '0') {
        return true;
    }
    value = std::getenv("LLAMA_INFINITUM_GEMMA4_DIRECT_F16");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool gemma4_infinitum_q4_slot_report_enabled() {
    const char * value = std::getenv("LLAMA_INFINITUM_GEMMA4_Q4_SLOT_REPORT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static ggml_type gemma4_infinitum_ggml_expert_pack_type() {
    const char * value = std::getenv("LLAMA_INFINITUM_GEMMA4_GGML_EXPERT_PACK_TYPE");
    if (value != nullptr && std::strcmp(value, "f16") == 0) {
        return GGML_TYPE_F16;
    }
    if (value != nullptr && std::strcmp(value, "q8_0") == 0) {
        return GGML_TYPE_Q8_0;
    }
    return GGML_TYPE_Q4_0;
}

static bool gemma4_infinitum_direct_f16_enabled() {
    const char * value = std::getenv("LLAMA_INFINITUM_GEMMA4_DIRECT_F16");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool gemma4_infinitum_direct_q8_enabled() {
    const char * value = std::getenv("LLAMA_INFINITUM_GEMMA4_DIRECT_Q8");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static int64_t gemma4_infinitum_top_k_limit(int64_t n_expert_used) {
    const char * value = std::getenv("LLAMA_INFINITUM_GEMMA4_TOP_K_LIMIT");
    if (value == nullptr || value[0] == '\0') {
        return n_expert_used;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return n_expert_used;
    }
    return std::max<int64_t>(1, std::min<int64_t>(n_expert_used, parsed));
}

static float gemma4_infinitum_tensor_f32(const ggml_tensor * tensor, int64_t i0, int64_t i1, int64_t i2) {
    const char * base = static_cast<const char *>(tensor->data);
    const char * ptr = base + i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2];
    return *reinterpret_cast<const float *>(ptr);
}

static int gemma4_infinitum_tensor_i32(const ggml_tensor * tensor, int64_t i0, int64_t i1) {
    const char * base = static_cast<const char *>(tensor->data);
    const char * ptr = base + i0 * tensor->nb[0] + i1 * tensor->nb[1];
    return *reinterpret_cast<const int32_t *>(ptr);
}

struct gemma4_infinitum_q4_slots_userdata {
    const llama_infinitum_moe_index_info * expert_index = nullptr;
    int layer_index = -1;
};

static gemma4_infinitum_q4_slots_userdata * gemma4_infinitum_q4_slots_userdata_for_layer(
        const llama_infinitum_moe_index_info & expert_index,
        int layer_index) {
    static std::array<gemma4_infinitum_q4_slots_userdata, 96> userdata_by_layer = {};
    if (layer_index < 0 || layer_index >= int(userdata_by_layer.size())) {
        GGML_ABORT("Gemma4 Infinitum Q4 slot-cache layer index exceeds userdata cache");
    }
    gemma4_infinitum_q4_slots_userdata & userdata = userdata_by_layer[std::size_t(layer_index)];
    userdata.expert_index = &expert_index;
    userdata.layer_index = layer_index;
    return &userdata;
}

static const llama_infinitum_moe_index_info & gemma4_infinitum_expert_index() {
    static const llama_infinitum_moe_index_info index = llama_infinitum_moe_index_from_env();
    return index;
}

static void gemma4_infinitum_q4_slots_op(
        ggml_tensor * dst,
        const ggml_tensor * hidden_tensor,
        const ggml_tensor * selected_tensor,
        const ggml_tensor * weights_tensor,
        int ith,
        int /*nth*/,
        void * userdata) {
    if (ith != 0) {
        return;
    }
    if (hidden_tensor->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            selected_tensor->type != GGML_TYPE_I32 || weights_tensor->type != GGML_TYPE_F32) {
        GGML_ABORT("Gemma4 Infinitum Q4 slot-cache custom op received unsupported tensor types");
    }
    auto * data = static_cast<gemma4_infinitum_q4_slots_userdata *>(userdata);
    if (data == nullptr || data->expert_index == nullptr) {
        GGML_ABORT("Gemma4 Infinitum Q4 slot-cache custom op missing userdata");
    }

    const int hidden_size = static_cast<int>(hidden_tensor->ne[0]);
    const int64_t n_tokens = std::max<int64_t>(1, hidden_tensor->ne[1]);
    const int64_t n_expert_used = selected_tensor->ne[0];
    thread_local std::vector<float> hidden_scratch;
    thread_local std::vector<float> output_scratch;
    thread_local std::vector<int> selected_scratch;
    thread_local std::vector<float> weights_scratch;
    selected_scratch.resize(std::size_t(n_expert_used));
    weights_scratch.resize(std::size_t(n_expert_used));

    for (int64_t token = 0; token < n_tokens; ++token) {
        const float * hidden_data = nullptr;
        if (hidden_tensor->nb[0] == static_cast<int64_t>(sizeof(float))) {
            const char * base = static_cast<const char *>(hidden_tensor->data);
            hidden_data = reinterpret_cast<const float *>(base + token * hidden_tensor->nb[1]);
        } else {
            hidden_scratch.resize(std::size_t(hidden_size));
            for (int i = 0; i < hidden_size; ++i) {
                hidden_scratch[std::size_t(i)] = gemma4_infinitum_tensor_f32(hidden_tensor, i, token, 0);
            }
            hidden_data = hidden_scratch.data();
        }

        float * output_data = nullptr;
        const bool output_contiguous = dst->nb[0] == static_cast<int64_t>(sizeof(float));
        if (output_contiguous) {
            char * base = static_cast<char *>(dst->data);
            output_data = reinterpret_cast<float *>(base + token * dst->nb[1]);
        } else {
            output_scratch.resize(std::size_t(hidden_size));
            output_data = output_scratch.data();
        }

        for (int64_t k = 0; k < n_expert_used; ++k) {
            selected_scratch[std::size_t(k)] = gemma4_infinitum_tensor_i32(selected_tensor, k, token);
            weights_scratch[std::size_t(k)] = gemma4_infinitum_tensor_f32(weights_tensor, 0, k, token);
        }

        const llama_infinitum_moe_expert_mlp_result result =
            gemma4_infinitum_direct_f16_enabled() ?
            llama_infinitum_moe_execute_gemma4_f16_pack_selected_experts_into(
                *data->expert_index, data->layer_index, selected_scratch, weights_scratch,
                hidden_data, hidden_size, output_data) :
            gemma4_infinitum_direct_q8_enabled() ?
            llama_infinitum_moe_execute_gemma4_q8_pack_selected_experts_into(
                *data->expert_index, data->layer_index, selected_scratch, weights_scratch,
                hidden_data, hidden_size, output_data) :
            llama_infinitum_moe_execute_gemma4_q4_pack_selected_experts_into(
                *data->expert_index, data->layer_index, selected_scratch, weights_scratch,
                hidden_data, hidden_size, output_data);
        if (gemma4_infinitum_q4_slot_report_enabled() && token == 0 && data->layer_index < 2) {
            std::fprintf(stderr,
                    "gemma4_q4_slot_report: layer=%d backend=%s norm=%.6g first_id=%d first_weight=%.6g resident_mb=%.1f hits=%llu misses=%llu evict=%llu loaded_mb=%.1f\n",
                    data->layer_index,
                    result.backend.c_str(),
                    double(result.output_norm),
                    selected_scratch.empty() ? -1 : selected_scratch[0],
                    weights_scratch.empty() ? 0.0 : double(weights_scratch[0]),
                    double(result.resident_bytes) / (1024.0 * 1024.0),
                    static_cast<unsigned long long>(result.cache_hits_delta),
                    static_cast<unsigned long long>(result.cache_misses_delta),
                    static_cast<unsigned long long>(result.cache_evictions_delta),
                    double(result.loaded_bytes_delta) / (1024.0 * 1024.0));
        }
        if (!result.ok) {
            gemma4_infinitum_debug_log("q4 slot-cache failed layer=%d error=%s", data->layer_index, result.error.c_str());
            GGML_ABORT("Gemma4 Infinitum Q4 slot-cache custom op failed");
        }

        if (!output_contiguous) {
            for (int i = 0; i < hidden_size; ++i) {
                char * base = static_cast<char *>(dst->data);
                char * ptr = base + i * dst->nb[0] + token * dst->nb[1];
                std::memcpy(ptr, &output_data[i], sizeof(float));
            }
        }
    }
}

void llama_model_gemma4::load_arch_hparams(llama_model_loader & ml) {
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);

    uint32_t n_kv_shared_layers = 0;
    ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);

    hparams.n_layer_kv_from_start = hparams.n_layer - (int32_t)n_kv_shared_layers;
    hparams.f_attention_scale     = 1.0f; // Gemma4 uses self.scaling = 1.0 (no pre-attn scaling)

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,    hparams.n_embd_head_k_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,  hparams.n_embd_head_v_swa);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    switch (hparams.n_layer) {
        case 30: type = LLM_TYPE_26B_A4B; break;
        case 35: type = LLM_TYPE_E2B; break;
        case 42: type = LLM_TYPE_E4B; break;
        case 60: type = LLM_TYPE_31B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_gemma4::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const uint32_t n_embd_per_layer = hparams.n_embd_per_layer;
    const int64_t  n_ff_exp         = hparams.n_ff_exp;
    const bool external_moe = llama_infinitum_moe_selective_enabled();
    gemma4_infinitum_debug_log(
            "load_arch_tensors external_moe=%d n_layer=%d n_ff_exp=%lld n_expert=%lld",
            external_moe ? 1 : 0, (int) n_layer, (long long) n_ff_exp, (long long) n_expert);

    if (n_embd_head_k != n_embd_head_v) {
        throw std::runtime_error("Gemma 4 requires n_embd_head_k == n_embd_head_v");
    }
    if (hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa) {
        throw std::runtime_error("Gemma 4 requires n_embd_head_k_swa == n_embd_head_v_swa");
    }

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    if (n_embd_per_layer > 0) {
        per_layer_tok_embd   = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),    {n_embd_per_layer * n_layer, n_vocab}, 0);
        per_layer_model_proj = create_tensor(tn(LLM_TENSOR_PER_LAYER_MODEL_PROJ, "weight", 0), {n_embd, n_embd_per_layer * n_layer}, 0);
        per_layer_proj_norm  = create_tensor(tn(LLM_TENSOR_PER_LAYER_PROJ_NORM,  "weight", 0), {n_embd_per_layer}, 0);
    }

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer; ++i) {
        gemma4_infinitum_debug_log("load_arch_tensors layer=%d begin", i);
        auto & layer = layers[i];
        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_embd_k    = hparams.n_embd_k_gqa(i);
        const int64_t n_embd_v    = hparams.n_embd_v_gqa(i);
        const int     kv_flags    = hparams.has_kv(i) ? 0 : TENSOR_NOT_REQUIRED;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // note: use_alternative_attention (v_proj is optional, if it's not present, use k_proj)
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k}, kv_flags);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v}, TENSOR_NOT_REQUIRED);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head * n_head, n_embd}, 0);

        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head}, 0);
        layer.attn_k_norm    = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), {n_embd_head}, kv_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.out_scale = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), {1u}, TENSOR_NOT_REQUIRED);

        if (!hparams.is_swa(i)) {
            // full_attention layers use rope_freqs for proportional rope
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head/2}, rope_freqs_flag);
            rope_freqs_flag = TENSOR_DUPLICATED;
        }

        // handle use_double_wide_mlp
        int64_t n_ff_cur = hparams.n_ff(i);

        // for expert layers, we use normal FFN as shared expert (same as python code)
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_cur, n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

        // MoE router
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);
        bool has_expert = layer.ffn_gate_inp != nullptr;

        // norm
        if (has_expert) {
            gemma4_infinitum_debug_log("load_arch_tensors layer=%d has_expert", i);
            layer.ffn_gate_inp_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "scale", i), {n_embd}, 0);

            layer.ffn_pre_norm_2  = create_tensor(tn(LLM_TENSOR_FFN_PRE_NORM_2,  "weight", i), {n_embd}, 0);
            layer.ffn_post_norm_1 = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM_1, "weight", i), {n_embd}, 0);
            layer.ffn_post_norm_2 = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM_2, "weight", i), {n_embd}, 0);

            // MoE FFN
            layer.ffn_gate_up_exps  = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS,  "weight", i), {n_embd, n_ff_exp * 2, n_expert}, TENSOR_NOT_REQUIRED);

            if (layer.ffn_gate_up_exps == nullptr) {
                const int split_expert_flags = external_moe ? TENSOR_NOT_REQUIRED : 0;
                layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, split_expert_flags);
                layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, split_expert_flags);
            }

            layer.ffn_down_exps     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,     "weight", i), {n_ff_exp, n_embd, n_expert}, external_moe ? TENSOR_NOT_REQUIRED : 0);

            layer.ffn_down_exps_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
        }
        gemma4_infinitum_debug_log("load_arch_tensors layer=%d done", i);

        // per-layer embeddings
        if (n_embd_per_layer > 0) {
            layer.per_layer_inp_gate   = create_tensor(tn(LLM_TENSOR_PER_LAYER_INP_GATE,  "weight", i), {n_embd, n_embd_per_layer}, 0);
            layer.per_layer_proj       = create_tensor(tn(LLM_TENSOR_PER_LAYER_PROJ,      "weight", i), {n_embd_per_layer, n_embd}, 0);
            layer.per_layer_post_norm  = create_tensor(tn(LLM_TENSOR_PER_LAYER_POST_NORM, "weight", i), {n_embd}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

void llama_model_gemma4_assistant::load_arch_hparams(llama_model_loader & ml) {
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);

    uint32_t n_kv_shared_layers = 0;
    ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);

    hparams.n_layer_kv_from_start = hparams.n_layer - (int32_t) n_kv_shared_layers;
    hparams.f_attention_scale     = 1.0f;

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,        hparams.nextn_predict_layers, false);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,    hparams.n_embd_head_k_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,  hparams.n_embd_head_v_swa);

    if (hparams.n_layer == 4) {
        type = LLM_TYPE_31B;
    }
}

void llama_model_gemma4_assistant::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (n_embd_head_k != n_embd_head_v) {
        throw std::runtime_error("Gemma 4 assistant requires n_embd_head_k == n_embd_head_v");
    }
    if (hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa) {
        throw std::runtime_error("Gemma 4 assistant requires n_embd_head_k_swa == n_embd_head_v_swa");
    }
    if (hparams.n_embd_out() == n_embd) {
        throw std::runtime_error("Gemma 4 assistant requires embedding_length_out to carry the target hidden size");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);

    const int64_t n_embd_backbone = hparams.n_embd_out();
    nextn_pre_proj  = create_tensor(tn(LLM_TENSOR_NEXTN_PRE_PROJ,  "weight"), { 2*n_embd_backbone, n_embd }, 0);
    nextn_post_proj = create_tensor(tn(LLM_TENSOR_NEXTN_POST_PROJ, "weight"), { n_embd, n_embd_backbone }, 0);

    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff        = hparams.n_ff(i);

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), { n_embd, n_embd_head*n_head }, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), { n_embd_head*n_head, n_embd }, 0);

        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), { n_embd_head }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, 0);

        layer.out_scale = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), { 1u }, 0);

        if (!hparams.is_swa(i)) {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_embd_head/2 }, rope_freqs_flag);
            rope_freqs_flag = TENSOR_DUPLICATED;
        }

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), { n_embd }, 0);
        layer.ffn_gate      = create_tensor(tn(LLM_TENSOR_FFN_GATE,      "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_up        = create_tensor(tn(LLM_TENSOR_FFN_UP,        "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down      = create_tensor(tn(LLM_TENSOR_FFN_DOWN,      "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), { n_embd }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4_assistant::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gemma4_assistant::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params) {
    GGML_ASSERT(src_mctx  && "Gemma 4 assistant graph requires an MTP source (llama_set_mtp_source)");
    GGML_ASSERT(src_model && "Gemma 4 assistant graph requires a source model");
    GGML_ASSERT(src_model->tok_embd && "source model missing tok_embd");

    const auto & src_hparams = src_model->hparams;

    const int32_t src_layer_full = (int32_t) src_hparams.n_layer - 1;
    const int32_t src_layer_swa  = (int32_t) src_hparams.n_layer - 2;
    GGML_ASSERT(!src_hparams.is_swa(src_layer_full) && "target's last layer must be full attention");
    GGML_ASSERT( src_hparams.is_swa(src_layer_swa)  && "target's penultimate layer must be SWA");

    const int64_t n_embd_backbone = hparams.n_embd_out();

    ggml_tensor * inp_tokens;
    ggml_tensor * inp_h;
    {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd_backbone);

        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        inp_tokens = inp->tokens;
        res->t_inp_tokens = inp->tokens;

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_backbone, ubatch.n_tokens);
        cb(inp->embd, "inp_h", -1);
        ggml_set_input(inp->embd);
        inp_h = inp->embd;
        res->t_inp_embd = inp->embd;

        res->add_input(std::move(inp));
    }

    ggml_tensor * x = ggml_get_rows(ctx0, src_model->tok_embd, inp_tokens);
    x = ggml_scale(ctx0, x, sqrtf((float) n_embd_backbone));
    cb(x, "inp_embd_target", -1);

    ggml_tensor * xh = ggml_concat(ctx0, x, inp_h, 0);
    cb(xh, "inp_xh", -1);

    ggml_tensor * cur = ggml_mul_mat(ctx0, model.nextn_pre_proj, xh);
    cb(cur, "pre_proj", -1);

    auto *        inp_attn    = build_attn_inp_src_kv_iswa();
    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * inpL = cur;

    for (int il = 0; il < n_layer; ++il) {
        const bool    is_swa = hparams.is_swa(il);
        const int32_t il_src = is_swa ? src_layer_swa : src_layer_full;

        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        const int64_t n_head      = hparams.n_head(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        ggml_tensor * cur_norm = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur_norm, "attn_norm", il);

        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur_norm);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        ggml_tensor * freq_factors = is_swa ? nullptr : model.layers[il].rope_freqs;
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_pos", il);

        cur = build_attn(inp_attn, model.layers[il].wo, nullptr, nullptr,
                Qcur, nullptr, nullptr, nullptr, hparams.f_attention_scale, il, il_src);

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur,  inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        cur = build_norm(attn_out, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, attn_out);

        cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
        cb(cur, "out_scaled", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    ggml_tensor * logits = build_lora_mm(model.output, cur);
    cb(logits, "result_output", -1);
    res->t_logits = logits;

    ggml_tensor * h_next = ggml_mul_mat(ctx0, model.nextn_post_proj, cur);
    cb(h_next, "result_h_pre_norm", -1);
    res->t_h_pre_norm = h_next;

    ggml_build_forward_expand(gf, logits);
    ggml_build_forward_expand(gf, h_next);
}

// get 2D slice view from a 3D tensor, the idx corresponds to the 3rd dim
static ggml_tensor * ggml_view_2d_slice(ggml_context * ctx0, ggml_tensor * x, int idx) {
    GGML_ASSERT(idx < (int) x->ne[2]);
    return ggml_view_2d(ctx0, x, x->ne[0], x->ne[1], ggml_row_size(x->type, x->ne[0]),
                        idx * x->ne[0] * x->ne[1] * ggml_element_size(x));
}

llama_model_gemma4::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model),
        n_embd_per_layer(model.hparams.n_embd_per_layer) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);
    cb(inpL, "inp_scaled", -1);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    // TODO: is causal == true correct? might need some changes
    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * inp_per_layer = nullptr;
    if (model.per_layer_tok_embd) {
        inp_per_layer = build_inp_per_layer();
        ggml_build_forward_expand(gf, inp_per_layer);

        // inp_per_layer shape: [n_embd_per_layer, n_tokens, n_layer]
        inp_per_layer = project_per_layer_inputs(inpL, inp_per_layer);
    }

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);
        const int64_t n_ff_exp  = hparams.n_ff_exp;
        llama_infinitum_moe_index_info infinitum_moe_index;

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * freq_factors = nullptr;
        if (!hparams.is_swa(il)) {
            // full_attention layers use rope_freqs for proportional rope
            freq_factors = model.layers[il].rope_freqs;
        }

        // Q projection (shared for both non-KV and KV layers)
        // this is to mirror Gemma4Attention in pytorch code
        ggml_tensor * Qcur;
        {
            Qcur = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
            cb(Qcur, "Qcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_pos", il);
        }

        // self-attention
        if (hparams.has_kv(il)) {
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = model.layers[il].wv
                                    ? build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s)
                                    : Kcur; // if v_proj is not present, use Kcur as Vcur
            cb(Vcur, "Vcur", il);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);

            cb(Kcur, "Kcur_normed", il);
            cb(Vcur, "Vcur_normed", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Kcur, "Kcur_pos", il);

            cur = build_attn(inp_attn, model.layers[il].wo,
                    nullptr, model.layers[il].wo_s, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    hparams.f_attention_scale, il);
        } else {
            // reuse KV cache of earlier layers
            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr, model.layers[il].wo_s,
                    Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        }

        // TODO @ngxson : strip unused token right after the last KV layer to speed up prompt processing
        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = build_norm(cur,
                model.layers[il].attn_post_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        // feed-forward network
        const bool is_moe_layer = model.layers[il].ffn_gate_inp != nullptr;
        if (is_moe_layer) {
            // MLP (shared exp)
            ggml_tensor * cur_mlp = build_norm(attn_out,
                    model.layers[il].ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_norm_1", il);

            cur_mlp = build_ffn(cur_mlp,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cur_mlp = build_norm(cur_mlp,
                    model.layers[il].ffn_post_norm_1, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_mlp", il);

            // Expert FFN
            ggml_tensor * cur_moe = build_norm(attn_out,
                    model.layers[il].ffn_pre_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_norm_2", il);

            // custom MoE logits calculation (router operates on attn_out, not cur)
            ggml_tensor * tmp = ggml_rms_norm(ctx0, attn_out, hparams.f_norm_rms_eps);
            tmp = ggml_scale(ctx0, tmp, 1.0f / sqrtf((float) n_embd));
            tmp = ggml_mul(ctx0, tmp, model.layers[il].ffn_gate_inp_s);
            ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, tmp); // [n_expert, n_tokens]
            cb(logits, "ffn_moe_logits", il);

            ggml_tensor * ffn_gate_up_exps = model.layers[il].ffn_gate_up_exps;
            ggml_tensor * ffn_down_exps = model.layers[il].ffn_down_exps;
            bool q4_slots_done = false;
            if ((ffn_gate_up_exps == nullptr || ffn_down_exps == nullptr) &&
                    gemma4_infinitum_q4_slots_enabled() &&
                    llama_infinitum_moe_ggml_pack_enabled()) {
                const llama_infinitum_moe_index_info & q4_slots_index = gemma4_infinitum_expert_index();
                infinitum_moe_index = q4_slots_index;
                if (q4_slots_index.enabled && q4_slots_index.error.empty()) {
                    const int64_t n_moe_tokens = logits->ne[1];
                    const int64_t n_expert_compute = gemma4_infinitum_top_k_limit(n_expert_used);
                    ggml_tensor * probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
                    cb(probs, "ffn_moe_probs_q4_slots", il);

                    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, probs, n_expert_compute); // [n_expert_compute, n_tokens]
                    cb(selected_experts->src[0], "ffn_moe_argsort_q4_slots", il);
                    cb(selected_experts, "ffn_moe_topk_q4_slots", il);

                    probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_moe_tokens);
                    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_compute, n_tokens]
                    cb(weights, "ffn_moe_weights_q4_slots", il);

                    weights = ggml_reshape_2d(ctx0, weights, n_expert_compute, n_moe_tokens);
                    ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
                    weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
                    weights = ggml_div(ctx0, weights, weights_sum);
                    weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_compute, n_moe_tokens);
                    cb(weights, "ffn_moe_weights_norm_q4_slots", il);

                    if (model.layers[il].ffn_down_exps_s) {
                        ggml_tensor * down_s = ggml_reshape_3d(ctx0, model.layers[il].ffn_down_exps_s, 1, n_expert, 1);
                        down_s = ggml_repeat_4d(ctx0, down_s, 1, n_expert, n_moe_tokens, 1);
                        down_s = ggml_get_rows(ctx0, down_s, selected_experts);
                        weights = ggml_mul(ctx0, weights, down_s);
                        cb(weights, "ffn_moe_weights_down_scaled_q4_slots", il);
                    }

                    ggml_build_forward_expand(gf, weights);
                    auto * userdata = gemma4_infinitum_q4_slots_userdata_for_layer(q4_slots_index, static_cast<int>(il));
                    cur_moe = ggml_map_custom3(ctx0, cur_moe, selected_experts, weights,
                            gemma4_infinitum_q4_slots_op, 1, userdata);
                    cb(cur_moe, "ffn_moe_out_q4_slots", il);
                    q4_slots_done = true;
                } else {
                    GGML_ABORT("Gemma4 Q4 slot-cache requested but LLAMA_INFINITUM_EXPERT_INDEX is invalid");
                }
            }
            if (!q4_slots_done && (ffn_gate_up_exps == nullptr || ffn_down_exps == nullptr) && llama_infinitum_moe_ggml_pack_enabled()) {
                const llama_infinitum_moe_index_info & full_pack_index = gemma4_infinitum_expert_index();
                infinitum_moe_index = full_pack_index;
                if (full_pack_index.enabled && full_pack_index.error.empty()) {
                    const ggml_type pack_type = gemma4_infinitum_ggml_expert_pack_type();
                    ffn_gate_up_exps = llama_infinitum_moe_ggml_pack_tensor_shaped(
                            ctx0, full_pack_index, il, "gate_up", pack_type,
                            n_embd, n_ff_exp * 2, n_expert);
                    ffn_down_exps = llama_infinitum_moe_ggml_pack_tensor_shaped(
                            ctx0, full_pack_index, il, "down", pack_type,
                            n_ff_exp, n_embd, n_expert);
                }
            }
            if (!q4_slots_done && (ffn_gate_up_exps == nullptr || ffn_down_exps == nullptr)) {
                GGML_ABORT("Gemma4 external MoE experts are missing; set LLAMA_INFINITUM_EXPERT_INDEX and LLAMA_INFINITUM_V2_GGML_EXPERT_PACK, or load a full GGUF");
            }

            if (!q4_slots_done) {
                cur_moe = build_moe_ffn(cur_moe,
                        nullptr, // gate_inp
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        ffn_down_exps,
                        nullptr, // exp_probs_b (not used for gemma4)
                        n_expert, n_expert_used,
                        LLM_FFN_GELU, true,
                        1.0f,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        il, logits,
                        ffn_gate_up_exps,
                        model.layers[il].ffn_up_exps_s,
                        model.layers[il].ffn_gate_exps_s,
                        model.layers[il].ffn_down_exps_s);
            }
            cur_moe = build_norm(cur_moe,
                    model.layers[il].ffn_post_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_moe", il);

            cur = ggml_add(ctx0, cur_mlp, cur_moe);
            cb(cur, "ffn_moe_combined", il);
        } else {
            cur = build_norm(attn_out,
                    model.layers[il].ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }
        cur = build_norm(cur,
                model.layers[il].ffn_post_norm, nullptr,
                LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        // residual connection
        cur = ggml_add(ctx0, cur, attn_out);

        // per-layer embedding
        if (inp_per_layer) {
            ggml_tensor * pe_in = cur;
            cb(cur, "pe_in", il);

            cur = build_lora_mm(model.layers[il].per_layer_inp_gate, cur); // [n_embd_per_layer, n_tokens]
            cur = ggml_gelu(ctx0, cur);

            ggml_tensor * inp_this_layer = ggml_view_2d_slice(ctx0, inp_per_layer, il); // [n_embd_per_layer, n_tokens]

            // TODO @ngxson : improve this
            if (il == n_layer - 1 && inp_out_ids) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_out_ids);
            }

            cur = ggml_mul(ctx0, cur, inp_this_layer);
            cur = build_lora_mm(model.layers[il].per_layer_proj, cur); // [n_embd, n_tokens]
            cur = build_norm(cur, model.layers[il].per_layer_post_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "per_layer_embd_out", il);

            // residual connection
            cur = ggml_add(ctx0, pe_in, cur);
        }

        // layer_scalar
        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    res->t_h_pre_norm = cur;

    cur = build_norm(cur,
            model.output_norm, nullptr,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// equivalent to get_per_layer_inputs() in python code
// output shape: [n_embd_per_layer, n_layer, n_tokens]
ggml_tensor * llama_model_gemma4::graph::build_inp_per_layer() {
    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    ggml_tensor * inp_per_layer;
    float tok_embd_scale = sqrtf((float) n_embd_per_layer);
    if (ubatch.token) {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;

        inp_per_layer = ggml_get_rows  (ctx0, model.per_layer_tok_embd, inp->tokens);
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, n_tokens);
        inp_per_layer = ggml_scale     (ctx0, inp_per_layer, tok_embd_scale);
        cb(inp_per_layer, "inp_per_layer_selected", -1);

        res->add_input(std::move(inp));
    } else {
        // Multimodal embedding path: use padding token (ID=0) embedding
        // TODO: verify if this is the correct behavior in transformers implementation
        const int64_t embd_size = model.per_layer_tok_embd->ne[0];  // n_embd_per_layer * n_layer

        // Extract and dequantize padding token embedding (row 0)
        ggml_tensor * padding = ggml_view_1d(ctx0, model.per_layer_tok_embd, embd_size, 0);
        inp_per_layer = ggml_cast (ctx0, padding, GGML_TYPE_F32);
        inp_per_layer = ggml_scale(ctx0, inp_per_layer, tok_embd_scale);

        // Reshape to [n_embd_per_layer, n_layer, 1]
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, 1);
        cb(inp_per_layer, "inp_per_layer_multimodal", -1);
    }
    return inp_per_layer;
}

// equivalent to project_per_layer_inputs() in python code
// this calculates the per-layer inputs, so the final tensor shape will have n_layer as the last dim
// inp_batch     shape: [n_embd, n_tokens]
// inp_per_layer shape: [n_embd_per_layer, n_layer, n_tokens] (from build_inp_per_layer)
// output shape: [n_embd_per_layer, n_tokens, n_layer]
ggml_tensor * llama_model_gemma4::graph::project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer) {
    const float per_layer_projection_scale = 1.0f / sqrtf((float) n_embd);
    const float per_layer_input_scale      = 1.0f / sqrtf(2.0f);

    // note: this matrix multiplication will be performed in the input layer (i.e. on the CPU)
    ggml_tensor * per_layer_proj;
    per_layer_proj = ggml_mul_mat   (ctx0, model.per_layer_model_proj, inp_batch);
    per_layer_proj = ggml_scale     (ctx0, per_layer_proj, per_layer_projection_scale);
    per_layer_proj = ggml_reshape_3d(ctx0, per_layer_proj, n_embd_per_layer, n_layer, n_tokens);

    per_layer_proj = build_norm(per_layer_proj, model.per_layer_proj_norm, nullptr, LLM_NORM_RMS, -1);
    cb(per_layer_proj, "per_layer_proj", -1);

    inp_per_layer = ggml_add  (ctx0, per_layer_proj, inp_per_layer);
    inp_per_layer = ggml_scale(ctx0, inp_per_layer, per_layer_input_scale);
    cb(inp_per_layer, "inp_per_layer", -1);

    // permute to shape: [n_embd_per_layer, n_tokens, n_layer]
    inp_per_layer = ggml_cont(ctx0, ggml_permute(ctx0, inp_per_layer, 0, 2, 1, 3));
    return inp_per_layer;
}
