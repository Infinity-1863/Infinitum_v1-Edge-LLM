#include "models.h"
#include "llama-infinitum-moe.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

static bool llama_openai_moe_infinitum_selective_enabled() {
    return llama_infinitum_moe_selective_enabled();
}

static bool llama_openai_moe_infinitum_profile_enabled() {
    const char * value = std::getenv("LLAMA_INFINITUM_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool llama_openai_moe_infinitum_pack_slots_prefill_enabled() {
    const char * value = std::getenv("LLAMA_INFINITUM_V2_GGML_EXPERT_PACK_SLOTS_PREFILL");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool llama_openai_moe_infinitum_ggml_pack_prefetch_enabled() {
    const char * value = std::getenv("LLAMA_INFINITUM_GGML_PACK_PREFETCH");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static int llama_openai_moe_infinitum_ggml_pack_prefetch_max_experts() {
    const char * value = std::getenv("LLAMA_INFINITUM_GGML_PACK_PREFETCH_MAX_EXPERTS");
    if (value == nullptr || value[0] == '\0') {
        return 4;
    }
    const int parsed = std::atoi(value);
    if (parsed <= 0) {
        return 4;
    }
    return std::min(parsed, 64);
}

static double llama_openai_moe_elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static std::string llama_openai_moe_int_vector_json(const std::vector<int> & values) {
    std::string json = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            json += ",";
        }
        json += std::to_string(values[i]);
    }
    json += "]";
    return json;
}

enum class llama_openai_moe_infinitum_predictor_mode {
    recent,
    learned,
    off,
};

static llama_openai_moe_infinitum_predictor_mode llama_openai_moe_infinitum_expert_predictor_mode() {
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_PREDICTOR");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "recent") == 0) {
        return llama_openai_moe_infinitum_predictor_mode::recent;
    }
    if (std::strcmp(value, "learned") == 0 || std::strcmp(value, "online") == 0) {
        return llama_openai_moe_infinitum_predictor_mode::learned;
    }
    if (std::strcmp(value, "off") == 0 || std::strcmp(value, "none") == 0 || std::strcmp(value, "0") == 0) {
        return llama_openai_moe_infinitum_predictor_mode::off;
    }
    return llama_openai_moe_infinitum_predictor_mode::recent;
}

static int llama_openai_moe_infinitum_expert_predictor_top_k() {
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_PREDICTOR_TOP_K");
    if (value == nullptr || value[0] == '\0') {
        return 4;
    }
    const int parsed = std::atoi(value);
    if (parsed <= 0) {
        return 4;
    }
    return std::min(parsed, 16);
}

static int llama_openai_moe_infinitum_expert_predictor_lookahead() {
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_PREDICTOR_LOOKAHEAD");
    if (value == nullptr || value[0] == '\0') {
        return 1;
    }
    const int parsed = std::atoi(value);
    if (parsed <= 0) {
        return 1;
    }
    return std::min(parsed, 4);
}

static int llama_openai_moe_infinitum_expert_count_from_index(const llama_infinitum_moe_index_info & expert_index) {
    static std::atomic<int> cached_expert_count { 0 };
    const int cached = cached_expert_count.load(std::memory_order_relaxed);
    if (cached > 0) {
        return cached;
    }
    int expert_count = 0;
    for (const llama_infinitum_moe_expert_slice & slice : expert_index.slices) {
        if (slice.expert_id >= 0) {
            expert_count = std::max(expert_count, slice.expert_id + 1);
        }
    }
    if (expert_count <= 0) {
        expert_count = 128;
    }
    expert_count = std::min(expert_count, 512);
    cached_expert_count.store(expert_count, std::memory_order_relaxed);
    return expert_count;
}

struct llama_openai_moe_infinitum_learned_predictor_edge {
    int expert_count = 0;
    std::vector<std::uint16_t> transition_counts;
    std::vector<std::uint32_t> hot_counts;
    std::uint64_t observations = 0;

    void ensure(int count) {
        if (expert_count == count && transition_counts.size() == std::size_t(count) * std::size_t(count) &&
                hot_counts.size() == std::size_t(count)) {
            return;
        }
        expert_count = count;
        transition_counts.assign(std::size_t(count) * std::size_t(count), 0);
        hot_counts.assign(std::size_t(count), 0);
        observations = 0;
    }

    void decay_if_needed() {
        ++observations;
        if ((observations & 1023ull) != 0ull) {
            return;
        }
        for (std::uint16_t & value : transition_counts) {
            value = static_cast<std::uint16_t>(value >> 1);
        }
        for (std::uint32_t & value : hot_counts) {
            value >>= 1;
        }
    }

    std::uint16_t transition(int source_expert, int target_expert) const {
        return transition_counts[std::size_t(source_expert) * std::size_t(expert_count) + std::size_t(target_expert)];
    }

    void add_transition(int source_expert, int target_expert) {
        std::uint16_t & value = transition_counts[std::size_t(source_expert) * std::size_t(expert_count) + std::size_t(target_expert)];
        if (value != UINT16_MAX) {
            ++value;
        }
    }
};

struct llama_openai_moe_infinitum_learned_predictor_state {
    std::mutex mutex;
    int expert_count = 0;
    std::array<std::vector<int>, 64> selected_by_layer = {};
    std::array<std::vector<int>, 64> predicted_by_layer = {};
    std::array<llama_openai_moe_infinitum_learned_predictor_edge, 64> next_layer_edges = {};

    void ensure_expert_count(int count) {
        if (expert_count == count) {
            return;
        }
        expert_count = count;
        selected_by_layer = {};
        predicted_by_layer = {};
        for (llama_openai_moe_infinitum_learned_predictor_edge & edge : next_layer_edges) {
            edge.ensure(count);
        }
    }
};

static llama_openai_moe_infinitum_learned_predictor_state & llama_openai_moe_infinitum_learned_predictor_state_get() {
    static llama_openai_moe_infinitum_learned_predictor_state state;
    return state;
}

static bool llama_openai_moe_infinitum_append_unique(std::vector<int> & target, int expert_id, int limit) {
    if (expert_id < 0 || static_cast<int>(target.size()) >= limit) {
        return false;
    }
    if (std::find(target.begin(), target.end(), expert_id) != target.end()) {
        return false;
    }
    target.push_back(expert_id);
    return true;
}

static void llama_openai_moe_infinitum_append_unique_limited(
        std::vector<int> & target,
        const std::vector<int> & source,
        int limit) {
    for (const int expert_id : source) {
        if (static_cast<int>(target.size()) >= limit) {
            break;
        }
        llama_openai_moe_infinitum_append_unique(target, expert_id, limit);
    }
}

static void llama_openai_moe_infinitum_truncate(std::vector<int> & values, int limit) {
    if (limit > 0 && static_cast<int>(values.size()) > limit) {
        values.resize(static_cast<std::size_t>(limit));
    }
}

static int llama_openai_moe_infinitum_intersection_count(
        const std::vector<int> & predicted,
        const std::vector<int> & selected) {
    int hits = 0;
    for (const int expert_id : selected) {
        if (std::find(predicted.begin(), predicted.end(), expert_id) != predicted.end()) {
            ++hits;
        }
    }
    return hits;
}

static std::vector<int> llama_openai_moe_infinitum_predict_from_edge_locked(
        const llama_openai_moe_infinitum_learned_predictor_edge & edge,
        const std::vector<int> & source_selected,
        int limit) {
    std::vector<int> predicted;
    if (edge.expert_count <= 0 || source_selected.empty() || limit <= 0) {
        return predicted;
    }

    std::vector<std::pair<std::uint64_t, int>> scored;
    scored.reserve(std::size_t(edge.expert_count));
    for (int target_expert = 0; target_expert < edge.expert_count; ++target_expert) {
        std::uint64_t score = edge.hot_counts[std::size_t(target_expert)];
        for (const int source_expert : source_selected) {
            if (source_expert >= 0 && source_expert < edge.expert_count) {
                score += std::uint64_t(edge.transition(source_expert, target_expert)) * 16ull;
            }
        }
        if (score > 0) {
            scored.emplace_back(score, target_expert);
        }
    }

    std::sort(scored.begin(), scored.end(), [](const auto & a, const auto & b) {
        if (a.first != b.first) {
            return a.first > b.first;
        }
        return a.second < b.second;
    });
    for (const auto & item : scored) {
        if (static_cast<int>(predicted.size()) >= limit) {
            break;
        }
        predicted.push_back(item.second);
    }
    return predicted;
}

struct llama_openai_moe_infinitum_prefetch_prediction {
    std::vector<int> predicted_current;
    std::vector<int> next_1;
    std::vector<int> next_2;
    int current_hits = 0;
};

static llama_openai_moe_infinitum_prefetch_prediction llama_openai_moe_infinitum_prefetch_learned_next_layers(
        const llama_infinitum_moe_index_info & expert_index,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        const std::vector<int> & current_selected) {
    llama_openai_moe_infinitum_prefetch_prediction prediction;
    if (!llama_infinitum_moe_prefetch_enabled() || layer_index < 0 || layer_index >= 64) {
        return prediction;
    }
    const llama_openai_moe_infinitum_predictor_mode mode = llama_openai_moe_infinitum_expert_predictor_mode();
    if (mode == llama_openai_moe_infinitum_predictor_mode::off) {
        return prediction;
    }
    const int expert_count = llama_openai_moe_infinitum_expert_count_from_index(expert_index);
    const int predictor_top_k = llama_openai_moe_infinitum_expert_predictor_top_k();
    const int predictor_lookahead = llama_openai_moe_infinitum_expert_predictor_lookahead();

    llama_openai_moe_infinitum_learned_predictor_state & state = llama_openai_moe_infinitum_learned_predictor_state_get();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.ensure_expert_count(expert_count);
        prediction.predicted_current = state.predicted_by_layer[std::size_t(layer_index)];
        prediction.current_hits = llama_openai_moe_infinitum_intersection_count(
            prediction.predicted_current, current_selected);
        if (layer_index + 1 < static_cast<int>(state.selected_by_layer.size())) {
            if (mode == llama_openai_moe_infinitum_predictor_mode::learned) {
                prediction.next_1 = llama_openai_moe_infinitum_predict_from_edge_locked(
                    state.next_layer_edges[std::size_t(layer_index)], current_selected, predictor_top_k);
                llama_openai_moe_infinitum_append_unique_limited(
                    prediction.next_1, state.selected_by_layer[std::size_t(layer_index + 1)], predictor_top_k);
            } else {
                prediction.next_1 = state.selected_by_layer[std::size_t(layer_index + 1)];
            }
        }
        if (predictor_lookahead >= 2 && layer_index + 2 < static_cast<int>(state.selected_by_layer.size())) {
            if (mode == llama_openai_moe_infinitum_predictor_mode::learned) {
                prediction.next_2 = llama_openai_moe_infinitum_predict_from_edge_locked(
                    state.next_layer_edges[std::size_t(layer_index + 1)], prediction.next_1, predictor_top_k);
                llama_openai_moe_infinitum_append_unique_limited(
                    prediction.next_2, state.selected_by_layer[std::size_t(layer_index + 2)], predictor_top_k);
            } else {
                prediction.next_2 = state.selected_by_layer[std::size_t(layer_index + 2)];
            }
        }
        if (prediction.next_1.empty()) {
            llama_openai_moe_infinitum_append_unique_limited(prediction.next_1, current_selected, predictor_top_k);
        }
        if (predictor_lookahead >= 2 && prediction.next_2.empty()) {
            llama_openai_moe_infinitum_append_unique_limited(prediction.next_2, prediction.next_1, predictor_top_k);
        }
        llama_openai_moe_infinitum_truncate(prediction.next_1, predictor_top_k);
        llama_openai_moe_infinitum_truncate(prediction.next_2, predictor_top_k);
        if (!prediction.next_1.empty() && layer_index + 1 < static_cast<int>(state.predicted_by_layer.size())) {
            state.predicted_by_layer[std::size_t(layer_index + 1)] = prediction.next_1;
        }
        if (!prediction.next_2.empty() && layer_index + 2 < static_cast<int>(state.predicted_by_layer.size())) {
            state.predicted_by_layer[std::size_t(layer_index + 2)] = prediction.next_2;
        }
    }
    if (!prediction.next_1.empty()) {
        llama_infinitum_moe_prefetch_selected_experts_async(expert_index, cache, layer_index + 1, prediction.next_1);
    }
    if (!prediction.next_2.empty()) {
        llama_infinitum_moe_prefetch_selected_experts_async(expert_index, cache, layer_index + 2, prediction.next_2);
    }
    return prediction;
}

static void llama_openai_moe_infinitum_learned_predictor_record(
        const llama_infinitum_moe_index_info & expert_index,
        int layer_index,
        const std::vector<int> & selected) {
    if (!llama_infinitum_moe_prefetch_enabled() || layer_index < 0 || layer_index >= 64) {
        return;
    }
    if (llama_openai_moe_infinitum_expert_predictor_mode() == llama_openai_moe_infinitum_predictor_mode::off) {
        return;
    }
    const int expert_count = llama_openai_moe_infinitum_expert_count_from_index(expert_index);
    llama_openai_moe_infinitum_learned_predictor_state & state = llama_openai_moe_infinitum_learned_predictor_state_get();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.ensure_expert_count(expert_count);
    if (layer_index > 0) {
        const std::vector<int> & previous_layer_selected = state.selected_by_layer[std::size_t(layer_index - 1)];
        if (!previous_layer_selected.empty()) {
            llama_openai_moe_infinitum_learned_predictor_edge & edge =
                state.next_layer_edges[std::size_t(layer_index - 1)];
            edge.ensure(expert_count);
            edge.decay_if_needed();
            for (const int target_expert : selected) {
                if (target_expert < 0 || target_expert >= expert_count) {
                    continue;
                }
                std::uint32_t & hot = edge.hot_counts[std::size_t(target_expert)];
                if (hot != UINT32_MAX) {
                    ++hot;
                }
                for (const int source_expert : previous_layer_selected) {
                    if (source_expert >= 0 && source_expert < expert_count) {
                        edge.add_transition(source_expert, target_expert);
                    }
                }
            }
        }
    }
    state.selected_by_layer[std::size_t(layer_index)] = selected;
}

struct llama_openai_moe_infinitum_external_mlp_userdata {
    const llama_infinitum_moe_index_info * expert_index = nullptr;
    llama_infinitum_moe_slice_cache * cache = nullptr;
    int layer_index = -1;
    int expert_workers = 3;
};

struct llama_openai_moe_infinitum_prefetch_userdata {
    const llama_infinitum_moe_index_info * expert_index = nullptr;
    llama_infinitum_moe_slice_cache * cache = nullptr;
    int layer_index = -1;
    int lookahead = 1;
    int selected_expert_count = 0;
};

static llama_openai_moe_infinitum_external_mlp_userdata * llama_openai_moe_infinitum_external_mlp_userdata_for_layer(
        const llama_infinitum_moe_index_info & expert_index,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        int expert_workers) {
    static std::array<llama_openai_moe_infinitum_external_mlp_userdata, 64> userdata_by_layer = {};
    if (layer_index < 0 || layer_index >= int(userdata_by_layer.size())) {
        GGML_ABORT("LLAMA_INFINITUM_SELECTIVE_MOE external MLP layer index exceeds userdata cache");
    }
    llama_openai_moe_infinitum_external_mlp_userdata & userdata = userdata_by_layer[std::size_t(layer_index)];
    userdata.expert_index = &expert_index;
    userdata.cache = &cache;
    userdata.layer_index = layer_index;
    userdata.expert_workers = expert_workers;
    return &userdata;
}

static llama_openai_moe_infinitum_prefetch_userdata * llama_openai_moe_infinitum_prefetch_userdata_for_layer(
        const llama_infinitum_moe_index_info & expert_index,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        int lookahead = 1,
        int selected_expert_count = 0) {
    static std::array<llama_openai_moe_infinitum_prefetch_userdata, 64> userdata_by_layer = {};
    if (layer_index < 0 || layer_index >= int(userdata_by_layer.size())) {
        GGML_ABORT("LLAMA_INFINITUM_GGML_PACK_PREFETCH layer index exceeds userdata cache");
    }
    llama_openai_moe_infinitum_prefetch_userdata & userdata = userdata_by_layer[std::size_t(layer_index)];
    userdata.expert_index = &expert_index;
    userdata.cache = &cache;
    userdata.layer_index = layer_index;
    userdata.lookahead = std::max(1, std::min(lookahead, 4));
    userdata.selected_expert_count = std::max(0, selected_expert_count);
    return &userdata;
}

static const llama_infinitum_moe_index_info & llama_openai_moe_infinitum_expert_index() {
    static const llama_infinitum_moe_index_info index = llama_infinitum_moe_index_from_env();
    return index;
}

static int llama_openai_moe_infinitum_expert_workers() {
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_WORKERS");
    if (value == nullptr || value[0] == '\0') {
#if defined(__ANDROID__)
        return 4;
#else
        return 6;
#endif
    }
    return std::max(1, std::atoi(value));
}

static int64_t llama_openai_moe_infinitum_expert_top_k_limit(int layer_index, int64_t n_expert_used) {
    const char * layer_values = std::getenv("LLAMA_INFINITUM_EXPERT_TOP_K_LAYERS");
    if (layer_values != nullptr && layer_values[0] != '\0') {
        const char * ptr = layer_values;
        for (int layer = 0; layer <= layer_index && *ptr != '\0'; ++layer) {
            char * end = nullptr;
            const long parsed = std::strtol(ptr, &end, 10);
            if (end == ptr) {
                break;
            }
            if (layer == layer_index && parsed > 0) {
                return std::max<int64_t>(1, std::min<int64_t>(n_expert_used, parsed));
            }
            ptr = end;
            while (*ptr == ',' || *ptr == ';' || *ptr == ' ') {
                ++ptr;
            }
        }
    }
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_TOP_K");
    if (value == nullptr || value[0] == '\0') {
        return n_expert_used;
    }
    const int parsed = std::atoi(value);
    if (parsed <= 0) {
        return n_expert_used;
    }
    return std::max<int64_t>(1, std::min<int64_t>(n_expert_used, parsed));
}

static int64_t llama_openai_moe_infinitum_resident_expert_top_k_limit(int layer_index, int64_t n_expert_used) {
    const char * layer_values = std::getenv("LLAMA_INFINITUM_RESIDENT_EXPERT_TOP_K_LAYERS");
    if (layer_values != nullptr && layer_values[0] != '\0') {
        const char * ptr = layer_values;
        for (int layer = 0; layer <= layer_index && *ptr != '\0'; ++layer) {
            char * end = nullptr;
            const long parsed = std::strtol(ptr, &end, 10);
            if (end == ptr) {
                break;
            }
            if (layer == layer_index && parsed > 0) {
                return std::max<int64_t>(1, std::min<int64_t>(n_expert_used, parsed));
            }
            ptr = end;
            while (*ptr == ',' || *ptr == ';' || *ptr == ' ') {
                ++ptr;
            }
        }
    }
    const char * value = std::getenv("LLAMA_INFINITUM_RESIDENT_EXPERT_TOP_K");
    if (value == nullptr || value[0] == '\0') {
        return n_expert_used;
    }
    const int parsed = std::atoi(value);
    if (parsed <= 0) {
        return n_expert_used;
    }
    return std::max<int64_t>(1, std::min<int64_t>(n_expert_used, parsed));
}

static float llama_openai_moe_infinitum_env_float(const char * name, float fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value || !std::isfinite(parsed)) {
        return fallback;
    }
    return parsed;
}

static int64_t llama_openai_moe_infinitum_env_i64(const char * name, int64_t fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return fallback;
    }
    return int64_t(parsed);
}

static int64_t llama_openai_moe_infinitum_adaptive_top_k_limit(
        int64_t n_expert_limit,
        const std::vector<float> & weights) {
    const float mass_threshold = llama_openai_moe_infinitum_env_float("LLAMA_INFINITUM_EXPERT_TOP_K_MASS", 0.0f);
    if (mass_threshold <= 0.0f || mass_threshold >= 1.0f || n_expert_limit <= 1) {
        return n_expert_limit;
    }

    const int64_t min_top_k = std::max<int64_t>(
            1,
            std::min<int64_t>(
                n_expert_limit,
                llama_openai_moe_infinitum_env_i64("LLAMA_INFINITUM_EXPERT_MIN_TOP_K", 2)));
    float total = 0.0f;
    for (int64_t i = 0; i < n_expert_limit && i < int64_t(weights.size()); ++i) {
        const float weight = weights[std::size_t(i)];
        if (std::isfinite(weight) && weight > 0.0f) {
            total += weight;
        }
    }
    if (total <= 0.0f || !std::isfinite(total)) {
        return n_expert_limit;
    }

    float cumulative = 0.0f;
    for (int64_t i = 0; i < n_expert_limit && i < int64_t(weights.size()); ++i) {
        const float weight = weights[std::size_t(i)];
        if (std::isfinite(weight) && weight > 0.0f) {
            cumulative += weight;
        }
        const int64_t top_k = i + 1;
        if (top_k >= min_top_k && cumulative / total >= mass_threshold) {
            return top_k;
        }
    }
    return n_expert_limit;
}

static float llama_openai_moe_tensor_f32(const ggml_tensor * tensor, int64_t i0, int64_t i1, int64_t i2) {
    const char * base = static_cast<const char *>(tensor->data);
    const char * ptr = base + i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2];
    return *reinterpret_cast<const float *>(ptr);
}

static int llama_openai_moe_tensor_i32(const ggml_tensor * tensor, int64_t i0, int64_t i1) {
    const char * base = static_cast<const char *>(tensor->data);
    const char * ptr = base + i0 * tensor->nb[0] + i1 * tensor->nb[1];
    return *reinterpret_cast<const int32_t *>(ptr);
}

static void llama_openai_moe_infinitum_prefetch_selected_op(
        ggml_tensor * dst,
        const ggml_tensor * selected_tensor,
        int ith,
        int /*nth*/,
        void * userdata) {
    if (ith != 0) {
        return;
    }
    if (dst->type != GGML_TYPE_I32 || selected_tensor->type != GGML_TYPE_I32) {
        GGML_ABORT("LLAMA_INFINITUM_GGML_PACK_PREFETCH selected expert tensor must be I32");
    }
    std::memcpy(dst->data, selected_tensor->data, ggml_nbytes(selected_tensor));
    auto * data = static_cast<llama_openai_moe_infinitum_prefetch_userdata *>(userdata);
    if (data == nullptr || data->expert_index == nullptr || data->cache == nullptr) {
        GGML_ABORT("LLAMA_INFINITUM_GGML_PACK_PREFETCH custom op missing userdata");
    }

    std::vector<int> selected;
    std::vector<int> router_selected;
    const int64_t n_expert_used = selected_tensor->ne[0];
    const int64_t n_tokens = std::max<int64_t>(1, selected_tensor->ne[1]);
    const int max_prefetch_experts = llama_openai_moe_infinitum_ggml_pack_prefetch_max_experts();
    const int router_expert_count = data->selected_expert_count > 0 ?
        std::min<int>(data->selected_expert_count, static_cast<int>(n_expert_used)) :
        static_cast<int>(n_expert_used);
    selected.reserve(std::size_t(max_prefetch_experts));
    router_selected.reserve(std::size_t(router_expert_count));
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t k = 0; k < n_expert_used; ++k) {
            const int expert_id = llama_openai_moe_tensor_i32(dst, k, token);
            if (k < router_expert_count) {
                llama_openai_moe_infinitum_append_unique(router_selected, expert_id, router_expert_count);
            }
            if (static_cast<int>(selected.size()) < max_prefetch_experts) {
                llama_openai_moe_infinitum_append_unique(selected, expert_id, max_prefetch_experts);
            }
        }
        if (!router_selected.empty()) {
            break;
        }
    }
    const std::vector<int> & predictor_input = !router_selected.empty() ? router_selected : selected;
    if (!predictor_input.empty()) {
        llama_openai_moe_infinitum_prefetch_learned_next_layers(
            *data->expert_index, *data->cache, data->layer_index, predictor_input);
        llama_openai_moe_infinitum_learned_predictor_record(
            *data->expert_index, data->layer_index, predictor_input);
    }
}

static void llama_openai_moe_infinitum_external_mlp_op(
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
        GGML_ABORT("LLAMA_INFINITUM_SELECTIVE_MOE external MLP custom op received unsupported tensor types");
    }
    auto * data = static_cast<llama_openai_moe_infinitum_external_mlp_userdata *>(userdata);
    if (data == nullptr || data->expert_index == nullptr || data->cache == nullptr) {
        GGML_ABORT("LLAMA_INFINITUM_SELECTIVE_MOE external MLP custom op missing userdata");
    }

    const int hidden_size = static_cast<int>(hidden_tensor->ne[0]);
    if (hidden_size <= 0 ||
            (data->expert_index->hidden_size > 0 && data->expert_index->hidden_size != hidden_size)) {
        GGML_ABORT("LLAMA_INFINITUM_SELECTIVE_MOE external MLP hidden size does not match expert index");
    }
    const int64_t n_tokens = std::max<int64_t>(1, hidden_tensor->ne[1]);
    const int64_t n_expert_used = selected_tensor->ne[0];
    const int64_t n_expert_limit = llama_openai_moe_infinitum_expert_top_k_limit(data->layer_index, n_expert_used);
    thread_local std::vector<float> hidden_scratch;
    thread_local std::vector<float> output_scratch;
    thread_local std::vector<int> selected_scratch;
    thread_local std::vector<float> weights_scratch;
    thread_local std::vector<float> raw_weights_scratch;
    std::vector<float> & hidden = hidden_scratch;
    std::vector<float> & output = output_scratch;
    std::vector<int> & selected = selected_scratch;
    std::vector<float> & weights = weights_scratch;
    std::vector<float> & raw_weights = raw_weights_scratch;
    selected.reserve(std::size_t(n_expert_limit));
    weights.reserve(std::size_t(n_expert_limit));
    raw_weights.reserve(std::size_t(n_expert_limit));
    const bool profile_enabled = llama_openai_moe_infinitum_profile_enabled();

    for (int64_t token = 0; token < n_tokens; ++token) {
        const auto custom_op_start = std::chrono::steady_clock::now();
        const float * hidden_data = nullptr;
        if (hidden_tensor->nb[0] == static_cast<int64_t>(sizeof(float))) {
            const char * base = static_cast<const char *>(hidden_tensor->data);
            hidden_data = reinterpret_cast<const float *>(base + token * hidden_tensor->nb[1]);
        } else {
            hidden.resize(std::size_t(hidden_size));
            for (int i = 0; i < hidden_size; ++i) {
                hidden[i] = llama_openai_moe_tensor_f32(hidden_tensor, i, token, 0);
            }
            hidden_data = hidden.data();
        }
        float * output_data = nullptr;
        const bool output_contiguous = dst->nb[0] == static_cast<int64_t>(sizeof(float));
        if (output_contiguous) {
            char * base = static_cast<char *>(dst->data);
            output_data = reinterpret_cast<float *>(base + token * dst->nb[1]);
        } else {
            output.resize(std::size_t(hidden_size));
            output_data = output.data();
        }
        const auto custom_op_after_input = std::chrono::steady_clock::now();
        selected.clear();
        weights.clear();
        raw_weights.clear();
        for (int64_t k = 0; k < n_expert_limit; ++k) {
            raw_weights.push_back(llama_openai_moe_tensor_f32(weights_tensor, 0, k, token));
        }
        const int64_t n_expert_compute = llama_openai_moe_infinitum_adaptive_top_k_limit(n_expert_limit, raw_weights);
        float weight_sum = 0.0f;
        for (int64_t k = 0; k < n_expert_compute; ++k) {
            selected.push_back(llama_openai_moe_tensor_i32(selected_tensor, k, token));
            const float weight = raw_weights[std::size_t(k)];
            weights.push_back(weight);
            weight_sum += weight;
        }
        if (weight_sum > 0.0f && std::isfinite(weight_sum)) {
            for (float & weight : weights) {
                weight /= weight_sum;
            }
        }
        const auto custom_op_after_selection = std::chrono::steady_clock::now();
        const llama_openai_moe_infinitum_prefetch_prediction prefetch_prediction =
            llama_openai_moe_infinitum_prefetch_learned_next_layers(
                *data->expert_index, *data->cache, data->layer_index, selected);
        const auto custom_op_after_prefetch_submit = std::chrono::steady_clock::now();
        const llama_infinitum_moe_expert_mlp_result result =
            llama_infinitum_moe_execute_selected_experts_into(
                *data->expert_index, *data->cache, data->layer_index, selected, weights,
                hidden_data, hidden_size, output_data, data->expert_workers);
        llama_openai_moe_infinitum_learned_predictor_record(*data->expert_index, data->layer_index, selected);
        if (!result.ok) {
            const std::string message = "LLAMA_INFINITUM_SELECTIVE_MOE external MLP custom op failed: " + result.error;
            GGML_ABORT("%s", message.c_str());
        }
        const auto custom_op_after_compute = std::chrono::steady_clock::now();
        if (!output_contiguous) {
            for (int i = 0; i < hidden_size; ++i) {
                char * base = static_cast<char *>(dst->data);
                char * ptr = base + i * dst->nb[0] + token * dst->nb[1];
                std::memcpy(ptr, &output_data[i], sizeof(float));
            }
        }
        const auto custom_op_end = std::chrono::steady_clock::now();
        if (profile_enabled) {
            const double total_custom_op_ms = llama_openai_moe_elapsed_ms(custom_op_start, custom_op_end);
            const double input_ms = llama_openai_moe_elapsed_ms(custom_op_start, custom_op_after_input);
            const double selection_ms = llama_openai_moe_elapsed_ms(custom_op_after_input, custom_op_after_selection);
            const double prefetch_submit_ms = llama_openai_moe_elapsed_ms(custom_op_after_selection, custom_op_after_prefetch_submit);
            const double output_copy_ms = llama_openai_moe_elapsed_ms(custom_op_after_compute, custom_op_end);
            const std::string selected_ids_json = llama_openai_moe_int_vector_json(selected);
            const std::string predicted_current_json = llama_openai_moe_int_vector_json(prefetch_prediction.predicted_current);
            const std::string predicted_next_1_json = llama_openai_moe_int_vector_json(prefetch_prediction.next_1);
            const std::string predicted_next_2_json = llama_openai_moe_int_vector_json(prefetch_prediction.next_2);
            const int predicted_current_count = static_cast<int>(prefetch_prediction.predicted_current.size());
            const int predicted_current_waste = std::max(0, predicted_current_count - prefetch_prediction.current_hits);
            std::fprintf(stderr,
                "LLAMA_INFINITUM_PROFILE {"
                "\"layer\":%d,\"token\":%lld,\"router_ms\":0.0,"
                "\"router_ms_note\":\"not available inside custom op; selected tensor is already materialized\","
                "\"cache_hits_delta\":%llu,\"cache_misses_delta\":%llu,\"cache_evictions_delta\":%llu,"
                "\"gpu_slot_hits_delta\":%llu,\"gpu_slot_misses_delta\":%llu,\"gpu_slot_evictions_delta\":%llu,"
                "\"slice_cache_hits_delta\":%llu,\"slice_cache_misses_delta\":%llu,"
                "\"loaded_bytes_delta\":%llu,\"touched_bytes_delta\":%llu,"
                "\"mapped_bytes_delta\":%llu,\"copied_bytes_delta\":%llu,\"prepacked_bytes_delta\":%llu,"
                "\"resident_bytes\":%llu,\"process_resident_bytes\":%llu,"
                "\"input_ms\":%.3f,\"selection_ms\":%.3f,\"prefetch_submit_ms\":%.3f,"
                "\"load_ms\":%.3f,\"expert_upload_ms\":%.3f,"
                "\"graph_input_ms\":%.3f,\"graph_compute_ms\":%.3f,\"graph_output_ms\":%.3f,"
                "\"gate_up_ms\":%.3f,\"activation_ms\":%.3f,\"down_ms\":%.3f,"
                "\"accumulate_ms\":%.3f,"
                "\"total_compute_ms\":%.3f,\"output_copy_ms\":%.3f,\"total_custom_op_ms\":%.3f,"
                "\"expert_workers\":%d,\"selected_experts\":%lld,\"backend\":\"%s\",\"fallback\":%d,"
                "\"prediction_hits\":%d,\"prediction_count\":%d,\"prediction_waste\":%d,"
                "\"prefetch_next1_count\":%zu,\"prefetch_next2_count\":%zu,"
                "\"selected_ids\":%s,\"predicted_current_ids\":%s,\"prefetch_next1_ids\":%s,\"prefetch_next2_ids\":%s}\n",
                data->layer_index,
                static_cast<long long>(token),
                static_cast<unsigned long long>(result.cache_hits_delta),
                static_cast<unsigned long long>(result.cache_misses_delta),
                static_cast<unsigned long long>(result.cache_evictions_delta),
                static_cast<unsigned long long>(result.gpu_slot_hits_delta),
                static_cast<unsigned long long>(result.gpu_slot_misses_delta),
                static_cast<unsigned long long>(result.gpu_slot_evictions_delta),
                static_cast<unsigned long long>(result.cache_hits_delta),
                static_cast<unsigned long long>(result.cache_misses_delta),
                static_cast<unsigned long long>(result.loaded_bytes_delta),
                static_cast<unsigned long long>(result.touched_bytes_delta),
                static_cast<unsigned long long>(result.mapped_bytes_delta),
                static_cast<unsigned long long>(result.copied_bytes_delta),
                static_cast<unsigned long long>(result.prepacked_bytes_delta),
                static_cast<unsigned long long>(result.resident_bytes),
                static_cast<unsigned long long>(result.process_resident_bytes),
                input_ms,
                selection_ms,
                prefetch_submit_ms,
                result.load_ms,
                result.expert_upload_ms,
                result.graph_input_ms,
                result.graph_compute_ms,
                result.graph_output_ms,
                result.gate_up_ms,
                result.activation_ms,
                result.down_ms,
                result.accumulate_ms,
                result.total_compute_ms,
                output_copy_ms,
                total_custom_op_ms,
                data->expert_workers,
                static_cast<long long>(n_expert_compute),
                result.backend.c_str(),
                result.backend_fallback ? 1 : 0,
                prefetch_prediction.current_hits,
                predicted_current_count,
                predicted_current_waste,
                prefetch_prediction.next_1.size(),
                prefetch_prediction.next_2.size(),
                selected_ids_json.c_str(),
                predicted_current_json.c_str(),
                predicted_next_1_json.c_str(),
                predicted_next_2_json.c_str());
        }
    }
}

void llama_model_openai_moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    uint32_t swa_period = 2;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false);
    hparams.set_swa_pattern(swa_period);

    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);

    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_20B; break;
        case 36: type = LLM_TYPE_120B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_openai_moe::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp = hparams.n_ff_exp;
    const int expert_flags = llama_openai_moe_infinitum_selective_enabled() ? TENSOR_NOT_REQUIRED : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_head * n_rot, n_head_kv * n_rot, n_head_kv * n_rot, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_rot, n_embd}, 0);

        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), {n_head}, 0);

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {  n_embd, n_expert}, 0);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, expert_flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, expert_flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, expert_flags);

        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, 0);

        layer.ffn_gate_inp_b  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "bias", i), {n_expert}, 0);
        layer.ffn_gate_exps_b = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "bias", i), {n_ff_exp, n_expert}, expert_flags);
        layer.ffn_down_exps_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "bias", i), {  n_embd, n_expert}, expert_flags);
        layer.ffn_up_exps_b   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "bias", i), {n_ff_exp, n_expert}, expert_flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_openai_moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_openai_moe::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_rot, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, model.layers[il].attn_sinks, nullptr, 1.0f/sqrtf(float(n_rot)), il);

            cb(cur, "attn_out", il);
        }
        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = ffn_inp;
        cur = build_norm(cur,
                model.layers[il].attn_post_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        // Infinitum router probe: build router top-k without resident experts.
        const bool infinitum_router_probe =
            llama_infinitum_moe_selective_enabled() && llama_infinitum_moe_router_probe_enabled();
        if (infinitum_router_probe) {
            ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, cur);
            cb(logits, "ffn_moe_logits", il);

            if (model.layers[il].ffn_gate_inp_b) {
                logits = ggml_add(ctx0, logits, model.layers[il].ffn_gate_inp_b);
                cb(logits, "ffn_moe_logits_biased", il);
            }

            ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, logits, n_expert_used);
            cb(selected_experts->src[0], "ffn_moe_argsort", il);
            cb(selected_experts, "ffn_moe_topk", il);

            ggml_tensor * probs = ggml_reshape_3d(ctx0, logits, 1, n_expert, cur->ne[1]);
            ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts);
            cb(weights, "ffn_moe_weights", il);

            weights = ggml_reshape_2d(ctx0, weights, n_expert_used, cur->ne[1]);
            weights = ggml_soft_max(ctx0, weights);
            weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, cur->ne[1]);
            cb(weights, "ffn_moe_weights_softmax", il);

            ggml_build_forward_expand(gf, weights);
            cur = ggml_scale(ctx0, ffn_inp, 0.0f);
            cb(cur, "ffn_moe_out_router_probe_zero", il);
        } else if (!model.layers[il].ffn_up_exps || !model.layers[il].ffn_gate_exps || !model.layers[il].ffn_down_exps) {
            if (llama_openai_moe_infinitum_selective_enabled()) {
                const llama_infinitum_moe_index_info & expert_index = llama_openai_moe_infinitum_expert_index();
                if (expert_index.path_exists && expert_index.expert_entry_count > 0) {
                    const llama_infinitum_moe_expert_slice * slice = llama_infinitum_moe_find_slice(expert_index, il, 0, "down_proj_blocks");
                    if (slice != nullptr) {
                        static llama_infinitum_moe_slice_cache cache(llama_infinitum_moe_cache_bytes_from_env());
                        ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, cur);
                        cb(logits, "ffn_moe_logits_external", il);

                        if (model.layers[il].ffn_gate_inp_b) {
                            logits = ggml_add(ctx0, logits, model.layers[il].ffn_gate_inp_b);
                            cb(logits, "ffn_moe_logits_external_biased", il);
                        }

                        const bool use_ggml_pack_slots =
                            llama_infinitum_moe_ggml_pack_slots_enabled() &&
                            (cur->ne[1] == 1 || llama_openai_moe_infinitum_pack_slots_prefill_enabled());
                        if (cur->ne[1] == 1 && llama_infinitum_moe_ggml_pack_enabled() && !use_ggml_pack_slots &&
                                llama_openai_moe_infinitum_ggml_pack_prefetch_enabled()) {
                            const int prefetch_top_k = llama_openai_moe_infinitum_ggml_pack_prefetch_max_experts();
                            const int64_t n_prefetch_expert_used =
                                std::max<int64_t>(n_expert_used, std::min<int64_t>(n_expert, prefetch_top_k));
                            ggml_tensor * prefetch_experts = ggml_argsort_top_k(ctx0, logits, n_prefetch_expert_used);
                            cb(prefetch_experts, "ffn_moe_topk_external_prefetch_candidates", il);
                            auto * userdata = llama_openai_moe_infinitum_prefetch_userdata_for_layer(
                                expert_index,
                                cache,
                                static_cast<int>(il),
                                llama_openai_moe_infinitum_expert_predictor_lookahead(),
                                static_cast<int>(n_expert_used));
                            ggml_tensor * prefetch_ids = ggml_map_custom1(ctx0, prefetch_experts,
                                    llama_openai_moe_infinitum_prefetch_selected_op, 1, userdata);
                            cb(prefetch_ids, "ffn_moe_topk_external_prefetch", il);
                            ggml_build_forward_expand(gf, prefetch_ids);
                        }

                        ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, logits, n_expert_used);
                        cb(selected_experts, "ffn_moe_topk_external", il);

                        ggml_tensor * probs = ggml_reshape_3d(ctx0, logits, 1, n_expert, cur->ne[1]);
                        ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts);
                        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, cur->ne[1]);
                        weights = ggml_soft_max(ctx0, weights);
                        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, cur->ne[1]);
                        cb(weights, "ffn_moe_weights_external_softmax", il);

                        if (llama_infinitum_moe_ggml_pack_enabled() && !use_ggml_pack_slots) {
                            ggml_tensor * gate_up_exps = llama_infinitum_moe_ggml_pack_tensor(ctx0, expert_index, static_cast<int>(il), "gate_up");
                            ggml_tensor * down_exps    = llama_infinitum_moe_ggml_pack_tensor(ctx0, expert_index, static_cast<int>(il), "down");
                            ggml_tensor * gate_up_b    = llama_infinitum_moe_ggml_pack_tensor(ctx0, expert_index, static_cast<int>(il), "gate_up_bias");
                            ggml_tensor * down_b       = llama_infinitum_moe_ggml_pack_tensor(ctx0, expert_index, static_cast<int>(il), "down_bias");
                            if (gate_up_exps == nullptr || down_exps == nullptr || gate_up_b == nullptr || down_b == nullptr) {
                                GGML_ABORT("LLAMA_INFINITUM_V2_GGML_EXPERT_PACK is enabled but external GGML expert pack tensor creation failed");
                            }

                            ggml_build_forward_expand(gf, weights);
                            ggml_tensor * moe_inp = ggml_reshape_3d(ctx0, cur, n_embd, 1, cur->ne[1]);
                            ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, moe_inp, selected_experts);
                            cb(gate_up, "ffn_moe_gate_up_external_ggml", il);
                            gate_up = ggml_add_id(ctx0, gate_up, gate_up_b, selected_experts);
                            cb(gate_up, "ffn_moe_gate_up_external_ggml_biased", il);

                            const int64_t n_ff = gate_up->ne[0] / 2;
                            ggml_tensor * gate = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
                            ggml_tensor * up = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
                            cur = ggml_swiglu_oai(ctx0, gate, up, 1.702f, 7.0f);
                            cb(cur, "ffn_moe_swiglu_external_ggml", il);

                            ggml_tensor * experts = build_lora_mm_id(down_exps, cur, selected_experts);
                            cb(experts, "ffn_moe_down_external_ggml", il);
                            experts = ggml_add_id(ctx0, experts, down_b, selected_experts);
                            experts = ggml_mul(ctx0, experts, weights);
                            cb(experts, "ffn_moe_weighted_external_ggml", il);

                            ggml_tensor * moe_out = nullptr;
                            for (int64_t ie = 0; ie < n_expert_used; ++ie) {
                                ggml_tensor * expert_view = ggml_view_2d(ctx0, experts, n_embd, experts->ne[2], experts->nb[2], ie * experts->nb[1]);
                                moe_out = moe_out == nullptr ? expert_view : ggml_add(ctx0, moe_out, expert_view);
                            }
                            cur = moe_out;
                            cb(cur, "ffn_moe_out_external_ggml", il);
                        } else {
                            auto * userdata = llama_openai_moe_infinitum_external_mlp_userdata_for_layer(
                                expert_index,
                                cache,
                                static_cast<int>(il),
                                llama_openai_moe_infinitum_expert_workers());
                            cur = ggml_map_custom3(ctx0, cur, selected_experts, weights,
                                    llama_openai_moe_infinitum_external_mlp_op, 1, userdata);
                            cb(cur, "ffn_moe_out_external", il);
                        }
                    } else {
                        GGML_ABORT("LLAMA_INFINITUM_SELECTIVE_MOE loaded an OpenAI-MoE core with external expert index, but no layer slice lookup matched the current graph layer");
                    }
                } else {
                    GGML_ABORT("LLAMA_INFINITUM_SELECTIVE_MOE loaded an OpenAI-MoE core without resident experts and without a valid LLAMA_INFINITUM_EXPERT_INDEX");
                }
            } else {
                GGML_ABORT("OpenAI-MoE expert tensors are missing");
            }
        } else {
            // MoE branch
            const int64_t resident_n_expert_used =
                llama_openai_moe_infinitum_resident_expert_top_k_limit(static_cast<int>(il), n_expert_used);
            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                    model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                    model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                    model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                    nullptr,
                    n_expert, resident_n_expert_used,
                    LLM_FFN_SWIGLU_OAI_MOE, false,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
                    il);
            cb(cur, "ffn_moe_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
