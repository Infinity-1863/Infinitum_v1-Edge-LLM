#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"

struct llama_infinitum_moe_expert_slice {
    int layer_index = -1;
    int expert_id = -1;
    std::string kind;
    std::string source_file;
    std::uint64_t byte_offset = 0;
    std::uint64_t byte_length = 0;
};

struct llama_infinitum_moe_index_info {
    bool enabled = false;
    bool path_exists = false;
    std::string path;
    std::string base_dir;
    int hidden_size = 0;
    int gate_up_rows = 0;
    int down_rows = 0;
    std::size_t expert_entry_count = 0;
    std::uint64_t expert_slice_bytes = 0;
    std::vector<llama_infinitum_moe_expert_slice> slices;
    std::string error;
};

struct llama_infinitum_moe_cache_stats {
    std::uint64_t cache_hits = 0;
    std::uint64_t cache_misses = 0;
    std::uint64_t evictions = 0;
    std::uint64_t loaded_bytes = 0;
    std::uint64_t touched_bytes = 0;
    std::uint64_t mapped_bytes = 0;
    std::uint64_t copied_bytes = 0;
    std::uint64_t prepacked_bytes = 0;
    std::uint64_t resident_bytes = 0;
    std::uint64_t process_resident_bytes = 0;
};

struct llama_infinitum_moe_loaded_slice {
    llama_infinitum_moe_expert_slice meta;
    std::vector<std::uint8_t> bytes;
    std::vector<std::int8_t> sdot_values;
    std::vector<std::int8_t> sdot_kmajor_values;
    std::vector<float> table_scale_values;
    std::shared_ptr<const void> mapped_source;
    const std::uint8_t * mapped_data = nullptr;
    std::size_t mapped_size = 0;
};

struct llama_infinitum_moe_expert_mlp_result {
    bool ok = false;
    int layer_index = -1;
    int expert_id = -1;
    std::uint64_t loaded_bytes = 0;
    std::uint64_t cache_hits_delta = 0;
    std::uint64_t cache_misses_delta = 0;
    std::uint64_t cache_evictions_delta = 0;
    std::uint64_t gpu_slot_hits_delta = 0;
    std::uint64_t gpu_slot_misses_delta = 0;
    std::uint64_t gpu_slot_evictions_delta = 0;
    std::uint64_t loaded_bytes_delta = 0;
    std::uint64_t touched_bytes_delta = 0;
    std::uint64_t mapped_bytes_delta = 0;
    std::uint64_t copied_bytes_delta = 0;
    std::uint64_t prepacked_bytes_delta = 0;
    std::uint64_t resident_bytes = 0;
    std::uint64_t process_resident_bytes = 0;
    double load_ms = 0.0;
    double expert_upload_ms = 0.0;
    double graph_input_ms = 0.0;
    double graph_compute_ms = 0.0;
    double graph_output_ms = 0.0;
    double gate_up_ms = 0.0;
    double activation_ms = 0.0;
    double down_ms = 0.0;
    double accumulate_ms = 0.0;
    double total_compute_ms = 0.0;
    float output_norm = 0.0f;
    std::vector<float> output;
    std::string backend = "cpu";
    bool backend_fallback = false;
    std::string error;
};

class llama_infinitum_moe_slice_cache {
public:
    explicit llama_infinitum_moe_slice_cache(std::uint64_t max_bytes);

    bool get_or_load(
            const llama_infinitum_moe_index_info & info,
            const llama_infinitum_moe_expert_slice & slice,
            llama_infinitum_moe_loaded_slice & out,
            std::string & error);

    std::shared_ptr<const llama_infinitum_moe_loaded_slice> get_or_load_ptr(
            const llama_infinitum_moe_index_info & info,
            const llama_infinitum_moe_expert_slice & slice,
            std::string & error);

    llama_infinitum_moe_cache_stats stats() const;

private:
    std::uint64_t max_bytes;
    llama_infinitum_moe_cache_stats cache_stats;
    mutable std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<llama_infinitum_moe_loaded_slice>> items;
    std::vector<std::string> order;

    static std::string key_for(const llama_infinitum_moe_expert_slice & slice);
    void touch(const std::string & key);
    void evict_if_needed(const std::string & protected_key);
};

bool llama_infinitum_moe_selective_enabled();

bool llama_infinitum_moe_router_probe_enabled();

bool llama_infinitum_moe_two_phase_bridge_enabled();

bool llama_infinitum_moe_ggml_pack_enabled();

bool llama_infinitum_moe_ggml_pack_slots_enabled();

std::uint64_t llama_infinitum_moe_cache_bytes_from_env();

std::uint64_t llama_infinitum_moe_gpu_cache_bytes_from_env();

bool llama_infinitum_moe_prefetch_enabled();

llama_infinitum_moe_index_info llama_infinitum_moe_index_from_env();

const llama_infinitum_moe_expert_slice * llama_infinitum_moe_find_slice(
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        int expert_id,
        const char * kind);

ggml_tensor * llama_infinitum_moe_ggml_pack_tensor(
        ggml_context * ctx,
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        const char * kind);

ggml_tensor * llama_infinitum_moe_ggml_pack_tensor_shaped(
        ggml_context * ctx,
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        const char * kind,
        ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2);

bool llama_infinitum_moe_read_slice(
        const llama_infinitum_moe_index_info & info,
        const llama_infinitum_moe_expert_slice & slice,
        std::vector<std::uint8_t> & out,
        std::string & error);

llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_expert_mlp_cpu(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        int expert_id,
        const std::vector<float> & hidden);

llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_selected_experts_cpu(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        const std::vector<int> & expert_ids,
        const std::vector<float> & expert_weights,
        const std::vector<float> & hidden,
        int expert_workers);

llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_selected_experts_into(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        const std::vector<int> & expert_ids,
        const std::vector<float> & expert_weights,
        const float * hidden,
        int hidden_size,
        float * output,
        int expert_workers);

llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_gemma4_q4_pack_selected_experts_into(
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        const std::vector<int> & expert_ids,
        const std::vector<float> & expert_weights,
        const float * hidden,
        int hidden_size,
        float * output);

llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_gemma4_f16_pack_selected_experts_into(
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        const std::vector<int> & expert_ids,
        const std::vector<float> & expert_weights,
        const float * hidden,
        int hidden_size,
        float * output);

llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_gemma4_q8_pack_selected_experts_into(
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        const std::vector<int> & expert_ids,
        const std::vector<float> & expert_weights,
        const float * hidden,
        int hidden_size,
        float * output);

llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_selected_experts(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        const std::vector<int> & expert_ids,
        const std::vector<float> & expert_weights,
        const std::vector<float> & hidden,
        int expert_workers);

bool llama_infinitum_moe_prefetch_selected_experts(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        const std::vector<int> & expert_ids,
        std::string & error);

void llama_infinitum_moe_prefetch_selected_experts_async(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        const std::vector<int> & expert_ids);

bool llama_infinitum_moe_prefetch_selected_gpu_experts(
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        const std::vector<int> & expert_ids,
        std::string & error);
