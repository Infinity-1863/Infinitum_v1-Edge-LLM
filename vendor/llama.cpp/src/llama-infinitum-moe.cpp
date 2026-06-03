#include "llama-infinitum-moe.h"
#include "llama-mmap.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <array>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <deque>
#include <functional>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_set>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#endif

static bool llama_infinitum_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool llama_infinitum_env_disabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] == '0';
}

static bool llama_infinitum_debug_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_DEBUG");
}

static bool llama_infinitum_pack_report_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_PACK_REPORT");
}

static void llama_infinitum_debug_log(const char * fmt, ...) {
    if (!llama_infinitum_debug_enabled()) {
        return;
    }
    std::fprintf(stderr, "infinitum: ");
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
}

static bool llama_infinitum_mxfp4_rowmajor_prepack_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_EXPERT_PREPACK");
}

static bool llama_infinitum_mxfp4_kmajor_prepack_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_EXPERT_SDOT") &&
        llama_infinitum_env_enabled("LLAMA_INFINITUM_EXPERT_KMAJOR_PREPACK");
}

static double llama_infinitum_elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct llama_infinitum_moe_compute_profile {
    double gate_up_ms = 0.0;
    double activation_ms = 0.0;
    double down_ms = 0.0;
};

struct llama_infinitum_quantized_input {
    std::vector<std::int8_t> values;
    std::vector<float> scales;
};

struct llama_infinitum_mxfp4_runtime_flags {
    bool use_sdot = false;
    bool use_sdot_gate_up = false;
    bool use_sdot_down = false;
    bool layer_packed_sdot = false;
    bool use_cpu_table = false;
    bool use_scaled_pair_table = false;
    bool use_input_pair_table = false;
};

enum class llama_infinitum_moe_backend_kind {
    cpu,
    cpu_table,
    simd,
    vulkan,
    fused_arena,
};

struct llama_infinitum_packed_expert_mlp {
    int layer_index = -1;
    int expert_id = -1;
    std::vector<std::int8_t> gate_up_kmajor_values;
    std::vector<std::int8_t> down_kmajor_values;

    std::size_t bytes() const {
        return gate_up_kmajor_values.size() + down_kmajor_values.size();
    }
};

struct llama_infinitum_ggml_pack_mapping;

struct llama_infinitum_ggml_packed_expert_mlp {
    int layer_index = -1;
    int expert_id = -1;
    std::vector<std::uint8_t> gate_blocks;
    std::vector<std::uint8_t> up_blocks;
    std::vector<std::uint8_t> down_blocks;
    std::vector<float> gate_bias;
    std::vector<float> up_bias;
    std::vector<float> down_bias;
    std::shared_ptr<const llama_infinitum_ggml_pack_mapping> mapped_owner;
    const std::uint8_t * mapped_gate_blocks = nullptr;
    const std::uint8_t * mapped_up_blocks = nullptr;
    const std::uint8_t * mapped_down_blocks = nullptr;
    const float * mapped_gate_bias = nullptr;
    const float * mapped_up_bias = nullptr;
    const float * mapped_down_bias = nullptr;
    std::size_t mapped_matrix_bytes = 0;
    int mapped_hidden_size = 0;

    std::size_t gate_blocks_size() const { return mapped_gate_blocks != nullptr ? mapped_matrix_bytes : gate_blocks.size(); }
    std::size_t up_blocks_size() const { return mapped_up_blocks != nullptr ? mapped_matrix_bytes : up_blocks.size(); }
    std::size_t down_blocks_size() const { return mapped_down_blocks != nullptr ? mapped_matrix_bytes : down_blocks.size(); }
    int gate_bias_size() const { return mapped_gate_bias != nullptr ? mapped_hidden_size : static_cast<int>(gate_bias.size()); }
    int up_bias_size() const { return mapped_up_bias != nullptr ? mapped_hidden_size : static_cast<int>(up_bias.size()); }
    int down_bias_size() const { return mapped_down_bias != nullptr ? mapped_hidden_size : static_cast<int>(down_bias.size()); }
    const std::uint8_t * gate_blocks_data() const { return mapped_gate_blocks != nullptr ? mapped_gate_blocks : gate_blocks.data(); }
    const std::uint8_t * up_blocks_data() const { return mapped_up_blocks != nullptr ? mapped_up_blocks : up_blocks.data(); }
    const std::uint8_t * down_blocks_data() const { return mapped_down_blocks != nullptr ? mapped_down_blocks : down_blocks.data(); }
    const float * gate_bias_data() const { return mapped_gate_bias != nullptr ? mapped_gate_bias : gate_bias.data(); }
    const float * up_bias_data() const { return mapped_up_bias != nullptr ? mapped_up_bias : up_bias.data(); }
    const float * down_bias_data() const { return mapped_down_bias != nullptr ? mapped_down_bias : down_bias.data(); }

    bool has_contiguous_gate_up() const {
        return mapped_gate_blocks != nullptr && mapped_up_blocks == mapped_gate_blocks + mapped_matrix_bytes;
    }

    bool has_contiguous_gate_up_bias() const {
        return mapped_gate_bias != nullptr && mapped_up_bias == mapped_gate_bias + mapped_hidden_size;
    }

    std::size_t bytes() const {
        return gate_blocks_size() + up_blocks_size() + down_blocks_size() +
            std::size_t(gate_bias_size() + up_bias_size() + down_bias_size()) * sizeof(float);
    }
};

static int llama_infinitum_mxfp4_block_count_for_hidden(int hidden_size) {
    constexpr int block_size = 32;
    return hidden_size > 0 ? (hidden_size + block_size - 1) / block_size : 0;
}

static bool llama_infinitum_mxfp4_layer_packed_sdot_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_EXPERT_LAYER_PACKED_SDOT");
}

static bool llama_infinitum_mxfp4_sdot_down_only_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_EXPERT_SDOT_DOWN_ONLY");
}

static bool llama_infinitum_mxfp4_sdot_requested();
static bool llama_infinitum_mxfp4_sdot_available();

static llama_infinitum_moe_backend_kind llama_infinitum_moe_backend_kind_from_env();

static llama_infinitum_mxfp4_runtime_flags llama_infinitum_mxfp4_runtime_flags_for_backend(llama_infinitum_moe_backend_kind backend) {
    llama_infinitum_mxfp4_runtime_flags flags;
    const bool simd_backend = backend == llama_infinitum_moe_backend_kind::simd ||
        backend == llama_infinitum_moe_backend_kind::vulkan ||
        backend == llama_infinitum_moe_backend_kind::fused_arena;
    flags.use_sdot = (llama_infinitum_mxfp4_sdot_requested() || simd_backend) && llama_infinitum_mxfp4_sdot_available();
    flags.use_sdot_gate_up = flags.use_sdot && !llama_infinitum_mxfp4_sdot_down_only_enabled();
    flags.use_sdot_down = flags.use_sdot;
    flags.layer_packed_sdot = llama_infinitum_mxfp4_layer_packed_sdot_enabled();
    flags.use_cpu_table = backend == llama_infinitum_moe_backend_kind::cpu_table;
    flags.use_input_pair_table = llama_infinitum_env_enabled("LLAMA_INFINITUM_EXPERT_TABLE_INPUT_PAIR") &&
        flags.use_cpu_table;
    flags.use_scaled_pair_table = llama_infinitum_env_enabled("LLAMA_INFINITUM_EXPERT_TABLE_SCALED_PAIR") &&
        (backend == llama_infinitum_moe_backend_kind::cpu_table || simd_backend);
    return flags;
}

static llama_infinitum_mxfp4_runtime_flags llama_infinitum_mxfp4_runtime_flags_from_env() {
    return llama_infinitum_mxfp4_runtime_flags_for_backend(llama_infinitum_moe_backend_kind_from_env());
}

static bool llama_infinitum_mxfp4_sidecar_kmajor_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_EXPERT_KMAJOR_SIDECAR");
}

static std::uint64_t llama_infinitum_mxfp4_packed_cache_bytes_from_env() {
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_PACKED_CACHE_MB");
    if (value == nullptr || value[0] == '\0') {
        return 512ull * 1024ull * 1024ull;
    }
    char * end = nullptr;
    const unsigned long long mb = std::strtoull(value, &end, 10);
    if (end == value || mb == 0) {
        return 512ull * 1024ull * 1024ull;
    }
    return mb * 1024ull * 1024ull;
}

static std::uint64_t llama_infinitum_mxfp4_packed_admit_hits_from_env() {
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_PACKED_ADMIT_HITS");
    if (value == nullptr || value[0] == '\0') {
        return 2;
    }
    char * end = nullptr;
    const unsigned long long hits = std::strtoull(value, &end, 10);
    if (end == value || hits == 0) {
        return 2;
    }
    return hits;
}

static int llama_infinitum_expert_row_threads_from_env() {
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_ROW_THREADS");
    if (value == nullptr || value[0] == '\0') {
        return 1;
    }
    char * end = nullptr;
    const long threads = std::strtol(value, &end, 10);
    if (end == value || threads <= 1) {
        return 1;
    }
    return std::min<long>(threads, 8);
}

class llama_infinitum_worker_pool {
public:
    ~llama_infinitum_worker_pool() {
        stop_workers();
    }

    void run(int n_threads, int task_count, const std::function<void(int)> & task_fn) {
        if (task_count <= 0) {
            return;
        }
        if (n_threads <= 1 || task_count == 1) {
            for (int i = 0; i < task_count; ++i) {
                task_fn(i);
            }
            return;
        }

        ensure_workers(n_threads - 1);
        {
            std::lock_guard<std::mutex> lock(mutex);
            task = &task_fn;
            tasks = task_count;
            next.store(0, std::memory_order_relaxed);
            active_workers = static_cast<int>(workers.size());
            generation++;
        }
        start_cv.notify_all();

        while (true) {
            const int index = next.fetch_add(1, std::memory_order_relaxed);
            if (index >= task_count) {
                break;
            }
            task_fn(index);
        }

        std::unique_lock<std::mutex> lock(mutex);
        done_cv.wait(lock, [&]() { return active_workers == 0; });
        task = nullptr;
    }

private:
    std::mutex mutex;
    std::condition_variable start_cv;
    std::condition_variable done_cv;
    std::vector<std::thread> workers;
    const std::function<void(int)> * task = nullptr;
    std::atomic<int> next{0};
    std::uint64_t generation = 0;
    int tasks = 0;
    int active_workers = 0;
    bool stop = false;

    void stop_workers() {
        std::unique_lock<std::mutex> lock(mutex);
        if (workers.empty()) {
            return;
        }
        stop = true;
        generation++;
        start_cv.notify_all();
        std::vector<std::thread> old_workers;
        old_workers.swap(workers);
        lock.unlock();
        for (std::thread & worker : old_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        lock.lock();
        stop = false;
        active_workers = 0;
        task = nullptr;
    }

    void ensure_workers(int wanted) {
        std::unique_lock<std::mutex> lock(mutex);
        if (static_cast<int>(workers.size()) == wanted) {
            return;
        }
        lock.unlock();
        stop_workers();
        lock.lock();
        stop = false;
        for (int i = 0; i < wanted; ++i) {
            workers.emplace_back([this]() { worker_loop(); });
        }
    }

    void worker_loop() {
        std::uint64_t seen_generation = 0;
        while (true) {
            const std::function<void(int)> * current_task = nullptr;
            int current_tasks = 0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                start_cv.wait(lock, [&]() { return stop || generation != seen_generation; });
                if (stop) {
                    break;
                }
                seen_generation = generation;
                current_task = task;
                current_tasks = tasks;
            }
            if (current_task != nullptr) {
                while (true) {
                    const int index = next.fetch_add(1, std::memory_order_relaxed);
                    if (index >= current_tasks) {
                        break;
                    }
                    (*current_task)(index);
                }
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                active_workers--;
                if (active_workers == 0) {
                    done_cv.notify_one();
                }
            }
        }
    }
};

static std::uint64_t llama_infinitum_parse_json_u64(const std::string & text, const std::string & key) {
    const std::string quoted_key = "\"" + key + "\"";
    const std::size_t key_pos = text.find(quoted_key);
    if (key_pos == std::string::npos) {
        return 0;
    }
    const std::size_t colon_pos = text.find(':', key_pos + quoted_key.size());
    if (colon_pos == std::string::npos) {
        return 0;
    }
    std::size_t pos = colon_pos + 1;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\r' || text[pos] == '\n')) {
        ++pos;
    }
    std::uint64_t value = 0;
    while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
        value = value * 10 + static_cast<std::uint64_t>(text[pos] - '0');
        ++pos;
    }
    return value;
}

static std::string llama_infinitum_parse_json_object(const std::string & text, const std::string & key) {
    const std::string quoted_key = "\"" + key + "\"";
    const std::size_t key_pos = text.find(quoted_key);
    if (key_pos == std::string::npos) {
        return "";
    }
    const std::size_t colon_pos = text.find(':', key_pos + quoted_key.size());
    if (colon_pos == std::string::npos) {
        return "";
    }
    const std::size_t object_start = text.find('{', colon_pos + 1);
    if (object_start == std::string::npos) {
        return "";
    }
    int depth = 0;
    bool in_string = false;
    for (std::size_t pos = object_start; pos < text.size(); ++pos) {
        const char ch = text[pos];
        if (ch == '"' && (pos == 0 || text[pos - 1] != '\\')) {
            in_string = !in_string;
        }
        if (in_string) {
            continue;
        }
        if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            --depth;
            if (depth == 0) {
                return text.substr(object_start, pos - object_start + 1);
            }
        }
    }
    return "";
}

static std::uint64_t llama_infinitum_parse_json_summary_u64(const std::string & text, const std::string & key) {
    const std::string summary = llama_infinitum_parse_json_object(text, "summary");
    if (summary.empty()) {
        return llama_infinitum_parse_json_u64(text, key);
    }
    return llama_infinitum_parse_json_u64(summary, key);
}

static std::string llama_infinitum_parse_json_string(const std::string & text, const std::string & key) {
    const std::string quoted_key = "\"" + key + "\"";
    const std::size_t key_pos = text.find(quoted_key);
    if (key_pos == std::string::npos) {
        return "";
    }
    const std::size_t colon_pos = text.find(':', key_pos + quoted_key.size());
    if (colon_pos == std::string::npos) {
        return "";
    }
    std::size_t pos = text.find('"', colon_pos + 1);
    if (pos == std::string::npos) {
        return "";
    }
    const std::size_t start = ++pos;
    while (pos < text.size()) {
        if (text[pos] == '"' && (pos == start || text[pos - 1] != '\\')) {
            return text.substr(start, pos - start);
        }
        ++pos;
    }
    return "";
}

static void llama_infinitum_parse_expert_slices(const std::string & text, llama_infinitum_moe_index_info & info) {
    const std::size_t entries_pos = text.find("\"entries\"");
    if (entries_pos == std::string::npos) {
        return;
    }
    const std::size_t array_pos = text.find('[', entries_pos);
    if (array_pos == std::string::npos) {
        return;
    }

    std::size_t pos = array_pos + 1;
    while (pos < text.size()) {
        const std::size_t object_start = text.find('{', pos);
        const std::size_t array_end = text.find(']', pos);
        if (object_start == std::string::npos || (array_end != std::string::npos && array_end < object_start)) {
            break;
        }
        const std::size_t object_end = text.find('}', object_start + 1);
        if (object_end == std::string::npos) {
            break;
        }
        const std::string object = text.substr(object_start, object_end - object_start + 1);
        if (object.find("\"expert_id\"") != std::string::npos && object.find("\"byte_offset\"") != std::string::npos) {
            llama_infinitum_moe_expert_slice slice;
            slice.layer_index = static_cast<int>(llama_infinitum_parse_json_u64(object, "layer_index"));
            slice.expert_id = static_cast<int>(llama_infinitum_parse_json_u64(object, "expert_id"));
            slice.kind = llama_infinitum_parse_json_string(object, "kind");
            slice.source_file = llama_infinitum_parse_json_string(object, "source_file");
            slice.byte_offset = llama_infinitum_parse_json_u64(object, "byte_offset");
            slice.byte_length = llama_infinitum_parse_json_u64(object, "byte_length");
            info.slices.push_back(slice);
        }
        pos = object_end + 1;
    }
}

static void llama_infinitum_infer_index_shape(llama_infinitum_moe_index_info & info) {
    for (const llama_infinitum_moe_expert_slice & slice : info.slices) {
        if (slice.kind == "down_proj_bias" && slice.byte_length >= 2 && slice.byte_length % 2 == 0) {
            info.down_rows = static_cast<int>(slice.byte_length / 2);
            if (info.hidden_size == 0) {
                info.hidden_size = info.down_rows;
            }
        } else if (slice.kind == "gate_up_proj_bias" && slice.byte_length >= 2 && slice.byte_length % 2 == 0) {
            info.gate_up_rows = static_cast<int>(slice.byte_length / 2);
        }
        if (info.hidden_size > 0 && info.gate_up_rows > 0 && info.down_rows > 0) {
            break;
        }
    }
    if (info.hidden_size == 0 && info.down_rows > 0) {
        info.hidden_size = info.down_rows;
    }
    if (info.down_rows == 0 && info.hidden_size > 0) {
        info.down_rows = info.hidden_size;
    }
    if (info.gate_up_rows == 0 && info.hidden_size > 0) {
        info.gate_up_rows = 2 * info.hidden_size;
    }
}

static std::string llama_infinitum_join_path(const std::string & base, const std::string & name) {
    if (base.empty() || name.empty()) {
        return base.empty() ? name : base;
    }
    const char last = base[base.size() - 1];
    if (last == '/' || last == '\\') {
        return base + name;
    }
#if defined(_WIN32)
    return base + "\\" + name;
#else
    return base + "/" + name;
#endif
}

bool llama_infinitum_moe_selective_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_SELECTIVE_MOE");
}

bool llama_infinitum_moe_router_probe_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_ROUTER_PROBE");
}

bool llama_infinitum_moe_two_phase_bridge_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_TWO_PHASE_BRIDGE");
}

bool llama_infinitum_moe_ggml_pack_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_V2_GGML_EXPERT_PACK");
}

bool llama_infinitum_moe_ggml_pack_slots_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_V2_GGML_EXPERT_PACK_SLOTS");
}

static bool llama_infinitum_moe_ggml_pack_prefetch_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_GGML_PACK_PREFETCH");
}

static bool llama_infinitum_moe_ggml_pack_prefetch_touch_fallback_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_GGML_PACK_PREFETCH_TOUCH_FALLBACK");
}

std::uint64_t llama_infinitum_moe_cache_bytes_from_env() {
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_CACHE_MB");
    std::uint64_t mb = 1024;
    if (value != nullptr && value[0] != '\0') {
        mb = 0;
        for (const char * p = value; *p >= '0' && *p <= '9'; ++p) {
            mb = mb * 10 + static_cast<std::uint64_t>(*p - '0');
        }
        if (mb == 0) {
            mb = 1024;
        }
    }
    return mb * 1024ull * 1024ull;
}

std::uint64_t llama_infinitum_moe_gpu_cache_bytes_from_env() {
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_GPU_CACHE_MB");
    std::uint64_t mb = 0;
    if (value != nullptr && value[0] != '\0') {
        for (const char * p = value; *p >= '0' && *p <= '9'; ++p) {
            mb = mb * 10 + static_cast<std::uint64_t>(*p - '0');
        }
    }
    return mb * 1024ull * 1024ull;
}

static std::uint64_t llama_infinitum_moe_gpu_cache_bytes_or_default() {
    const std::uint64_t configured = llama_infinitum_moe_gpu_cache_bytes_from_env();
    if (configured != 0) {
        return configured;
    }
    return 4096ull * 1024ull * 1024ull;
}

static int llama_infinitum_moe_gpu_layer_slots_from_env() {
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_GPU_LAYER_SLOTS");
    if (value == nullptr || value[0] == '\0') {
        return 8;
    }
    char * end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return 8;
    }
    if (parsed > 128) {
        parsed = 128;
    }
    return static_cast<int>(parsed);
}

static int llama_infinitum_moe_gpu_global_slots_from_env() {
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_GPU_GLOBAL_SLOTS");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    if (std::strcmp(value, "auto") == 0 || std::strcmp(value, "AUTO") == 0) {
        return 0;
    }
    char * end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return 0;
    }
    if (parsed > 512) {
        parsed = 512;
    }
    return static_cast<int>(parsed);
}

static bool llama_infinitum_moe_gpu_global_slots_enabled() {
    return llama_infinitum_moe_gpu_global_slots_from_env() > 0;
}

static int llama_infinitum_moe_gpu_selected_slots_from_env() {
    return llama_infinitum_moe_gpu_global_slots_enabled() ?
        llama_infinitum_moe_gpu_global_slots_from_env() :
        llama_infinitum_moe_gpu_layer_slots_from_env();
}

static int llama_infinitum_moe_gpu_layer_slots_requested_from_env() {
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_GPU_LAYER_SLOTS");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    if (std::strcmp(value, "auto") == 0 || std::strcmp(value, "AUTO") == 0) {
        return 0;
    }
    char * end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return 0;
    }
    if (parsed > 128) {
        parsed = 128;
    }
    return static_cast<int>(parsed);
}

static int llama_infinitum_moe_layer_count(const llama_infinitum_moe_index_info & info) {
    int max_layer = -1;
    for (const llama_infinitum_moe_expert_slice & slice : info.slices) {
        max_layer = std::max(max_layer, slice.layer_index);
    }
    return max_layer + 1;
}

static bool llama_infinitum_moe_vulkan_q8_input_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_EXPERT_VULKAN_Q8_INPUT");
}

static bool llama_infinitum_moe_vulkan_f16_input_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_EXPERT_VULKAN_F16_INPUT");
}

static bool llama_infinitum_moe_gpu_stream_layers_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_EXPERT_GPU_STREAM_LAYERS");
}

static int llama_infinitum_moe_gpu_layer_cache_limit_from_env() {
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_GPU_LAYER_CACHE_LIMIT");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    char * end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return 0;
    }
    if (parsed > 256) {
        parsed = 256;
    }
    return static_cast<int>(parsed);
}

static bool llama_infinitum_gemma4_q4_slot_views_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_GEMMA4_Q4_SLOT_VIEWS");
}

static bool llama_infinitum_gemma4_q4_slot_f16_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_GEMMA4_Q4_SLOT_F16");
}

static bool llama_infinitum_gemma4_f16_pack_enabled() {
    const char * value = std::getenv("LLAMA_INFINITUM_GEMMA4_GGML_EXPERT_PACK_TYPE");
    return value != nullptr && std::strcmp(value, "f16") == 0;
}

static ggml_type llama_infinitum_gemma4_pack_type_from_env() {
    const char * value = std::getenv("LLAMA_INFINITUM_GEMMA4_GGML_EXPERT_PACK_TYPE");
    if (value != nullptr && std::strcmp(value, "f16") == 0) {
        return GGML_TYPE_F16;
    }
    if (value != nullptr && std::strcmp(value, "q8_0") == 0) {
        return GGML_TYPE_Q8_0;
    }
    return GGML_TYPE_Q4_0;
}

static bool llama_infinitum_gemma4_direct_f16_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_GEMMA4_DIRECT_F16");
}

bool llama_infinitum_moe_prefetch_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_EXPERT_PREFETCH");
}

static std::string llama_infinitum_moe_backend_from_env() {
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_BACKEND");
    if (value == nullptr || value[0] == '\0') {
        return "cpu";
    }
    const std::string backend(value);
    if (backend == "virtual") {
        return "cpu";
    }
    if (backend == "sycl_arena" || backend == "cuda_arena") {
        return "fused_arena";
    }
    if (backend == "cpu" || backend == "cpu_table" || backend == "simd" || backend == "vulkan" ||
            backend == "fused_arena") {
        return backend;
    }
    return "cpu";
}

static llama_infinitum_moe_backend_kind llama_infinitum_moe_backend_kind_from_env() {
    const std::string backend = llama_infinitum_moe_backend_from_env();
    if (backend == "cpu_table") {
        return llama_infinitum_moe_backend_kind::cpu_table;
    }
    if (backend == "simd") {
        return llama_infinitum_moe_backend_kind::simd;
    }
    if (backend == "vulkan") {
        return llama_infinitum_moe_backend_kind::vulkan;
    }
    if (backend == "fused_arena") {
        return llama_infinitum_moe_backend_kind::fused_arena;
    }
    return llama_infinitum_moe_backend_kind::cpu;
}

static const char * llama_infinitum_moe_backend_name(llama_infinitum_moe_backend_kind backend) {
    switch (backend) {
        case llama_infinitum_moe_backend_kind::cpu_table:
            return "cpu_table";
        case llama_infinitum_moe_backend_kind::simd:
            return "simd";
        case llama_infinitum_moe_backend_kind::vulkan:
            return "vulkan";
        case llama_infinitum_moe_backend_kind::fused_arena:
            return "fused_arena";
        case llama_infinitum_moe_backend_kind::cpu:
        default:
            return "cpu";
    }
}

static std::string llama_infinitum_moe_fused_target_from_env() {
    const char * value = std::getenv("LLAMA_INFINITUM_EXPERT_FUSED_TARGET");
    if (value == nullptr || value[0] == '\0') {
        const char * backend_value = std::getenv("LLAMA_INFINITUM_EXPERT_BACKEND");
        if (backend_value != nullptr && std::strcmp(backend_value, "sycl_arena") == 0) {
            return "sycl";
        }
        if (backend_value != nullptr && std::strcmp(backend_value, "cuda_arena") == 0) {
            return "cuda";
        }
        return "auto";
    }
    const std::string target(value);
    if (target == "auto" || target == "vulkan" || target == "sycl" || target == "cuda") {
        return target;
    }
    return "auto";
}

static bool llama_infinitum_moe_scaled_pair_table_enabled() {
    const llama_infinitum_moe_backend_kind backend = llama_infinitum_moe_backend_kind_from_env();
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_EXPERT_TABLE_SCALED_PAIR") &&
        (backend == llama_infinitum_moe_backend_kind::cpu_table ||
         backend == llama_infinitum_moe_backend_kind::simd ||
         backend == llama_infinitum_moe_backend_kind::vulkan ||
         backend == llama_infinitum_moe_backend_kind::fused_arena);
}

llama_infinitum_moe_index_info llama_infinitum_moe_index_from_env() {
    llama_infinitum_moe_index_info info;
    info.enabled = llama_infinitum_moe_selective_enabled();

    const char * path = std::getenv("LLAMA_INFINITUM_EXPERT_INDEX");
    if (path == nullptr || path[0] == '\0') {
        info.error = "LLAMA_INFINITUM_EXPERT_INDEX is not set";
        return info;
    }
    info.path = path;

    std::ifstream input(info.path, std::ios::binary);
    if (!input.good()) {
        info.error = "expert index file is not readable";
        return info;
    }
    info.path_exists = true;

    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string text = buffer.str();

    info.expert_entry_count = static_cast<std::size_t>(llama_infinitum_parse_json_summary_u64(text, "expert_entry_count"));
    if (info.expert_entry_count == 0) {
        info.expert_entry_count = static_cast<std::size_t>(llama_infinitum_parse_json_summary_u64(text, "expert_slice_count"));
    }
    info.expert_slice_bytes = llama_infinitum_parse_json_summary_u64(text, "expert_slice_bytes");
    info.hidden_size = static_cast<int>(llama_infinitum_parse_json_summary_u64(text, "hidden_size"));
    info.gate_up_rows = static_cast<int>(llama_infinitum_parse_json_summary_u64(text, "gate_up_rows"));
    info.down_rows = static_cast<int>(llama_infinitum_parse_json_summary_u64(text, "down_rows"));
    info.base_dir = llama_infinitum_parse_json_string(text, "base_dir");
    llama_infinitum_parse_expert_slices(text, info);
    llama_infinitum_infer_index_shape(info);
    if (info.expert_entry_count == 0 && !info.slices.empty()) {
        info.expert_entry_count = info.slices.size();
    }
    if (info.expert_entry_count == 0) {
        info.error = "expert index summary has zero expert_entry_count";
    }
    return info;
}

const llama_infinitum_moe_expert_slice * llama_infinitum_moe_find_slice(
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        int expert_id,
        const char * kind) {
    const std::string wanted_kind = kind != nullptr ? kind : "";
    for (const llama_infinitum_moe_expert_slice & slice : info.slices) {
        if (slice.layer_index == layer_index && slice.expert_id == expert_id && slice.kind == wanted_kind) {
            return &slice;
        }
    }
    return nullptr;
}

static std::string llama_infinitum_dirname(const std::string & path) {
    const std::size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return ".";
    }
    return path.substr(0, pos);
}

static int llama_infinitum_moe_expert_count(const llama_infinitum_moe_index_info & info) {
    int count = 0;
    for (const llama_infinitum_moe_expert_slice & slice : info.slices) {
        count = std::max(count, slice.expert_id + 1);
    }
    return count;
}

static std::string llama_infinitum_moe_ggml_pack_path(const llama_infinitum_moe_index_info & info) {
    const char * explicit_path = std::getenv("LLAMA_INFINITUM_V2_GGML_EXPERT_PACK");
    if (explicit_path != nullptr && explicit_path[0] != '\0' && explicit_path[0] != '1') {
        return explicit_path;
    }
    return llama_infinitum_join_path(
        llama_infinitum_join_path(llama_infinitum_dirname(llama_infinitum_dirname(info.path)), "moe_ggml_pack"),
        "experts.ggml_mxfp4.bin");
}

struct llama_infinitum_ggml_pack_mapping {
    std::unique_ptr<llama_file> file;
    std::unique_ptr<llama_mmap> mapping;
};

struct llama_infinitum_ggml_pack_expert_layout {
    std::size_t matrix_expert_bytes = 0;
    std::size_t gate_up_expert_bytes = 0;
    std::size_t gate_up_bias_expert_bytes = 0;
    std::size_t down_bias_expert_bytes = 0;
    std::size_t gate_up_offset = 0;
    std::size_t down_offset = 0;
    std::size_t gate_up_bias_offset = 0;
    std::size_t down_bias_offset = 0;
};

static std::size_t llama_infinitum_os_page_size() {
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwPageSize > 0 ? static_cast<std::size_t>(info.dwPageSize) : 4096u;
#elif defined(__linux__)
    const long page_size_long = sysconf(_SC_PAGESIZE);
    return page_size_long > 0 ? static_cast<std::size_t>(page_size_long) : 4096u;
#else
    return 4096u;
#endif
}

static std::atomic<std::uint64_t> llama_infinitum_prefetch_touch_sink{0};

static void llama_infinitum_prefetch_memory_range(const void * ptr, std::size_t len) {
    if (ptr == nullptr || len == 0) {
        return;
    }

#if defined(_WIN32)
    WIN32_MEMORY_RANGE_ENTRY range;
    range.VirtualAddress = const_cast<PVOID>(static_cast<const void *>(ptr));
    range.NumberOfBytes = len;
    if (PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
        return;
    }
#elif defined(__linux__) && defined(MADV_WILLNEED)
    {
        const std::size_t page_size = llama_infinitum_os_page_size();
        const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(ptr);
        const std::uintptr_t page = raw & ~(static_cast<std::uintptr_t>(page_size) - 1u);
        const std::size_t adjust = static_cast<std::size_t>(raw - page);
        if (madvise(reinterpret_cast<void *>(page), len + adjust, MADV_WILLNEED) == 0) {
            return;
        }
    }
#endif

    if (!llama_infinitum_moe_ggml_pack_prefetch_touch_fallback_enabled()) {
        if (llama_infinitum_pack_report_enabled()) {
            std::fprintf(stderr, "infinitum_pack_prefetch: os_hint_failed=1 touch_fallback=0 nbytes=%llu\n",
                    static_cast<unsigned long long>(len));
        }
        return;
    }

    const std::size_t page_size = llama_infinitum_os_page_size();
    const auto * bytes = static_cast<const std::uint8_t *>(ptr);
    std::uint64_t touched = 0;
    for (std::size_t offset = 0; offset < len; offset += page_size) {
        touched += bytes[offset];
    }
    touched += bytes[len - 1];
    llama_infinitum_prefetch_touch_sink.fetch_add(touched, std::memory_order_relaxed);
}

static bool llama_infinitum_ggml_pack_expert_layout_for(
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        int expert_id,
        int hidden_size,
        int block_count,
        llama_infinitum_ggml_pack_expert_layout & layout,
        std::string & error) {
    const int experts = llama_infinitum_moe_expert_count(info);
    if (hidden_size <= 0 || block_count <= 0 || experts <= 0 || layer_index < 0 || expert_id < 0 || expert_id >= experts) {
        error = "invalid GGML expert pack selection";
        return false;
    }

    layout.matrix_expert_bytes = std::size_t(hidden_size) * std::size_t(block_count) * 17ull;
    layout.gate_up_expert_bytes = layout.matrix_expert_bytes * 2ull;
    const std::size_t gate_up_all_bytes = layout.gate_up_expert_bytes * std::size_t(experts);
    const std::size_t down_all_bytes = layout.matrix_expert_bytes * std::size_t(experts);
    layout.gate_up_bias_expert_bytes = std::size_t(hidden_size) * 2ull * sizeof(float);
    layout.down_bias_expert_bytes = std::size_t(hidden_size) * sizeof(float);
    const std::size_t gate_up_bias_all_bytes = layout.gate_up_bias_expert_bytes * std::size_t(experts);
    const std::size_t down_bias_all_bytes = layout.down_bias_expert_bytes * std::size_t(experts);
    const std::size_t layer_bytes = gate_up_all_bytes + down_all_bytes + gate_up_bias_all_bytes + down_bias_all_bytes;
    const std::size_t layer_base = std::size_t(layer_index) * layer_bytes;

    layout.gate_up_offset = layer_base + std::size_t(expert_id) * layout.gate_up_expert_bytes;
    layout.down_offset = layer_base + gate_up_all_bytes + std::size_t(expert_id) * layout.matrix_expert_bytes;
    layout.gate_up_bias_offset = layer_base + gate_up_all_bytes + down_all_bytes + std::size_t(expert_id) * layout.gate_up_bias_expert_bytes;
    layout.down_bias_offset = layer_base + gate_up_all_bytes + down_all_bytes + gate_up_bias_all_bytes + std::size_t(expert_id) * layout.down_bias_expert_bytes;
    (void) down_bias_all_bytes;
    return true;
}

static void llama_infinitum_moe_ggml_pack_madvise(const void * ptr, std::size_t len) {
#if defined(__linux__)
    const char * mode_value = std::getenv("LLAMA_INFINITUM_GGML_PACK_MADVISE");
    if (mode_value == nullptr || mode_value[0] == '\0' || mode_value[0] == '0' ||
            std::strcmp(mode_value, "none") == 0) {
        return;
    }
    if (ptr == nullptr || len == 0) {
        return;
    }

    int advice = 0;
    const char * mode = mode_value;
#if defined(MADV_RANDOM)
    if (std::strcmp(mode_value, "random") == 0) {
        advice = MADV_RANDOM;
    } else
#endif
#if defined(MADV_NORMAL)
    if (std::strcmp(mode_value, "normal") == 0) {
        advice = MADV_NORMAL;
    } else
#endif
#if defined(MADV_SEQUENTIAL)
    if (std::strcmp(mode_value, "sequential") == 0) {
        advice = MADV_SEQUENTIAL;
    } else
#endif
    {
        if (llama_infinitum_pack_report_enabled()) {
            std::fprintf(stderr, "infinitum_pack_madvise: mode=%s supported=0 nbytes=%llu\n",
                    mode_value, static_cast<unsigned long long>(len));
        }
        return;
    }

    const long page_size_long = sysconf(_SC_PAGESIZE);
    const std::uintptr_t page_size = page_size_long > 0 ? static_cast<std::uintptr_t>(page_size_long) : 4096u;
    const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(ptr);
    const std::uintptr_t page = raw & ~(page_size - 1u);
    const std::size_t adjust = static_cast<std::size_t>(raw - page);
    const int rc = madvise(reinterpret_cast<void *>(page), len + adjust, advice);
    if (llama_infinitum_pack_report_enabled()) {
        std::fprintf(stderr, "infinitum_pack_madvise: mode=%s rc=%d nbytes=%llu\n",
                mode, rc, static_cast<unsigned long long>(len));
    }
#else
    (void) ptr;
    (void) len;
#endif
}

struct llama_infinitum_ggml_pack_tensor_holder {
    std::shared_ptr<llama_infinitum_ggml_pack_mapping> mapping;
    ggml_backend_buffer_t buffer = nullptr;
    bool buffer_uses_backend_base = false;
    std::size_t buffer_tensor_offset = 0;

    ~llama_infinitum_ggml_pack_tensor_holder() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

static ggml_backend_buffer_t llama_infinitum_try_vulkan_hostptr_buffer(
        void * ptr,
        std::size_t nbytes,
        std::size_t * tensor_offset) {
    static std::atomic<bool> hostptr_runtime_disabled{false};
    if (tensor_offset != nullptr) {
        *tensor_offset = 0;
    }
    if (llama_infinitum_env_disabled("LLAMA_INFINITUM_V2_GGML_EXPERT_VK_HOSTPTR")) {
        if (llama_infinitum_pack_report_enabled()) {
            std::fprintf(stderr, "infinitum_hostptr_report: attempted=0 reason=disabled nbytes=%llu\n",
                    static_cast<unsigned long long>(nbytes));
        }
        return nullptr;
    }
    if (hostptr_runtime_disabled.load(std::memory_order_relaxed)) {
        if (llama_infinitum_pack_report_enabled()) {
            std::fprintf(stderr, "infinitum_hostptr_report: attempted=0 reason=runtime_disabled_after_failure nbytes=%llu\n",
                    static_cast<unsigned long long>(nbytes));
        }
        if (llama_infinitum_env_enabled("LLAMA_INFINITUM_REQUIRE_VK_HOSTPTR")) {
            GGML_ABORT("LLAMA_INFINITUM_REQUIRE_VK_HOSTPTR=1 but Vulkan host pointer import was disabled after an earlier failure");
        }
        return nullptr;
    }

    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (device == nullptr) {
        device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    }
    if (device == nullptr) {
        if (llama_infinitum_pack_report_enabled()) {
            std::fprintf(stderr, "infinitum_hostptr_report: attempted=0 reason=no_vulkan_device nbytes=%llu\n",
                    static_cast<unsigned long long>(nbytes));
        }
        return nullptr;
    }
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(device, &props);

    constexpr std::uintptr_t import_alignment = 65536ull;
    const std::uintptr_t raw_ptr = reinterpret_cast<std::uintptr_t>(ptr);
    const std::uintptr_t aligned_ptr = raw_ptr & ~(import_alignment - 1ull);
    const std::size_t aligned_delta = static_cast<std::size_t>(raw_ptr - aligned_ptr);
    const std::size_t aligned_nbytes =
        (aligned_delta + nbytes + import_alignment - 1ull) & ~(import_alignment - 1ull);

    ggml_backend_buffer_t buffer = ggml_backend_dev_buffer_from_host_ptr(
        device,
        reinterpret_cast<void *>(aligned_ptr),
        aligned_nbytes,
        aligned_nbytes);
    if (buffer != nullptr) {
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (tensor_offset != nullptr) {
            *tensor_offset = aligned_delta;
        }
    }
    if (llama_infinitum_pack_report_enabled()) {
        std::fprintf(stderr,
                "infinitum_hostptr_report: attempted=1 imported=%d nbytes=%llu aligned_nbytes=%llu tensor_offset=%llu device=%s caps_hostptr=%d\n",
                buffer != nullptr ? 1 : 0,
                static_cast<unsigned long long>(nbytes),
                static_cast<unsigned long long>(aligned_nbytes),
                static_cast<unsigned long long>(aligned_delta),
                ggml_backend_dev_name(device),
                props.caps.buffer_from_host_ptr ? 1 : 0);
    }
    if (buffer == nullptr) {
        hostptr_runtime_disabled.store(true, std::memory_order_relaxed);
        if (llama_infinitum_env_enabled("LLAMA_INFINITUM_REQUIRE_VK_HOSTPTR")) {
            GGML_ABORT("LLAMA_INFINITUM_REQUIRE_VK_HOSTPTR=1 but Vulkan host pointer import failed");
        }
    }
    return buffer;
}

static std::shared_ptr<llama_infinitum_ggml_pack_mapping> llama_infinitum_moe_ggml_pack_mapping_for(
        const std::string & path,
        std::string & error) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::shared_ptr<llama_infinitum_ggml_pack_mapping>> mappings;
    std::lock_guard<std::mutex> lock(mutex);
    auto found = mappings.find(path);
    if (found != mappings.end()) {
        return found->second;
    }
    if (!llama_mmap::SUPPORTED) {
        error = "mmap is not supported on this platform";
        return nullptr;
    }
    auto mapping = std::make_shared<llama_infinitum_ggml_pack_mapping>();
    try {
        mapping->file = std::make_unique<llama_file>(path.c_str(), "rb");
        mapping->mapping = std::make_unique<llama_mmap>(mapping->file.get(), 0, false);
        llama_infinitum_moe_ggml_pack_madvise(mapping->mapping->addr(), mapping->file->size());
    } catch (const std::exception & ex) {
        error = ex.what();
        return nullptr;
    }
    found = mappings.emplace(path, mapping).first;
    return found->second;
}

ggml_tensor * llama_infinitum_moe_ggml_pack_tensor(
        ggml_context * ctx,
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        const char * kind) {
    if (!llama_infinitum_moe_ggml_pack_enabled()) {
        return nullptr;
    }
    const int hidden = info.hidden_size;
    const int experts = llama_infinitum_moe_expert_count(info);
    const int blocks = llama_infinitum_mxfp4_block_count_for_hidden(hidden);
    if (hidden <= 0 || experts <= 0 || blocks <= 0) {
        return nullptr;
    }

    const std::size_t matrix_expert_bytes = std::size_t(hidden) * std::size_t(blocks) * 17ull;
    const std::size_t gate_up_expert_bytes = matrix_expert_bytes * 2ull;
    const std::size_t gate_up_all_bytes = gate_up_expert_bytes * std::size_t(experts);
    const std::size_t down_all_bytes = matrix_expert_bytes * std::size_t(experts);
    const std::size_t gate_up_bias_all_bytes = std::size_t(hidden) * 2ull * sizeof(float) * std::size_t(experts);
    const std::size_t down_bias_all_bytes = std::size_t(hidden) * sizeof(float) * std::size_t(experts);
    const std::size_t layer_bytes = gate_up_all_bytes + down_all_bytes + gate_up_bias_all_bytes + down_bias_all_bytes;
    const std::size_t layer_base = std::size_t(layer_index) * layer_bytes;

    ggml_type type = GGML_TYPE_COUNT;
    int64_t ne[3] = { hidden, hidden, experts };
    std::size_t offset = layer_base;
    std::size_t nbytes = gate_up_all_bytes;
    if (std::strcmp(kind, "gate_up") == 0) {
        type = GGML_TYPE_MXFP4;
        ne[1] = hidden * 2;
    } else if (std::strcmp(kind, "down") == 0) {
        type = GGML_TYPE_MXFP4;
        offset = layer_base + gate_up_all_bytes;
        nbytes = down_all_bytes;
    } else if (std::strcmp(kind, "gate_up_bias") == 0) {
        type = GGML_TYPE_F32;
        ne[0] = hidden * 2;
        ne[1] = experts;
        ne[2] = 1;
        offset = layer_base + gate_up_all_bytes + down_all_bytes;
        nbytes = gate_up_bias_all_bytes;
    } else if (std::strcmp(kind, "down_bias") == 0) {
        type = GGML_TYPE_F32;
        ne[0] = hidden;
        ne[1] = experts;
        ne[2] = 1;
        offset = layer_base + gate_up_all_bytes + down_all_bytes + gate_up_bias_all_bytes;
        nbytes = down_bias_all_bytes;
    } else {
        return nullptr;
    }

    const std::string pack_path = llama_infinitum_moe_ggml_pack_path(info);
    std::string error;
    auto mapping = llama_infinitum_moe_ggml_pack_mapping_for(pack_path, error);
    if (mapping == nullptr || offset + nbytes > mapping->file->size()) {
        return nullptr;
    }

    const std::string holder_key = pack_path + ":" + std::to_string(layer_index) + ":" + kind;
    static std::mutex holders_mutex;
    static std::unordered_map<std::string, std::shared_ptr<llama_infinitum_ggml_pack_tensor_holder>> holders;
    std::shared_ptr<llama_infinitum_ggml_pack_tensor_holder> holder;
    {
        std::lock_guard<std::mutex> lock(holders_mutex);
        auto found = holders.find(holder_key);
        if (found == holders.end()) {
            holder = std::make_shared<llama_infinitum_ggml_pack_tensor_holder>();
            holder->mapping = mapping;
            void * ptr = static_cast<std::uint8_t *>(mapping->mapping->addr()) + offset;
            holder->buffer = llama_infinitum_try_vulkan_hostptr_buffer(ptr, nbytes, &holder->buffer_tensor_offset);
            holder->buffer_uses_backend_base = holder->buffer != nullptr;
            if (llama_infinitum_pack_report_enabled()) {
                std::fprintf(stderr,
                    "infinitum_pack_report: layer=%d kind=%s hostptr=%d nbytes=%llu\n",
                    layer_index, kind != nullptr ? kind : "(null)",
                    holder->buffer_uses_backend_base ? 1 : 0,
                    static_cast<unsigned long long>(nbytes));
            }
            if (holder->buffer == nullptr) {
                holder->buffer = ggml_backend_cpu_buffer_from_ptr(ptr, nbytes);
                ggml_backend_buffer_set_usage(holder->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
            found = holders.emplace(holder_key, holder).first;
        }
        holder = found->second;
    }

    ggml_tensor * tensor = ne[2] == 1 ?
        ggml_new_tensor_2d(ctx, type, ne[0], ne[1]) :
        ggml_new_tensor_3d(ctx, type, ne[0], ne[1], ne[2]);
    void * ptr = static_cast<std::uint8_t *>(mapping->mapping->addr()) + offset;
    void * tensor_addr = holder->buffer_uses_backend_base ?
        static_cast<std::uint8_t *>(ggml_backend_buffer_get_base(holder->buffer)) + holder->buffer_tensor_offset :
        ptr;
    if (ggml_backend_tensor_alloc(holder->buffer, tensor, tensor_addr) != GGML_STATUS_SUCCESS) {
        return nullptr;
    }
    ggml_format_name(tensor, "infinitum_v2_%s_%d", kind, layer_index);
    return tensor;
}

ggml_tensor * llama_infinitum_moe_ggml_pack_tensor_shaped(
        ggml_context * ctx,
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        const char * kind,
        ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    if (!llama_infinitum_moe_ggml_pack_enabled()) {
        llama_infinitum_debug_log("ggml pack shaped disabled layer=%d kind=%s", layer_index, kind != nullptr ? kind : "(null)");
        return nullptr;
    }
    const int experts = llama_infinitum_moe_expert_count(info);
    if (experts <= 0 || ne0 <= 0 || ne1 <= 0 || ne2 <= 0 || ne2 != experts) {
        llama_infinitum_debug_log(
                "ggml pack shaped invalid shape layer=%d kind=%s experts=%d ne=(%lld,%lld,%lld)",
                layer_index, kind != nullptr ? kind : "(null)", experts,
                (long long) ne0, (long long) ne1, (long long) ne2);
        return nullptr;
    }

    const std::size_t gate_up_all_bytes =
        ggml_row_size(type, static_cast<int64_t>(info.hidden_size)) *
        static_cast<std::size_t>(info.gate_up_rows) *
        static_cast<std::size_t>(experts);
    const std::size_t down_all_bytes =
        ggml_row_size(type, static_cast<int64_t>(info.down_rows)) *
        static_cast<std::size_t>(info.hidden_size) *
        static_cast<std::size_t>(experts);
    if (gate_up_all_bytes == 0 || down_all_bytes == 0) {
        return nullptr;
    }

    std::size_t offset = static_cast<std::size_t>(layer_index) * (gate_up_all_bytes + down_all_bytes);
    std::size_t nbytes = 0;
    if (std::strcmp(kind, "gate_up") == 0) {
        nbytes = gate_up_all_bytes;
    } else if (std::strcmp(kind, "down") == 0) {
        offset += gate_up_all_bytes;
        nbytes = down_all_bytes;
    } else {
        return nullptr;
    }

    const std::string pack_path = llama_infinitum_moe_ggml_pack_path(info);
    std::string error;
    auto mapping = llama_infinitum_moe_ggml_pack_mapping_for(pack_path, error);
    if (mapping == nullptr || offset + nbytes > mapping->file->size()) {
        llama_infinitum_debug_log(
                "ggml pack shaped mapping failed layer=%d kind=%s path=%s offset=%llu nbytes=%llu file_size=%llu error=%s",
                layer_index, kind != nullptr ? kind : "(null)", pack_path.c_str(),
                (unsigned long long) offset, (unsigned long long) nbytes,
                (unsigned long long) (mapping != nullptr ? mapping->file->size() : 0), error.c_str());
        return nullptr;
    }

    const std::string holder_key = pack_path + ":shaped:" + std::to_string(layer_index) + ":" + kind;
    static std::mutex holders_mutex;
    static std::unordered_map<std::string, std::shared_ptr<llama_infinitum_ggml_pack_tensor_holder>> holders;
    std::shared_ptr<llama_infinitum_ggml_pack_tensor_holder> holder;
    {
        std::lock_guard<std::mutex> lock(holders_mutex);
        auto found = holders.find(holder_key);
        if (found == holders.end()) {
            holder = std::make_shared<llama_infinitum_ggml_pack_tensor_holder>();
            holder->mapping = mapping;
            void * ptr = static_cast<std::uint8_t *>(mapping->mapping->addr()) + offset;
            holder->buffer = llama_infinitum_try_vulkan_hostptr_buffer(ptr, nbytes, &holder->buffer_tensor_offset);
            holder->buffer_uses_backend_base = holder->buffer != nullptr;
            if (llama_infinitum_pack_report_enabled()) {
                std::fprintf(stderr,
                    "infinitum_pack_report: layer=%d kind=%s hostptr=%d nbytes=%llu\n",
                    layer_index, kind != nullptr ? kind : "(null)",
                    holder->buffer_uses_backend_base ? 1 : 0,
                    static_cast<unsigned long long>(nbytes));
            }
            llama_infinitum_debug_log(
                    "ggml pack shaped buffer layer=%d kind=%s hostptr=%d nbytes=%llu",
                    layer_index, kind != nullptr ? kind : "(null)",
                    holder->buffer_uses_backend_base ? 1 : 0,
                    (unsigned long long) nbytes);
            if (holder->buffer == nullptr) {
                holder->buffer = ggml_backend_cpu_buffer_from_ptr(ptr, nbytes);
                if (holder->buffer == nullptr) {
                    llama_infinitum_debug_log(
                            "ggml pack shaped cpu buffer failed layer=%d kind=%s nbytes=%llu",
                            layer_index, kind != nullptr ? kind : "(null)", (unsigned long long) nbytes);
                    return nullptr;
                }
                ggml_backend_buffer_set_usage(holder->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
            found = holders.emplace(holder_key, holder).first;
        }
        holder = found->second;
    }

    ggml_tensor * tensor = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
    void * ptr = static_cast<std::uint8_t *>(mapping->mapping->addr()) + offset;
    void * tensor_addr = holder->buffer_uses_backend_base ?
        static_cast<std::uint8_t *>(ggml_backend_buffer_get_base(holder->buffer)) + holder->buffer_tensor_offset :
        ptr;
    const ggml_status alloc_status = ggml_backend_tensor_alloc(holder->buffer, tensor, tensor_addr);
    if (alloc_status != GGML_STATUS_SUCCESS) {
        llama_infinitum_debug_log(
                "ggml pack shaped tensor alloc failed layer=%d kind=%s status=%d hostptr=%d tensor_addr=%p",
                layer_index, kind != nullptr ? kind : "(null)", (int) alloc_status,
                holder->buffer_uses_backend_base ? 1 : 0, tensor_addr);
        return nullptr;
    }
    ggml_format_name(tensor, "infinitum_gemma4_%s_%d", kind, layer_index);
    return tensor;
}

static bool llama_infinitum_moe_map_slice(
        const llama_infinitum_moe_index_info & info,
        const llama_infinitum_moe_expert_slice & slice,
        llama_infinitum_moe_loaded_slice & out,
        std::string & error);

static std::size_t llama_infinitum_loaded_slice_size(const llama_infinitum_moe_loaded_slice & slice);

static std::size_t llama_infinitum_loaded_slice_process_resident_size(const llama_infinitum_moe_loaded_slice & slice);

static std::size_t llama_infinitum_loaded_slice_prepacked_size(const llama_infinitum_moe_loaded_slice & slice);

static const std::uint8_t * llama_infinitum_loaded_slice_data(const llama_infinitum_moe_loaded_slice & slice);

static void llama_infinitum_mxfp4_fill_table_scales(llama_infinitum_moe_loaded_slice & slice);

static void llama_infinitum_mxfp4_prepack_sdot_values(
        const std::uint8_t * blocks,
        std::size_t block_bytes,
        std::vector<std::int8_t> & out) {
    static const std::int8_t value_lut[16] = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    constexpr int packed_per_block = 16;
    constexpr int unpacked_per_block = packed_per_block * 2;
    const std::size_t block_count = block_bytes / packed_per_block;
    out.resize(block_count * unpacked_per_block);
    for (std::size_t block = 0; block < block_count; ++block) {
        const std::uint8_t * src = blocks + block * packed_per_block;
        std::int8_t * dst = out.data() + block * unpacked_per_block;
        for (int i = 0; i < packed_per_block; ++i) {
            const std::uint8_t packed = src[i];
            dst[2 * i] = value_lut[packed & 0x0f];
            dst[2 * i + 1] = value_lut[(packed >> 4) & 0x0f];
        }
    }
}

static int llama_infinitum_mxfp4_rows_for_slice(
        const llama_infinitum_moe_index_info & info,
        const llama_infinitum_moe_expert_slice & slice) {
    if (slice.kind == "gate_up_proj_blocks" || slice.kind == "down_proj_blocks") {
        const int block_count = llama_infinitum_mxfp4_block_count_for_hidden(info.hidden_size);
        constexpr std::uint64_t packed_per_block = 16;
        const std::uint64_t row_bytes = static_cast<std::uint64_t>(block_count) * packed_per_block;
        if (row_bytes > 0 && slice.byte_length >= row_bytes && slice.byte_length % row_bytes == 0) {
            return static_cast<int>(slice.byte_length / row_bytes);
        }
    }
    return 0;
}

static void llama_infinitum_mxfp4_prepack_kmajor_sdot_values(
        const std::uint8_t * blocks,
        int rows,
        int block_count,
        std::vector<std::int8_t> & out) {
    static const std::int8_t value_lut[16] = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    constexpr int packed_per_block = 16;
    constexpr int block_size = 32;
    constexpr int tile_rows = 8;
    if (rows <= 0 || block_count <= 0) {
        out.clear();
        return;
    }
    const int tiles = (rows + tile_rows - 1) / tile_rows;
    out.assign(std::size_t(tiles) * block_count * tile_rows * block_size, 0);
    for (int tile = 0; tile < tiles; ++tile) {
        for (int block = 0; block < block_count; ++block) {
            for (int r = 0; r < tile_rows; ++r) {
                const int row = tile * tile_rows + r;
                std::int8_t * dst = out.data() + (((tile * block_count + block) * tile_rows + r) * block_size);
                if (row >= rows) {
                    continue;
                }
                const int src_base = (row * block_count + block) * packed_per_block;
                for (int p = 0; p < packed_per_block; ++p) {
                    const std::uint8_t packed = blocks[src_base + p];
                    dst[2 * p] = value_lut[packed & 0x0f];
                    dst[2 * p + 1] = value_lut[(packed >> 4) & 0x0f];
                }
            }
        }
    }
}

llama_infinitum_moe_slice_cache::llama_infinitum_moe_slice_cache(std::uint64_t max_bytes) :
    max_bytes(max_bytes == 0 ? 1024ull * 1024ull * 1024ull : max_bytes) {
    items.reserve(8192);
}

bool llama_infinitum_moe_slice_cache::get_or_load(
        const llama_infinitum_moe_index_info & info,
        const llama_infinitum_moe_expert_slice & slice,
        llama_infinitum_moe_loaded_slice & out,
        std::string & error) {
    std::shared_ptr<const llama_infinitum_moe_loaded_slice> ptr = get_or_load_ptr(info, slice, error);
    if (ptr == nullptr) {
        return false;
    }
    out = *ptr;
    return true;
}

std::shared_ptr<const llama_infinitum_moe_loaded_slice> llama_infinitum_moe_slice_cache::get_or_load_ptr(
        const llama_infinitum_moe_index_info & info,
        const llama_infinitum_moe_expert_slice & slice,
        std::string & error) {
    std::lock_guard<std::mutex> lock(mutex);
    const std::string key = key_for(slice);
    auto found = items.find(key);
    if (found != items.end()) {
        cache_stats.cache_hits++;
        touch(key);
        return found->second;
    }

    cache_stats.cache_misses++;
    auto loaded = std::make_shared<llama_infinitum_moe_loaded_slice>();
    loaded->meta = slice;
    if (!llama_infinitum_moe_map_slice(info, slice, *loaded, error)) {
        error.clear();
        if (!llama_infinitum_moe_read_slice(info, slice, loaded->bytes, error)) {
            return nullptr;
        }
    }
    if (llama_infinitum_mxfp4_rowmajor_prepack_enabled() && slice.kind.find("_blocks") != std::string::npos) {
        llama_infinitum_mxfp4_prepack_sdot_values(
            llama_infinitum_loaded_slice_data(*loaded), llama_infinitum_loaded_slice_size(*loaded), loaded->sdot_values);
    }
    const int kmajor_rows = llama_infinitum_mxfp4_rows_for_slice(info, slice);
    if (llama_infinitum_mxfp4_kmajor_prepack_enabled() && kmajor_rows > 0) {
        const int block_count = llama_infinitum_mxfp4_block_count_for_hidden(info.hidden_size);
        llama_infinitum_mxfp4_prepack_kmajor_sdot_values(
            llama_infinitum_loaded_slice_data(*loaded), kmajor_rows, block_count, loaded->sdot_kmajor_values);
    }
    if (llama_infinitum_moe_scaled_pair_table_enabled() && slice.kind.find("_scales") != std::string::npos) {
        llama_infinitum_mxfp4_fill_table_scales(*loaded);
    }

    const std::uint64_t loaded_size = static_cast<std::uint64_t>(llama_infinitum_loaded_slice_size(*loaded));
    const std::uint64_t process_resident_size =
        static_cast<std::uint64_t>(llama_infinitum_loaded_slice_process_resident_size(*loaded));
    const std::uint64_t prepacked_size =
        static_cast<std::uint64_t>(llama_infinitum_loaded_slice_prepacked_size(*loaded));
    cache_stats.loaded_bytes += loaded_size;
    cache_stats.touched_bytes += slice.byte_length;
    cache_stats.mapped_bytes += loaded->mapped_data != nullptr ? slice.byte_length : 0;
    cache_stats.copied_bytes += loaded->mapped_data == nullptr ? slice.byte_length : 0;
    cache_stats.prepacked_bytes += prepacked_size;
    cache_stats.resident_bytes += loaded_size;
    cache_stats.process_resident_bytes += process_resident_size;
    items[key] = loaded;
    touch(key);
    evict_if_needed(key);
    return loaded;
}

llama_infinitum_moe_cache_stats llama_infinitum_moe_slice_cache::stats() const {
    std::lock_guard<std::mutex> lock(mutex);
    return cache_stats;
}

std::string llama_infinitum_moe_slice_cache::key_for(const llama_infinitum_moe_expert_slice & slice) {
    return std::to_string(slice.layer_index) + ":" + std::to_string(slice.expert_id) + ":" + slice.kind;
}

static std::size_t llama_infinitum_loaded_slice_size(const llama_infinitum_moe_loaded_slice & slice) {
    const std::size_t raw_size = slice.mapped_data != nullptr ? slice.mapped_size : slice.bytes.size();
    return raw_size + slice.sdot_values.size() + slice.sdot_kmajor_values.size() +
        slice.table_scale_values.size() * sizeof(float);
}

static std::size_t llama_infinitum_loaded_slice_process_resident_size(const llama_infinitum_moe_loaded_slice & slice) {
    return slice.bytes.size() + slice.sdot_values.size() + slice.sdot_kmajor_values.size() +
        slice.table_scale_values.size() * sizeof(float);
}

static std::size_t llama_infinitum_loaded_slice_prepacked_size(const llama_infinitum_moe_loaded_slice & slice) {
    return slice.sdot_values.size() + slice.sdot_kmajor_values.size() +
        slice.table_scale_values.size() * sizeof(float);
}

static const std::uint8_t * llama_infinitum_loaded_slice_data(const llama_infinitum_moe_loaded_slice & slice) {
    return slice.mapped_data != nullptr ? slice.mapped_data : slice.bytes.data();
}

void llama_infinitum_moe_slice_cache::touch(const std::string & key) {
    order.erase(std::remove(order.begin(), order.end(), key), order.end());
    order.push_back(key);
}

void llama_infinitum_moe_slice_cache::evict_if_needed(const std::string & protected_key) {
    while (cache_stats.resident_bytes > max_bytes && order.size() > 1) {
        const std::string victim = order.front();
        order.erase(order.begin());
        if (victim == protected_key) {
            order.push_back(victim);
            continue;
        }
        auto found = items.find(victim);
        if (found != items.end()) {
            cache_stats.resident_bytes -= static_cast<std::uint64_t>(llama_infinitum_loaded_slice_size(*found->second));
            cache_stats.process_resident_bytes -=
                static_cast<std::uint64_t>(llama_infinitum_loaded_slice_process_resident_size(*found->second));
            items.erase(found);
            cache_stats.evictions++;
        }
    }
}

bool llama_infinitum_moe_read_slice(
        const llama_infinitum_moe_index_info & info,
        const llama_infinitum_moe_expert_slice & slice,
        std::vector<std::uint8_t> & out,
        std::string & error) {
    out.clear();
    error.clear();

    const std::string path = llama_infinitum_join_path(info.base_dir, slice.source_file);
    if (path.empty()) {
        error = "expert slice source path is empty";
        return false;
    }
    if (slice.byte_length == 0) {
        error = "expert slice byte_length is zero";
        return false;
    }

    try {
        llama_file file(path.c_str(), "rb");
        if (slice.byte_offset + slice.byte_length > file.size()) {
            error = "expert slice exceeds source file size";
            return false;
        }

        out.resize(static_cast<std::size_t>(slice.byte_length));
        file.seek(static_cast<std::size_t>(slice.byte_offset), SEEK_SET);
        file.read_raw(out.data(), out.size());
        return true;
    } catch (const std::exception & ex) {
        error = ex.what();
        return false;
    }
}

struct llama_infinitum_mapped_source {
    std::unique_ptr<llama_file> file;
    std::unique_ptr<llama_mmap> mapping;
};

static bool llama_infinitum_moe_map_slice(
        const llama_infinitum_moe_index_info & info,
        const llama_infinitum_moe_expert_slice & slice,
        llama_infinitum_moe_loaded_slice & out,
        std::string & error) {
    if (!llama_mmap::SUPPORTED || llama_infinitum_env_disabled("LLAMA_INFINITUM_EXPERT_MMAP")) {
        return false;
    }

    const std::string path = llama_infinitum_join_path(info.base_dir, slice.source_file);
    static std::mutex mapped_sources_mutex;
    static std::unordered_map<std::string, std::shared_ptr<llama_infinitum_mapped_source>> mapped_sources;

    std::lock_guard<std::mutex> lock(mapped_sources_mutex);
    auto found = mapped_sources.find(path);
    if (found == mapped_sources.end()) {
        auto source = std::make_shared<llama_infinitum_mapped_source>();
        try {
            source->file = std::make_unique<llama_file>(path.c_str(), "rb");
            source->mapping = std::make_unique<llama_mmap>(source->file.get(), 0, false);
        } catch (const std::exception & ex) {
            error = ex.what();
            return false;
        }
        found = mapped_sources.emplace(path, std::move(source)).first;
    }

    llama_infinitum_mapped_source & source = *found->second;
    if (slice.byte_offset + slice.byte_length > source.file->size()) {
        error = "expert slice exceeds source file size";
        return false;
    }
    const std::uint8_t * base = static_cast<const std::uint8_t *>(source.mapping->addr());
    out.mapped_source = found->second;
    out.mapped_data = base + slice.byte_offset;
    out.mapped_size = static_cast<std::size_t>(slice.byte_length);
    return true;
}

static float llama_infinitum_bf16_at(const std::uint8_t * bytes, int index) {
    const std::uint16_t hi = std::uint16_t(bytes[2 * index]) | (std::uint16_t(bytes[2 * index + 1]) << 8);
    const std::uint32_t raw = std::uint32_t(hi) << 16;
    float value = 0.0f;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

static std::uint8_t llama_infinitum_mxfp4_external_nibble(const std::uint8_t * block, int value_index) {
    const std::uint8_t packed = block[value_index / 2];
    return (value_index & 1) != 0 ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
}

static void llama_infinitum_mxfp4_pack_ggml_block(
        const std::uint8_t * external_block,
        std::uint8_t scale,
        std::uint8_t * ggml_block) {
    ggml_block[0] = scale;
    for (int j = 0; j < 16; ++j) {
        const std::uint8_t lo = llama_infinitum_mxfp4_external_nibble(external_block, j);
        const std::uint8_t hi = llama_infinitum_mxfp4_external_nibble(external_block, j + 16);
        ggml_block[1 + j] = std::uint8_t(lo | (hi << 4));
    }
}

static float llama_infinitum_mxfp4_value(std::uint8_t packed, bool high) {
    static const float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    const int nibble = high ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
    return lut[nibble];
}

static const float * llama_infinitum_mxfp4_pair_lut() {
    static const std::array<float, 512> lut = []() {
        std::array<float, 512> values = {};
        for (int packed = 0; packed < 256; ++packed) {
            values[2 * packed + 0] = llama_infinitum_mxfp4_value(static_cast<std::uint8_t>(packed), false);
            values[2 * packed + 1] = llama_infinitum_mxfp4_value(static_cast<std::uint8_t>(packed), true);
        }
        return values;
    }();
    return lut.data();
}

static const float * llama_infinitum_mxfp4_scale_lut() {
    static const std::array<float, 256> lut = []() {
        std::array<float, 256> values = {};
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = std::scalbn(1.0f, int(i) - 127);
        }
        return values;
    }();
    return lut.data();
}

static const float * llama_infinitum_mxfp4_scaled_pair_lut() {
    static const std::array<float, 256 * 256 * 2> lut = []() {
        std::array<float, 256 * 256 * 2> values = {};
        const float * pair_lut = llama_infinitum_mxfp4_pair_lut();
        const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
        for (int scale = 0; scale < 256; ++scale) {
            for (int packed = 0; packed < 256; ++packed) {
                const std::size_t base = (std::size_t(scale) * 256u + std::size_t(packed)) * 2u;
                values[base + 0] = pair_lut[2 * packed + 0] * scale_lut[scale];
                values[base + 1] = pair_lut[2 * packed + 1] * scale_lut[scale];
            }
        }
        return values;
    }();
    return lut.data();
}

static void llama_infinitum_mxfp4_fill_table_scales(llama_infinitum_moe_loaded_slice & slice) {
    const std::uint8_t * data = llama_infinitum_loaded_slice_data(slice);
    const std::size_t size = slice.mapped_data != nullptr ? slice.mapped_size : slice.bytes.size();
    if (data == nullptr || size == 0) {
        return;
    }
    const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
    slice.table_scale_values.resize(size);
    for (std::size_t i = 0; i < size; ++i) {
        slice.table_scale_values[i] = scale_lut[data[i]];
    }
}

static void llama_infinitum_mxfp4_fill_input_pair_table(
        const float * input,
        int hidden_size,
        std::vector<float> & table) {
    constexpr int values_per_pair = 256;
    const int pair_count = hidden_size / 2;
    if (input == nullptr || pair_count <= 0) {
        table.clear();
        return;
    }
    table.resize(std::size_t(pair_count) * values_per_pair);
    const float * pair_lut = llama_infinitum_mxfp4_pair_lut();
    for (int pair = 0; pair < pair_count; ++pair) {
        const float in_0 = input[pair * 2 + 0];
        const float in_1 = input[pair * 2 + 1];
        float * dst = table.data() + std::size_t(pair) * values_per_pair;
        for (int packed = 0; packed < values_per_pair; ++packed) {
            const float * weights = pair_lut + 2 * packed;
            dst[packed] = weights[0] * in_0 + weights[1] * in_1;
        }
    }
}

static void llama_infinitum_quantize_input_32(
        const float * input,
        int hidden_size,
        llama_infinitum_quantized_input & quantized) {
    constexpr int block_size = 32;
    const int block_count = (hidden_size + block_size - 1) / block_size;
    quantized.values.resize(hidden_size);
    quantized.scales.resize(block_count);
    for (int block = 0; block < block_count; ++block) {
        const int base = block * block_size;
        const int end = std::min(hidden_size, base + block_size);
        float max_abs = 0.0f;
        for (int i = base; i < end; ++i) {
            max_abs = std::max(max_abs, std::fabs(input[i]));
        }
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        quantized.scales[block] = scale;
        for (int i = base; i < end; ++i) {
            const float scaled = std::round(input[i] * inv_scale);
            const float clamped = std::max(-127.0f, std::min(127.0f, scaled));
            quantized.values[i] = static_cast<std::int8_t>(clamped);
        }
    }
}

static bool llama_infinitum_mxfp4_sdot_requested() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_EXPERT_SDOT");
}

static bool llama_infinitum_mxfp4_sdot_available() {
#if defined(__ARM_NEON) && defined(__aarch64__) && defined(__linux__) && defined(HWCAP_ASIMDDP)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    return false;
#endif
}

#if defined(__ARM_NEON) && defined(__aarch64__)
__attribute__((target("+dotprod")))
static inline int32x4_t llama_infinitum_mxfp4_sdot_acc_packed(
        uint8x16_t packed,
        int8x16_t input_0,
        int8x16_t input_1,
        uint8x16_t nibble_mask,
        int8x16_t value_lut) {
    const uint8x16_t low_nibbles = vandq_u8(packed, nibble_mask);
    const uint8x16_t high_nibbles = vandq_u8(vshrq_n_u8(packed, 4), nibble_mask);
    const int8x16_t low_values = vqtbl1q_s8(value_lut, low_nibbles);
    const int8x16_t high_values = vqtbl1q_s8(value_lut, high_nibbles);
    const int8x16_t weights_0 = vzip1q_s8(low_values, high_values);
    const int8x16_t weights_1 = vzip2q_s8(low_values, high_values);
    int32x4_t acc = vdupq_n_s32(0);
    acc = vdotq_s32(acc, weights_0, input_0);
    acc = vdotq_s32(acc, weights_1, input_1);
    return acc;
}

__attribute__((target("+dotprod")))
static float llama_infinitum_mxfp4_dot_row_sdot(
        const std::uint8_t * blocks,
        const std::uint8_t * scales,
        int row,
        const llama_infinitum_quantized_input & input) {
    constexpr int block_count = 90;
    constexpr int packed_per_block = 16;
    constexpr int block_size = packed_per_block * 2;
    const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);
    const int8x16_t value_lut = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    float total = 0.0f;
    for (int block = 0; block < block_count; ++block) {
        const int block_base = (row * block_count + block) * packed_per_block;
        const int input_base = block * block_size;
        const uint8x16_t packed = vld1q_u8(blocks + block_base);
        const int8x16_t input_0 = vld1q_s8(input.values.data() + input_base);
        const int8x16_t input_1 = vld1q_s8(input.values.data() + input_base + 16);
        const int32x4_t acc = llama_infinitum_mxfp4_sdot_acc_packed(packed, input_0, input_1, nibble_mask, value_lut);
        const float scale = scale_lut[scales[row * block_count + block]] * 0.5f * input.scales[block];
        total += float(vaddvq_s32(acc)) * scale;
    }
    return total;
}

__attribute__((target("+dotprod")))
static void llama_infinitum_mxfp4_dot_rows4_sdot(
        const std::uint8_t * blocks,
        const std::uint8_t * scales,
        int row,
        const llama_infinitum_quantized_input & input,
        float * out) {
    constexpr int block_count = 90;
    constexpr int packed_per_block = 16;
    constexpr int block_size = packed_per_block * 2;
    const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);
    const int8x16_t value_lut = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    float total_0 = 0.0f;
    float total_1 = 0.0f;
    float total_2 = 0.0f;
    float total_3 = 0.0f;
    for (int block = 0; block < block_count; ++block) {
        const int input_base = block * block_size;
        const int8x16_t input_0 = vld1q_s8(input.values.data() + input_base);
        const int8x16_t input_1 = vld1q_s8(input.values.data() + input_base + 16);
        const float input_scale = 0.5f * input.scales[block];

        const int row_0 = row;
        const int block_base_0 = (row_0 * block_count + block) * packed_per_block;
        const int32x4_t acc_0 = llama_infinitum_mxfp4_sdot_acc_packed(vld1q_u8(blocks + block_base_0), input_0, input_1, nibble_mask, value_lut);
        total_0 += float(vaddvq_s32(acc_0)) * scale_lut[scales[row_0 * block_count + block]] * input_scale;

        const int row_1 = row + 1;
        const int block_base_1 = (row_1 * block_count + block) * packed_per_block;
        const int32x4_t acc_1 = llama_infinitum_mxfp4_sdot_acc_packed(vld1q_u8(blocks + block_base_1), input_0, input_1, nibble_mask, value_lut);
        total_1 += float(vaddvq_s32(acc_1)) * scale_lut[scales[row_1 * block_count + block]] * input_scale;

        const int row_2 = row + 2;
        const int block_base_2 = (row_2 * block_count + block) * packed_per_block;
        const int32x4_t acc_2 = llama_infinitum_mxfp4_sdot_acc_packed(vld1q_u8(blocks + block_base_2), input_0, input_1, nibble_mask, value_lut);
        total_2 += float(vaddvq_s32(acc_2)) * scale_lut[scales[row_2 * block_count + block]] * input_scale;

        const int row_3 = row + 3;
        const int block_base_3 = (row_3 * block_count + block) * packed_per_block;
        const int32x4_t acc_3 = llama_infinitum_mxfp4_sdot_acc_packed(vld1q_u8(blocks + block_base_3), input_0, input_1, nibble_mask, value_lut);
        total_3 += float(vaddvq_s32(acc_3)) * scale_lut[scales[row_3 * block_count + block]] * input_scale;
    }
    out[0] = total_0;
    out[1] = total_1;
    out[2] = total_2;
    out[3] = total_3;
}

__attribute__((target("+dotprod")))
static void llama_infinitum_mxfp4_dot_rows8_sdot(
        const std::uint8_t * blocks,
        const std::uint8_t * scales,
        int row,
        const llama_infinitum_quantized_input & input,
        float * out) {
    constexpr int block_count = 90;
    constexpr int packed_per_block = 16;
    constexpr int block_size = packed_per_block * 2;
    const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);
    const int8x16_t value_lut = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    float32x4_t totals[8] = {
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
    };
    for (int block = 0; block < block_count; ++block) {
        const int input_base = block * block_size;
        const int8x16_t input_0 = vld1q_s8(input.values.data() + input_base);
        const int8x16_t input_1 = vld1q_s8(input.values.data() + input_base + 16);
        const float input_scale = 0.5f * input.scales[block];
        if (block + 1 < block_count) {
            __builtin_prefetch(input.values.data() + input_base + block_size, 0, 1);
        }
        for (int r = 0; r < 8; ++r) {
            const int current_row = row + r;
            const int block_base = (current_row * block_count + block) * packed_per_block;
            if (block + 1 < block_count) {
                __builtin_prefetch(blocks + block_base + packed_per_block, 0, 1);
                __builtin_prefetch(scales + current_row * block_count + block + 1, 0, 1);
            }
            const int32x4_t acc = llama_infinitum_mxfp4_sdot_acc_packed(
                vld1q_u8(blocks + block_base), input_0, input_1, nibble_mask, value_lut);
            const float scale = scale_lut[scales[current_row * block_count + block]] * input_scale;
            totals[r] = vmlaq_n_f32(totals[r], vcvtq_f32_s32(acc), scale);
        }
    }
    for (int r = 0; r < 8; ++r) {
        out[r] = vaddvq_f32(totals[r]);
    }
}

__attribute__((target("+dotprod")))
static void llama_infinitum_mxfp4_dot_rows8_sdot_kmajor(
        const std::int8_t * values,
        const std::uint8_t * scales,
        int row,
        const llama_infinitum_quantized_input & input,
        float * out) {
    constexpr int block_count = 90;
    constexpr int block_size = 32;
    constexpr int tile_rows = 8;
    const int tile = row / tile_rows;
    const int row_in_tile = row % tile_rows;
    const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
    float32x4_t totals[8] = {
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
    };
    for (int block = 0; block < block_count; ++block) {
        const int input_base = block * block_size;
        const int8x16_t input_0 = vld1q_s8(input.values.data() + input_base);
        const int8x16_t input_1 = vld1q_s8(input.values.data() + input_base + 16);
        const float input_scale = 0.5f * input.scales[block];
        if (block + 1 < block_count) {
            __builtin_prefetch(input.values.data() + input_base + block_size, 0, 1);
        }
        for (int r = 0; r < tile_rows; ++r) {
            const int current_row = row + r;
            const std::int8_t * weights = values + ((((tile * block_count + block) * tile_rows + row_in_tile + r) * block_size));
            if (block + 1 < block_count) {
                __builtin_prefetch(weights + tile_rows * block_size, 0, 1);
                __builtin_prefetch(scales + current_row * block_count + block + 1, 0, 1);
            }
            int32x4_t acc = vdupq_n_s32(0);
            acc = vdotq_s32(acc, vld1q_s8(weights), input_0);
            acc = vdotq_s32(acc, vld1q_s8(weights + 16), input_1);
            const float scale = scale_lut[scales[current_row * block_count + block]] * input_scale;
            totals[r] = vmlaq_n_f32(totals[r], vcvtq_f32_s32(acc), scale);
        }
    }
    for (int r = 0; r < tile_rows; ++r) {
        out[r] = vaddvq_f32(totals[r]);
    }
}

__attribute__((target("+dotprod")))
static float llama_infinitum_mxfp4_dot_row_sdot_prepacked(
        const std::int8_t * values,
        const std::uint8_t * scales,
        int row,
        const llama_infinitum_quantized_input & input) {
    constexpr int block_count = 90;
    constexpr int block_size = 32;
    const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
    float total = 0.0f;
    for (int block = 0; block < block_count; ++block) {
        const int value_base = (row * block_count + block) * block_size;
        const int input_base = block * block_size;
        const int8x16_t weights_0 = vld1q_s8(values + value_base);
        const int8x16_t weights_1 = vld1q_s8(values + value_base + 16);
        const int8x16_t input_0 = vld1q_s8(input.values.data() + input_base);
        const int8x16_t input_1 = vld1q_s8(input.values.data() + input_base + 16);
        int32x4_t acc = vdupq_n_s32(0);
        acc = vdotq_s32(acc, weights_0, input_0);
        acc = vdotq_s32(acc, weights_1, input_1);
        const float scale = scale_lut[scales[row * block_count + block]] * 0.5f * input.scales[block];
        total += float(vaddvq_s32(acc)) * scale;
    }
    return total;
}

__attribute__((target("+dotprod")))
static void llama_infinitum_mxfp4_dot_row_pair_sdot(
        const std::uint8_t * blocks,
        const std::uint8_t * scales,
        int row,
        const llama_infinitum_quantized_input & input,
        float & out_0,
        float & out_1) {
    constexpr int block_count = 90;
    constexpr int packed_per_block = 16;
    constexpr int block_size = packed_per_block * 2;
    const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);
    const int8x16_t value_lut = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    float total_0 = 0.0f;
    float total_1 = 0.0f;
    for (int block = 0; block < block_count; ++block) {
        const int block_base_0 = (row * block_count + block) * packed_per_block;
        const int block_base_1 = ((row + 1) * block_count + block) * packed_per_block;
        const int input_base = block * block_size;
        const int8x16_t input_0 = vld1q_s8(input.values.data() + input_base);
        const int8x16_t input_1 = vld1q_s8(input.values.data() + input_base + 16);

        const uint8x16_t packed_0 = vld1q_u8(blocks + block_base_0);
        const uint8x16_t low_nibbles_0 = vandq_u8(packed_0, nibble_mask);
        const uint8x16_t high_nibbles_0 = vandq_u8(vshrq_n_u8(packed_0, 4), nibble_mask);
        const int8x16_t low_values_0 = vqtbl1q_s8(value_lut, low_nibbles_0);
        const int8x16_t high_values_0 = vqtbl1q_s8(value_lut, high_nibbles_0);
        const int8x16_t weights_00 = vzip1q_s8(low_values_0, high_values_0);
        const int8x16_t weights_01 = vzip2q_s8(low_values_0, high_values_0);
        int32x4_t acc_0 = vdupq_n_s32(0);
        acc_0 = vdotq_s32(acc_0, weights_00, input_0);
        acc_0 = vdotq_s32(acc_0, weights_01, input_1);

        const uint8x16_t packed_1 = vld1q_u8(blocks + block_base_1);
        const uint8x16_t low_nibbles_1 = vandq_u8(packed_1, nibble_mask);
        const uint8x16_t high_nibbles_1 = vandq_u8(vshrq_n_u8(packed_1, 4), nibble_mask);
        const int8x16_t low_values_1 = vqtbl1q_s8(value_lut, low_nibbles_1);
        const int8x16_t high_values_1 = vqtbl1q_s8(value_lut, high_nibbles_1);
        const int8x16_t weights_10 = vzip1q_s8(low_values_1, high_values_1);
        const int8x16_t weights_11 = vzip2q_s8(low_values_1, high_values_1);
        int32x4_t acc_1 = vdupq_n_s32(0);
        acc_1 = vdotq_s32(acc_1, weights_10, input_0);
        acc_1 = vdotq_s32(acc_1, weights_11, input_1);

        const float input_scale = 0.5f * input.scales[block];
        total_0 += float(vaddvq_s32(acc_0)) * scale_lut[scales[row * block_count + block]] * input_scale;
        total_1 += float(vaddvq_s32(acc_1)) * scale_lut[scales[(row + 1) * block_count + block]] * input_scale;
    }
    out_0 = total_0;
    out_1 = total_1;
}

__attribute__((target("+dotprod")))
static void llama_infinitum_mxfp4_dot_row_pair_sdot_prepacked(
        const std::int8_t * values,
        const std::uint8_t * scales,
        int row,
        const llama_infinitum_quantized_input & input,
        float & out_0,
        float & out_1) {
    constexpr int block_count = 90;
    constexpr int block_size = 32;
    const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
    float total_0 = 0.0f;
    float total_1 = 0.0f;
    for (int block = 0; block < block_count; ++block) {
        const int value_base_0 = (row * block_count + block) * block_size;
        const int value_base_1 = ((row + 1) * block_count + block) * block_size;
        const int input_base = block * block_size;
        const int8x16_t input_0 = vld1q_s8(input.values.data() + input_base);
        const int8x16_t input_1 = vld1q_s8(input.values.data() + input_base + 16);

        const int8x16_t weights_00 = vld1q_s8(values + value_base_0);
        const int8x16_t weights_01 = vld1q_s8(values + value_base_0 + 16);
        int32x4_t acc_0 = vdupq_n_s32(0);
        acc_0 = vdotq_s32(acc_0, weights_00, input_0);
        acc_0 = vdotq_s32(acc_0, weights_01, input_1);

        const int8x16_t weights_10 = vld1q_s8(values + value_base_1);
        const int8x16_t weights_11 = vld1q_s8(values + value_base_1 + 16);
        int32x4_t acc_1 = vdupq_n_s32(0);
        acc_1 = vdotq_s32(acc_1, weights_10, input_0);
        acc_1 = vdotq_s32(acc_1, weights_11, input_1);

        const float input_scale = 0.5f * input.scales[block];
        total_0 += float(vaddvq_s32(acc_0)) * scale_lut[scales[row * block_count + block]] * input_scale;
        total_1 += float(vaddvq_s32(acc_1)) * scale_lut[scales[(row + 1) * block_count + block]] * input_scale;
    }
    out_0 = total_0;
    out_1 = total_1;
}

static inline float32x4_t llama_infinitum_i8x4_to_f32(int8x16_t values, int offset) {
    const int8x8_t values8 = offset < 8 ? vget_low_s8(values) : vget_high_s8(values);
    const int16x8_t values16 = vmovl_s8(values8);
    const int16x4_t values16_4 = (offset & 4) == 0 ? vget_low_s16(values16) : vget_high_s16(values16);
    return vcvtq_f32_s32(vmovl_s16(values16_4));
}

static inline void llama_infinitum_mxfp4_accum4(
        float32x4_t & sum,
        int8x16_t low_values,
        int8x16_t high_values,
        const float * input,
        int offset) {
    const float32x4x2_t input_pairs = vld2q_f32(input + offset * 2);
    const float32x4_t low = llama_infinitum_i8x4_to_f32(low_values, offset);
    const float32x4_t high = llama_infinitum_i8x4_to_f32(high_values, offset);
    sum = vmlaq_f32(sum, low, input_pairs.val[0]);
    sum = vmlaq_f32(sum, high, input_pairs.val[1]);
}

static inline void llama_infinitum_mxfp4_accum4_pair(
        float32x4_t & sum_0,
        float32x4_t & sum_1,
        int8x16_t low_values_0,
        int8x16_t high_values_0,
        int8x16_t low_values_1,
        int8x16_t high_values_1,
        const float * input,
        int offset) {
    const float32x4x2_t input_pairs = vld2q_f32(input + offset * 2);
    const float32x4_t low_0 = llama_infinitum_i8x4_to_f32(low_values_0, offset);
    const float32x4_t high_0 = llama_infinitum_i8x4_to_f32(high_values_0, offset);
    const float32x4_t low_1 = llama_infinitum_i8x4_to_f32(low_values_1, offset);
    const float32x4_t high_1 = llama_infinitum_i8x4_to_f32(high_values_1, offset);
    sum_0 = vmlaq_f32(sum_0, low_0, input_pairs.val[0]);
    sum_0 = vmlaq_f32(sum_0, high_0, input_pairs.val[1]);
    sum_1 = vmlaq_f32(sum_1, low_1, input_pairs.val[0]);
    sum_1 = vmlaq_f32(sum_1, high_1, input_pairs.val[1]);
}

static float llama_infinitum_mxfp4_dot_row_neon(
        const std::uint8_t * blocks,
        const std::uint8_t * scales,
        int row,
        const float * input) {
    constexpr int block_count = 90;
    constexpr int packed_per_block = 16;
    const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);
    const int8x16_t value_lut = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    float total = 0.0f;
    for (int block = 0; block < block_count; ++block) {
        const int block_base = (row * block_count + block) * packed_per_block;
        const int input_base = block * packed_per_block * 2;
        const uint8x16_t packed = vld1q_u8(blocks + block_base);
        const uint8x16_t low_nibbles = vandq_u8(packed, nibble_mask);
        const uint8x16_t high_nibbles = vandq_u8(vshrq_n_u8(packed, 4), nibble_mask);
        const int8x16_t low_values = vqtbl1q_s8(value_lut, low_nibbles);
        const int8x16_t high_values = vqtbl1q_s8(value_lut, high_nibbles);
        float32x4_t sum = vdupq_n_f32(0.0f);
        llama_infinitum_mxfp4_accum4(sum, low_values, high_values, input + input_base, 0);
        llama_infinitum_mxfp4_accum4(sum, low_values, high_values, input + input_base, 4);
        llama_infinitum_mxfp4_accum4(sum, low_values, high_values, input + input_base, 8);
        llama_infinitum_mxfp4_accum4(sum, low_values, high_values, input + input_base, 12);
        const float scale = scale_lut[scales[row * block_count + block]] * 0.5f;
        total += vaddvq_f32(sum) * scale;
    }
    return total;
}

static void llama_infinitum_mxfp4_dot_row_pair_neon(
        const std::uint8_t * blocks,
        const std::uint8_t * scales,
        int row,
        const float * input,
        float & out_0,
        float & out_1) {
    constexpr int block_count = 90;
    constexpr int packed_per_block = 16;
    const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);
    const int8x16_t value_lut = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    float total_0 = 0.0f;
    float total_1 = 0.0f;
    for (int block = 0; block < block_count; ++block) {
        const int block_base_0 = (row * block_count + block) * packed_per_block;
        const int block_base_1 = ((row + 1) * block_count + block) * packed_per_block;
        const int input_base = block * packed_per_block * 2;
        const uint8x16_t packed_0 = vld1q_u8(blocks + block_base_0);
        const uint8x16_t packed_1 = vld1q_u8(blocks + block_base_1);
        const uint8x16_t low_nibbles_0 = vandq_u8(packed_0, nibble_mask);
        const uint8x16_t high_nibbles_0 = vandq_u8(vshrq_n_u8(packed_0, 4), nibble_mask);
        const uint8x16_t low_nibbles_1 = vandq_u8(packed_1, nibble_mask);
        const uint8x16_t high_nibbles_1 = vandq_u8(vshrq_n_u8(packed_1, 4), nibble_mask);
        const int8x16_t low_values_0 = vqtbl1q_s8(value_lut, low_nibbles_0);
        const int8x16_t high_values_0 = vqtbl1q_s8(value_lut, high_nibbles_0);
        const int8x16_t low_values_1 = vqtbl1q_s8(value_lut, low_nibbles_1);
        const int8x16_t high_values_1 = vqtbl1q_s8(value_lut, high_nibbles_1);
        float32x4_t sum_0 = vdupq_n_f32(0.0f);
        float32x4_t sum_1 = vdupq_n_f32(0.0f);
        llama_infinitum_mxfp4_accum4_pair(sum_0, sum_1, low_values_0, high_values_0, low_values_1, high_values_1, input + input_base, 0);
        llama_infinitum_mxfp4_accum4_pair(sum_0, sum_1, low_values_0, high_values_0, low_values_1, high_values_1, input + input_base, 4);
        llama_infinitum_mxfp4_accum4_pair(sum_0, sum_1, low_values_0, high_values_0, low_values_1, high_values_1, input + input_base, 8);
        llama_infinitum_mxfp4_accum4_pair(sum_0, sum_1, low_values_0, high_values_0, low_values_1, high_values_1, input + input_base, 12);
        total_0 += vaddvq_f32(sum_0) * scale_lut[scales[row * block_count + block]] * 0.5f;
        total_1 += vaddvq_f32(sum_1) * scale_lut[scales[(row + 1) * block_count + block]] * 0.5f;
    }
    out_0 = total_0;
    out_1 = total_1;
}

static void llama_infinitum_mxfp4_dot_rows8_neon(
        const std::uint8_t * blocks,
        const std::uint8_t * scales,
        int row,
        const float * input,
        float * out) {
    constexpr int block_count = 90;
    constexpr int packed_per_block = 16;
    constexpr int block_size = packed_per_block * 2;
    constexpr int tile_rows = 8;
    const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);
    const int8x16_t value_lut = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    float totals[tile_rows] = {};
    for (int block = 0; block < block_count; ++block) {
        int8x16_t low_values[tile_rows];
        int8x16_t high_values[tile_rows];
        for (int r = 0; r < tile_rows; ++r) {
            const int block_base = ((row + r) * block_count + block) * packed_per_block;
            const uint8x16_t packed = vld1q_u8(blocks + block_base);
            low_values[r] = vqtbl1q_s8(value_lut, vandq_u8(packed, nibble_mask));
            high_values[r] = vqtbl1q_s8(value_lut, vandq_u8(vshrq_n_u8(packed, 4), nibble_mask));
        }
        float32x4_t sums[tile_rows] = {
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        };
        const int input_base = block * block_size;
        for (int offset = 0; offset < 16; offset += 4) {
            const float32x4x2_t input_pairs = vld2q_f32(input + input_base + offset * 2);
            for (int r = 0; r < tile_rows; ++r) {
                const float32x4_t low = llama_infinitum_i8x4_to_f32(low_values[r], offset);
                const float32x4_t high = llama_infinitum_i8x4_to_f32(high_values[r], offset);
                sums[r] = vmlaq_f32(sums[r], low, input_pairs.val[0]);
                sums[r] = vmlaq_f32(sums[r], high, input_pairs.val[1]);
            }
        }
        for (int r = 0; r < tile_rows; ++r) {
            totals[r] += vaddvq_f32(sums[r]) * scale_lut[scales[(row + r) * block_count + block]] * 0.5f;
        }
    }
    for (int r = 0; r < tile_rows; ++r) {
        out[r] = totals[r];
    }
}
#endif

static float llama_infinitum_mxfp4_dot_row(
        const std::uint8_t * blocks,
        const std::uint8_t * scales,
        int row,
        int block_count,
        const float * input) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    (void) block_count;
    return llama_infinitum_mxfp4_dot_row_neon(blocks, scales, row, input);
#else
    constexpr int packed_per_block = 16;
    const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
    const float * pair_lut = llama_infinitum_mxfp4_pair_lut();
    float sum = 0.0f;
    for (int block = 0; block < block_count; ++block) {
        const float scale = scale_lut[scales[row * block_count + block]];
        const int block_base = (row * block_count + block) * packed_per_block;
        const int input_base = block * packed_per_block * 2;
        float block_sum = 0.0f;
        for (int p = 0; p < packed_per_block; ++p) {
            const std::uint8_t packed = blocks[block_base + p];
            const float * weights = pair_lut + 2 * packed;
            block_sum += weights[0] * input[input_base + p * 2] + weights[1] * input[input_base + p * 2 + 1];
        }
        sum += scale * block_sum;
    }
    return sum;
#endif
}

static void llama_infinitum_mxfp4_dot_row_pair_scalar(
        const std::uint8_t * blocks,
        const std::uint8_t * scales,
        int row,
        int block_count,
        const float * input,
        float & out_0,
        float & out_1) {
    constexpr int packed_per_block = 16;
    const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
    const float * pair_lut = llama_infinitum_mxfp4_pair_lut();
    float sum_0 = 0.0f;
    float sum_1 = 0.0f;
    for (int block = 0; block < block_count; ++block) {
        const float scale_0 = scale_lut[scales[row * block_count + block]];
        const float scale_1 = scale_lut[scales[(row + 1) * block_count + block]];
        const int block_base_0 = (row * block_count + block) * packed_per_block;
        const int block_base_1 = ((row + 1) * block_count + block) * packed_per_block;
        const int input_base = block * packed_per_block * 2;
        float block_sum_0 = 0.0f;
        float block_sum_1 = 0.0f;
        for (int p = 0; p < packed_per_block; ++p) {
            const float in_0 = input[input_base + p * 2];
            const float in_1 = input[input_base + p * 2 + 1];
            const float * weights_0 = pair_lut + 2 * blocks[block_base_0 + p];
            const float * weights_1 = pair_lut + 2 * blocks[block_base_1 + p];
            block_sum_0 += weights_0[0] * in_0 + weights_0[1] * in_1;
            block_sum_1 += weights_1[0] * in_0 + weights_1[1] * in_1;
        }
        sum_0 += scale_0 * block_sum_0;
        sum_1 += scale_1 * block_sum_1;
    }
    out_0 = sum_0;
    out_1 = sum_1;
}

static void llama_infinitum_mxfp4_dot_row_pair_table(
        const std::uint8_t * blocks,
        const std::uint8_t * scales,
        const float * scale_values,
        int row,
        int block_count,
        const float * input,
        float & out_0,
        float & out_1) {
    constexpr int packed_per_block = 16;
    const float * pair_lut = llama_infinitum_mxfp4_pair_lut();
    const float * scaled_pair_lut = llama_infinitum_mxfp4_scaled_pair_lut();
    float sum_0 = 0.0f;
    float sum_1 = 0.0f;
    for (int block = 0; block < block_count; ++block) {
        const int scale_index_0 = (row * block_count + block);
        const int scale_index_1 = ((row + 1) * block_count + block);
        const int block_base_0 = scale_index_0 * packed_per_block;
        const int block_base_1 = scale_index_1 * packed_per_block;
        const int input_base = block * packed_per_block * 2;
        float block_sum_0 = 0.0f;
        float block_sum_1 = 0.0f;
        if (scale_values != nullptr) {
            const float scale_0 = scale_values[scale_index_0];
            const float scale_1 = scale_values[scale_index_1];
            for (int p = 0; p < packed_per_block; ++p) {
                const float in_0 = input[input_base + p * 2];
                const float in_1 = input[input_base + p * 2 + 1];
                const float * weights_0 = pair_lut + 2 * blocks[block_base_0 + p];
                const float * weights_1 = pair_lut + 2 * blocks[block_base_1 + p];
                block_sum_0 += weights_0[0] * in_0 + weights_0[1] * in_1;
                block_sum_1 += weights_1[0] * in_0 + weights_1[1] * in_1;
            }
            sum_0 += scale_0 * block_sum_0;
            sum_1 += scale_1 * block_sum_1;
        } else {
            const int scale_0 = scales[scale_index_0];
            const int scale_1 = scales[scale_index_1];
            for (int p = 0; p < packed_per_block; ++p) {
                const float in_0 = input[input_base + p * 2];
                const float in_1 = input[input_base + p * 2 + 1];
                const float * weights_0 = scaled_pair_lut + ((scale_0 * 256 + blocks[block_base_0 + p]) * 2);
                const float * weights_1 = scaled_pair_lut + ((scale_1 * 256 + blocks[block_base_1 + p]) * 2);
                block_sum_0 += weights_0[0] * in_0 + weights_0[1] * in_1;
                block_sum_1 += weights_1[0] * in_0 + weights_1[1] * in_1;
            }
            sum_0 += block_sum_0;
            sum_1 += block_sum_1;
        }
    }
    out_0 = sum_0;
    out_1 = sum_1;
}

static void llama_infinitum_mxfp4_dot_row_pair_input_table(
        const std::uint8_t * blocks,
        const std::uint8_t * scales,
        const float * input_pair_table,
        int row,
        int block_count,
        float & out_0,
        float & out_1) {
    constexpr int packed_per_block = 16;
    constexpr int values_per_pair = 256;
    const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
    float sum_0 = 0.0f;
    float sum_1 = 0.0f;
    for (int block = 0; block < block_count; ++block) {
        const int scale_index_0 = row * block_count + block;
        const int scale_index_1 = (row + 1) * block_count + block;
        const int block_base_0 = scale_index_0 * packed_per_block;
        const int block_base_1 = scale_index_1 * packed_per_block;
        const int input_pair_base = block * packed_per_block;
        const float scale_0 = scale_lut[scales[scale_index_0]];
        const float scale_1 = scale_lut[scales[scale_index_1]];
        float block_sum_0 = 0.0f;
        float block_sum_1 = 0.0f;
        for (int p = 0; p < packed_per_block; ++p) {
            const float * input_values = input_pair_table + std::size_t(input_pair_base + p) * values_per_pair;
            block_sum_0 += input_values[blocks[block_base_0 + p]];
            block_sum_1 += input_values[blocks[block_base_1 + p]];
        }
        sum_0 += scale_0 * block_sum_0;
        sum_1 += scale_1 * block_sum_1;
    }
    out_0 = sum_0;
    out_1 = sum_1;
}

static void llama_infinitum_mxfp4_dot_rows8_scalar(
        const std::uint8_t * blocks,
        const std::uint8_t * scales,
        int row,
        int block_count,
        const float * input,
        float out[8]) {
    constexpr int packed_per_block = 16;
    const float * scale_lut = llama_infinitum_mxfp4_scale_lut();
    const float * pair_lut = llama_infinitum_mxfp4_pair_lut();
    float sum[8] = {};
    for (int block = 0; block < block_count; ++block) {
        float scale[8];
        int block_base[8];
        for (int r = 0; r < 8; ++r) {
            scale[r] = scale_lut[scales[(row + r) * block_count + block]];
            block_base[r] = ((row + r) * block_count + block) * packed_per_block;
        }
        const int input_base = block * packed_per_block * 2;
        float block_sum[8] = {};
        for (int p = 0; p < packed_per_block; ++p) {
            const float in_0 = input[input_base + p * 2];
            const float in_1 = input[input_base + p * 2 + 1];
            for (int r = 0; r < 8; ++r) {
                const float * weights = pair_lut + 2 * blocks[block_base[r] + p];
                block_sum[r] += weights[0] * in_0 + weights[1] * in_1;
            }
        }
        for (int r = 0; r < 8; ++r) {
            sum[r] += scale[r] * block_sum[r];
        }
    }
    for (int r = 0; r < 8; ++r) {
        out[r] = sum[r];
    }
}

static void llama_infinitum_mxfp4_dot_rows8_table(
        const std::uint8_t * blocks,
        const std::uint8_t * scales,
        const float * scale_values,
        int row,
        int block_count,
        const float * input,
        float out[8]) {
    constexpr int packed_per_block = 16;
    const float * pair_lut = llama_infinitum_mxfp4_pair_lut();
    const float * scaled_pair_lut = llama_infinitum_mxfp4_scaled_pair_lut();
    float sum[8] = {};
    for (int block = 0; block < block_count; ++block) {
        const int input_base = block * packed_per_block * 2;
        float block_sum[8] = {};
        if (scale_values != nullptr) {
            float scale[8];
            int block_base[8];
            for (int r = 0; r < 8; ++r) {
                const int scale_index = (row + r) * block_count + block;
                scale[r] = scale_values[scale_index];
                block_base[r] = scale_index * packed_per_block;
            }
            for (int p = 0; p < packed_per_block; ++p) {
                const float in_0 = input[input_base + p * 2];
                const float in_1 = input[input_base + p * 2 + 1];
                for (int r = 0; r < 8; ++r) {
                    const float * weights = pair_lut + 2 * blocks[block_base[r] + p];
                    block_sum[r] += weights[0] * in_0 + weights[1] * in_1;
                }
            }
            for (int r = 0; r < 8; ++r) {
                sum[r] += scale[r] * block_sum[r];
            }
        } else {
            int scale[8];
            int block_base[8];
            for (int r = 0; r < 8; ++r) {
                const int scale_index = (row + r) * block_count + block;
                scale[r] = scales[scale_index];
                block_base[r] = scale_index * packed_per_block;
            }
            for (int p = 0; p < packed_per_block; ++p) {
                const float in_0 = input[input_base + p * 2];
                const float in_1 = input[input_base + p * 2 + 1];
                for (int r = 0; r < 8; ++r) {
                    const float * weights = scaled_pair_lut + ((scale[r] * 256 + blocks[block_base[r] + p]) * 2);
                    block_sum[r] += weights[0] * in_0 + weights[1] * in_1;
                }
            }
            for (int r = 0; r < 8; ++r) {
                sum[r] += block_sum[r];
            }
        }
    }
    for (int r = 0; r < 8; ++r) {
        out[r] = sum[r];
    }
}

static float llama_infinitum_mxfp4_dot_row_for_sdot(
        const std::uint8_t * blocks,
        const std::int8_t * sdot_values,
        const std::uint8_t * scales,
        int row,
        int block_count,
        const float * input,
        const llama_infinitum_quantized_input * quantized,
        bool use_sdot) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    if (use_sdot && quantized != nullptr) {
        if (sdot_values != nullptr) {
            return llama_infinitum_mxfp4_dot_row_sdot_prepacked(sdot_values, scales, row, *quantized);
        }
        return llama_infinitum_mxfp4_dot_row_sdot(blocks, scales, row, *quantized);
    }
#endif
    return llama_infinitum_mxfp4_dot_row(blocks, scales, row, block_count, input);
}

static void llama_infinitum_mxfp4_dot_row_pair(
        const std::uint8_t * blocks,
        const std::int8_t * sdot_values,
        const std::uint8_t * scales,
        int row,
        int block_count,
        const float * input,
        const llama_infinitum_quantized_input * quantized,
        bool use_sdot,
        float & out_0,
        float & out_1) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    if (use_sdot && quantized != nullptr) {
        if (sdot_values != nullptr) {
            llama_infinitum_mxfp4_dot_row_pair_sdot_prepacked(sdot_values, scales, row, *quantized, out_0, out_1);
        } else {
            llama_infinitum_mxfp4_dot_row_pair_sdot(blocks, scales, row, *quantized, out_0, out_1);
        }
        return;
    }
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
    llama_infinitum_mxfp4_dot_row_pair_neon(blocks, scales, row, input, out_0, out_1);
    return;
#endif
    llama_infinitum_mxfp4_dot_row_pair_scalar(blocks, scales, row, block_count, input, out_0, out_1);
}

static std::shared_ptr<const llama_infinitum_moe_loaded_slice> llama_infinitum_load_required_slice_ptr(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        int expert_id,
        const char * kind,
        std::string & error) {
    const llama_infinitum_moe_expert_slice * slice = llama_infinitum_moe_find_slice(info, layer_index, expert_id, kind);
    if (slice == nullptr) {
        error = std::string("missing expert slice: ") + kind;
        return nullptr;
    }
    return cache.get_or_load_ptr(info, *slice, error);
}

struct llama_infinitum_loaded_expert_mlp {
    int expert_id = -1;
    std::shared_ptr<const llama_infinitum_moe_loaded_slice> gate_up_blocks;
    std::shared_ptr<const llama_infinitum_moe_loaded_slice> gate_up_blocks_kmajor;
    std::shared_ptr<const llama_infinitum_moe_loaded_slice> gate_up_scales;
    std::shared_ptr<const llama_infinitum_moe_loaded_slice> gate_up_bias;
    std::shared_ptr<const llama_infinitum_moe_loaded_slice> down_blocks;
    std::shared_ptr<const llama_infinitum_moe_loaded_slice> down_blocks_kmajor;
    std::shared_ptr<const llama_infinitum_moe_loaded_slice> down_scales;
    std::shared_ptr<const llama_infinitum_moe_loaded_slice> down_bias;

    std::uint64_t loaded_bytes() const {
        return llama_infinitum_loaded_slice_size(*gate_up_blocks) + llama_infinitum_loaded_slice_size(*gate_up_scales) +
            llama_infinitum_loaded_slice_size(*gate_up_bias) + llama_infinitum_loaded_slice_size(*down_blocks) +
            llama_infinitum_loaded_slice_size(*down_scales) + llama_infinitum_loaded_slice_size(*down_bias);
    }
};

static void llama_infinitum_pack_ggml_mxfp4_rows(
        const std::uint8_t * blocks,
        const std::uint8_t * scales,
        int src_row_stride,
        int src_row_offset,
        int row_count,
        int block_count,
        std::vector<std::uint8_t> & out) {
    constexpr int external_block_bytes = 16;
    constexpr int ggml_block_bytes = 17;
    out.resize(std::size_t(row_count) * std::size_t(block_count) * ggml_block_bytes);
    for (int row = 0; row < row_count; ++row) {
        const int src_row = src_row_offset + row * src_row_stride;
        for (int block = 0; block < block_count; ++block) {
            const int src_index = src_row * block_count + block;
            const std::uint8_t * src_block = blocks + std::size_t(src_index) * external_block_bytes;
            std::uint8_t * dst_block = out.data() + (std::size_t(row) * std::size_t(block_count) + std::size_t(block)) * ggml_block_bytes;
            llama_infinitum_mxfp4_pack_ggml_block(src_block, scales[src_index], dst_block);
        }
    }
}

static std::shared_ptr<llama_infinitum_ggml_packed_expert_mlp> llama_infinitum_pack_ggml_expert(
        int layer_index,
        const llama_infinitum_loaded_expert_mlp & expert,
        int hidden_size,
        int block_count) {
    const std::uint8_t * gate_up_blocks = llama_infinitum_loaded_slice_data(*expert.gate_up_blocks);
    const std::uint8_t * gate_up_scales = llama_infinitum_loaded_slice_data(*expert.gate_up_scales);
    const std::uint8_t * gate_up_bias = llama_infinitum_loaded_slice_data(*expert.gate_up_bias);
    const std::uint8_t * down_blocks = llama_infinitum_loaded_slice_data(*expert.down_blocks);
    const std::uint8_t * down_scales = llama_infinitum_loaded_slice_data(*expert.down_scales);
    const std::uint8_t * down_bias = llama_infinitum_loaded_slice_data(*expert.down_bias);
    if (gate_up_blocks == nullptr || gate_up_scales == nullptr || gate_up_bias == nullptr ||
            down_blocks == nullptr || down_scales == nullptr || down_bias == nullptr) {
        return nullptr;
    }

    auto packed = std::make_shared<llama_infinitum_ggml_packed_expert_mlp>();
    packed->layer_index = layer_index;
    packed->expert_id = expert.expert_id;
    llama_infinitum_pack_ggml_mxfp4_rows(
        gate_up_blocks, gate_up_scales, 2, 0, hidden_size, block_count, packed->gate_blocks);
    llama_infinitum_pack_ggml_mxfp4_rows(
        gate_up_blocks, gate_up_scales, 2, 1, hidden_size, block_count, packed->up_blocks);
    llama_infinitum_pack_ggml_mxfp4_rows(
        down_blocks, down_scales, 1, 0, hidden_size, block_count, packed->down_blocks);
    packed->gate_bias.resize(hidden_size);
    packed->up_bias.resize(hidden_size);
    packed->down_bias.resize(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        packed->gate_bias[i] = llama_infinitum_bf16_at(gate_up_bias, 2 * i);
        packed->up_bias[i] = llama_infinitum_bf16_at(gate_up_bias, 2 * i + 1);
        packed->down_bias[i] = llama_infinitum_bf16_at(down_bias, i);
    }
    return packed;
}

class llama_infinitum_ggml_packed_expert_cache {
public:
    std::shared_ptr<const llama_infinitum_ggml_packed_expert_mlp> get_or_pack(
            int layer_index,
            const llama_infinitum_loaded_expert_mlp & expert,
            int hidden_size,
            int block_count) {
        std::lock_guard<std::mutex> lock(mutex);
        max_bytes = llama_infinitum_moe_gpu_cache_bytes_or_default();
        const std::uint64_t key = key_for(layer_index, expert.expert_id);
        auto found = items.find(key);
        if (found != items.end()) {
            touch(key);
            return found->second;
        }
        auto packed = llama_infinitum_pack_ggml_expert(layer_index, expert, hidden_size, block_count);
        if (packed == nullptr) {
            return nullptr;
        }
        const std::uint64_t packed_bytes = static_cast<std::uint64_t>(packed->bytes());
        if (packed_bytes > max_bytes) {
            return nullptr;
        }
        used_bytes += packed_bytes;
        items[key] = packed;
        order.push_back(key);
        evict_if_needed(key);
        return packed;
    }

private:
    std::mutex mutex;
    std::uint64_t max_bytes = 4096ull * 1024ull * 1024ull;
    std::uint64_t used_bytes = 0;
    std::unordered_map<std::uint64_t, std::shared_ptr<llama_infinitum_ggml_packed_expert_mlp>> items;
    std::list<std::uint64_t> order;

    static std::uint64_t key_for(int layer_index, int expert_id) {
        return (std::uint64_t(std::uint32_t(layer_index)) << 32) | std::uint32_t(expert_id);
    }

    void touch(std::uint64_t key) {
        order.remove(key);
        order.push_back(key);
    }

    void evict_if_needed(std::uint64_t protected_key) {
        while (used_bytes > max_bytes && order.size() > 1) {
            const std::uint64_t victim = order.front();
            order.pop_front();
            if (victim == protected_key) {
                order.push_back(victim);
                continue;
            }
            auto found = items.find(victim);
            if (found == items.end()) {
                continue;
            }
            used_bytes -= static_cast<std::uint64_t>(found->second->bytes());
            items.erase(found);
        }
    }
};

static std::shared_ptr<const llama_infinitum_ggml_packed_expert_mlp> llama_infinitum_load_ggml_pack_expert(
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        int expert_id,
        int hidden_size,
        int block_count,
        std::string & error) {
    if (!llama_infinitum_moe_ggml_pack_enabled()) {
        error = "GGML expert pack is not enabled";
        return nullptr;
    }
    llama_infinitum_ggml_pack_expert_layout layout;
    if (!llama_infinitum_ggml_pack_expert_layout_for(info, layer_index, expert_id, hidden_size, block_count, layout, error)) {
        return nullptr;
    }

    const std::string pack_path = llama_infinitum_moe_ggml_pack_path(info);
    auto mapping = llama_infinitum_moe_ggml_pack_mapping_for(pack_path, error);
    if (mapping == nullptr) {
        if (error.empty()) {
            error = "failed to mmap GGML expert pack";
        }
        return nullptr;
    }
    const std::size_t file_size = mapping->file->size();
    if (layout.gate_up_offset + layout.gate_up_expert_bytes > file_size ||
            layout.down_offset + layout.matrix_expert_bytes > file_size ||
            layout.gate_up_bias_offset + layout.gate_up_bias_expert_bytes > file_size ||
            layout.down_bias_offset + layout.down_bias_expert_bytes > file_size) {
        error = "GGML expert pack selection is out of range";
        return nullptr;
    }

    const auto * base = static_cast<const std::uint8_t *>(mapping->mapping->addr());
    auto packed = std::make_shared<llama_infinitum_ggml_packed_expert_mlp>();
    packed->layer_index = layer_index;
    packed->expert_id = expert_id;
    packed->mapped_owner = mapping;
    packed->mapped_matrix_bytes = layout.matrix_expert_bytes;
    packed->mapped_hidden_size = hidden_size;
    packed->mapped_gate_blocks = base + layout.gate_up_offset;
    packed->mapped_up_blocks = base + layout.gate_up_offset + layout.matrix_expert_bytes;
    packed->mapped_down_blocks = base + layout.down_offset;
    packed->mapped_gate_bias = reinterpret_cast<const float *>(base + layout.gate_up_bias_offset);
    packed->mapped_up_bias = reinterpret_cast<const float *>(base + layout.gate_up_bias_offset + std::size_t(hidden_size) * sizeof(float));
    packed->mapped_down_bias = reinterpret_cast<const float *>(base + layout.down_bias_offset);
    return packed;
}

static bool llama_infinitum_prefetch_ggml_pack_expert_pages(
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        int expert_id,
        int hidden_size,
        int block_count,
        std::string & error) {
    llama_infinitum_ggml_pack_expert_layout layout;
    if (!llama_infinitum_ggml_pack_expert_layout_for(info, layer_index, expert_id, hidden_size, block_count, layout, error)) {
        return false;
    }

    const std::string pack_path = llama_infinitum_moe_ggml_pack_path(info);
    auto mapping = llama_infinitum_moe_ggml_pack_mapping_for(pack_path, error);
    if (mapping == nullptr) {
        if (error.empty()) {
            error = "failed to mmap GGML expert pack";
        }
        return false;
    }
    const std::size_t file_size = mapping->file->size();
    if (layout.gate_up_offset + layout.gate_up_expert_bytes > file_size ||
            layout.down_offset + layout.matrix_expert_bytes > file_size ||
            layout.gate_up_bias_offset + layout.gate_up_bias_expert_bytes > file_size ||
            layout.down_bias_offset + layout.down_bias_expert_bytes > file_size) {
        error = "GGML expert pack prefetch selection is out of range";
        return false;
    }

    const auto * base = static_cast<const std::uint8_t *>(mapping->mapping->addr());
    llama_infinitum_prefetch_memory_range(base + layout.gate_up_offset, layout.gate_up_expert_bytes);
    llama_infinitum_prefetch_memory_range(base + layout.down_offset, layout.matrix_expert_bytes);
    llama_infinitum_prefetch_memory_range(base + layout.gate_up_bias_offset, layout.gate_up_bias_expert_bytes);
    llama_infinitum_prefetch_memory_range(base + layout.down_bias_offset, layout.down_bias_expert_bytes);
    return true;
}

class llama_infinitum_ggml_pack_expert_cache {
public:
    std::shared_ptr<const llama_infinitum_ggml_packed_expert_mlp> get_or_load(
            const llama_infinitum_moe_index_info & info,
            int layer_index,
            int expert_id,
            int hidden_size,
            int block_count,
            std::string & error) {
        std::lock_guard<std::mutex> lock(mutex);
        max_bytes = llama_infinitum_moe_gpu_cache_bytes_or_default();
        const std::uint64_t key = key_for(layer_index, expert_id);
        auto found = items.find(key);
        if (found != items.end()) {
            touch(key);
            return found->second;
        }

        auto packed = llama_infinitum_load_ggml_pack_expert(info, layer_index, expert_id, hidden_size, block_count, error);
        if (packed == nullptr) {
            return nullptr;
        }
        const std::uint64_t packed_bytes = static_cast<std::uint64_t>(packed->bytes());
        if (packed_bytes > max_bytes) {
            error = "single GGML packed expert exceeds GPU expert cache budget";
            return nullptr;
        }
        used_bytes += packed_bytes;
        items[key] = packed;
        order.push_back(key);
        evict_if_needed(key);
        return packed;
    }

private:
    std::mutex mutex;
    std::uint64_t max_bytes = 4096ull * 1024ull * 1024ull;
    std::uint64_t used_bytes = 0;
    std::unordered_map<std::uint64_t, std::shared_ptr<const llama_infinitum_ggml_packed_expert_mlp>> items;
    std::list<std::uint64_t> order;

    static std::uint64_t key_for(int layer_index, int expert_id) {
        return (std::uint64_t(std::uint32_t(layer_index)) << 32) | std::uint32_t(expert_id);
    }

    void touch(std::uint64_t key) {
        order.remove(key);
        order.push_back(key);
    }

    void evict_if_needed(std::uint64_t protected_key) {
        while (used_bytes > max_bytes && order.size() > 1) {
            const std::uint64_t victim = order.front();
            order.pop_front();
            if (victim == protected_key) {
                order.push_back(victim);
                continue;
            }
            auto found = items.find(victim);
            if (found == items.end()) {
                continue;
            }
            used_bytes -= static_cast<std::uint64_t>(found->second->bytes());
            items.erase(found);
        }
    }
};

static llama_infinitum_ggml_pack_expert_cache & llama_infinitum_ggml_pack_expert_cache_get() {
    static llama_infinitum_ggml_pack_expert_cache cache;
    return cache;
}

static int llama_infinitum_bf16_row_count(const llama_infinitum_moe_loaded_slice & slice) {
    return static_cast<int>(llama_infinitum_loaded_slice_size(slice) / 2);
}

class llama_infinitum_packed_expert_cache {
public:
    std::shared_ptr<const llama_infinitum_packed_expert_mlp> get_or_pack(
            int layer_index,
            int expert_id,
            const llama_infinitum_loaded_expert_mlp & expert) {
        std::lock_guard<std::mutex> lock(mutex);
        max_bytes = llama_infinitum_mxfp4_packed_cache_bytes_from_env();
        const std::uint64_t key = key_for(layer_index, expert_id);
        auto found = items.find(key);
        if (found != items.end()) {
            touch(key);
            return found->second;
        }
        const std::uint64_t admit_hits = llama_infinitum_mxfp4_packed_admit_hits_from_env();
        const std::uint64_t seen = ++seen_counts[key];
        if (seen < admit_hits) {
            return nullptr;
        }

        auto packed = std::make_shared<llama_infinitum_packed_expert_mlp>();
        packed->layer_index = layer_index;
        packed->expert_id = expert_id;
        const int gate_up_rows = llama_infinitum_bf16_row_count(*expert.gate_up_bias);
        const int down_rows = llama_infinitum_bf16_row_count(*expert.down_bias);
        const int hidden_size = down_rows;
        const int input_block_count = llama_infinitum_mxfp4_block_count_for_hidden(hidden_size);
        if (input_block_count <= 0 || gate_up_rows != 2 * hidden_size) {
            return nullptr;
        }
        llama_infinitum_mxfp4_prepack_kmajor_sdot_values(
            llama_infinitum_loaded_slice_data(*expert.gate_up_blocks), gate_up_rows, input_block_count, packed->gate_up_kmajor_values);
        llama_infinitum_mxfp4_prepack_kmajor_sdot_values(
            llama_infinitum_loaded_slice_data(*expert.down_blocks), down_rows, input_block_count, packed->down_kmajor_values);

        const std::uint64_t packed_bytes = static_cast<std::uint64_t>(packed->bytes());
        if (packed_bytes > max_bytes) {
            return nullptr;
        }
        used_bytes += packed_bytes;
        items[key] = packed;
        touch(key);
        evict_if_needed(key);
        return packed;
    }

private:
    std::mutex mutex;
    std::uint64_t max_bytes = 512ull * 1024ull * 1024ull;
    std::uint64_t used_bytes = 0;
    std::unordered_map<std::uint64_t, std::shared_ptr<llama_infinitum_packed_expert_mlp>> items;
    std::unordered_map<std::uint64_t, std::uint64_t> seen_counts;
    std::vector<std::uint64_t> order;

    static std::uint64_t key_for(int layer_index, int expert_id) {
        return (std::uint64_t(std::uint32_t(layer_index)) << 32) | std::uint32_t(expert_id);
    }

    void touch(std::uint64_t key) {
        order.erase(std::remove(order.begin(), order.end(), key), order.end());
        order.push_back(key);
    }

    void evict_if_needed(std::uint64_t protected_key) {
        while (used_bytes > max_bytes && order.size() > 1) {
            const std::uint64_t victim = order.front();
            order.erase(order.begin());
            if (victim == protected_key) {
                order.push_back(victim);
                continue;
            }
            auto found = items.find(victim);
            if (found == items.end()) {
                continue;
            }
            used_bytes -= static_cast<std::uint64_t>(found->second->bytes());
            items.erase(found);
        }
    }
};

static bool llama_infinitum_load_expert_mlp(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        int expert_id,
        llama_infinitum_loaded_expert_mlp & out,
        std::string & error) {
    out.expert_id = expert_id;
    out.gate_up_blocks = llama_infinitum_load_required_slice_ptr(info, cache, layer_index, expert_id, "gate_up_proj_blocks", error);
    if (llama_infinitum_mxfp4_sidecar_kmajor_enabled()) {
        out.gate_up_blocks_kmajor = llama_infinitum_load_required_slice_ptr(info, cache, layer_index, expert_id, "gate_up_proj_blocks_kmajor", error);
        if (out.gate_up_blocks_kmajor == nullptr) {
            error.clear();
        }
    }
    out.gate_up_scales = llama_infinitum_load_required_slice_ptr(info, cache, layer_index, expert_id, "gate_up_proj_scales", error);
    out.gate_up_bias = llama_infinitum_load_required_slice_ptr(info, cache, layer_index, expert_id, "gate_up_proj_bias", error);
    out.down_blocks = llama_infinitum_load_required_slice_ptr(info, cache, layer_index, expert_id, "down_proj_blocks", error);
    if (llama_infinitum_mxfp4_sidecar_kmajor_enabled()) {
        out.down_blocks_kmajor = llama_infinitum_load_required_slice_ptr(info, cache, layer_index, expert_id, "down_proj_blocks_kmajor", error);
        if (out.down_blocks_kmajor == nullptr) {
            error.clear();
        }
    }
    out.down_scales = llama_infinitum_load_required_slice_ptr(info, cache, layer_index, expert_id, "down_proj_scales", error);
    out.down_bias = llama_infinitum_load_required_slice_ptr(info, cache, layer_index, expert_id, "down_proj_bias", error);
    return out.gate_up_blocks != nullptr && out.gate_up_scales != nullptr && out.gate_up_bias != nullptr &&
        out.down_blocks != nullptr && out.down_scales != nullptr && out.down_bias != nullptr;
}

bool llama_infinitum_moe_prefetch_selected_experts(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        const std::vector<int> & expert_ids,
        std::string & error) {
    error.clear();
    for (const int expert_id : expert_ids) {
        llama_infinitum_loaded_expert_mlp expert;
        if (!llama_infinitum_load_expert_mlp(info, cache, layer_index, expert_id, expert, error)) {
            return false;
        }
    }
    return true;
}

static bool llama_infinitum_moe_prefetch_selected_ggml_pack_pages(
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        const std::vector<int> & expert_ids,
        std::string & error) {
    error.clear();
    const int hidden_size = info.hidden_size;
    const int block_count = llama_infinitum_mxfp4_block_count_for_hidden(hidden_size);
    if (hidden_size <= 0 || block_count <= 0) {
        error = "invalid hidden shape for GGML pack page prefetch";
        return false;
    }
    std::vector<int> unique_ids;
    unique_ids.reserve(expert_ids.size());
    for (const int expert_id : expert_ids) {
        if (std::find(unique_ids.begin(), unique_ids.end(), expert_id) != unique_ids.end()) {
            continue;
        }
        unique_ids.push_back(expert_id);
        if (!llama_infinitum_prefetch_ggml_pack_expert_pages(
                info, layer_index, expert_id, hidden_size, block_count, error)) {
            return false;
        }
    }
    if (llama_infinitum_pack_report_enabled() && !unique_ids.empty()) {
        std::fprintf(stderr, "infinitum_pack_prefetch: layer=%d experts=%zu mode=pages\n",
                layer_index, unique_ids.size());
    }
    return true;
}

class llama_infinitum_moe_prefetch_worker {
public:
    ~llama_infinitum_moe_prefetch_worker() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
            jobs.clear();
        }
        cv.notify_one();
        if (worker.joinable()) {
            worker.join();
        }
    }

    void enqueue(
            const llama_infinitum_moe_index_info & info,
            llama_infinitum_moe_slice_cache & cache,
            int layer_index,
            const std::vector<int> & expert_ids) {
        if (expert_ids.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        if (!worker.joinable()) {
            worker = std::thread([this]() { loop(); });
        }
        const std::uint64_t signature = job_signature(layer_index, expert_ids);
        if (seen_jobs.find(signature) != seen_jobs.end()) {
            return;
        }
        if (seen_jobs.size() > 4096) {
            seen_jobs.clear();
        }
        seen_jobs.insert(signature);
        coalesce_layer_jobs(layer_index);
        if (jobs.size() >= 16) {
            jobs.pop_front();
        }
        jobs.push_back(job{&info, &cache, layer_index, expert_ids});
        cv.notify_one();
    }

private:
    struct job {
        const llama_infinitum_moe_index_info * info = nullptr;
        llama_infinitum_moe_slice_cache * cache = nullptr;
        int layer_index = -1;
        std::vector<int> expert_ids;
    };

    std::mutex mutex;
    std::condition_variable cv;
    std::deque<job> jobs;
    std::unordered_set<std::uint64_t> seen_jobs;
    std::thread worker;
    bool stop = false;

    static std::uint64_t job_signature(int layer_index, const std::vector<int> & expert_ids) {
        std::uint64_t hash = 1469598103934665603ull;
        auto mix = [&](std::uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ull;
        };
        mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(layer_index)));
        for (const int expert_id : expert_ids) {
            mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(expert_id)));
        }
        return hash;
    }

    void coalesce_layer_jobs(int layer_index) {
        for (auto it = jobs.begin(); it != jobs.end();) {
            const job & pending = *it;
            if (pending.layer_index == layer_index) {
                it = jobs.erase(it);
            } else {
                ++it;
            }
        }
    }

    void loop() {
        while (true) {
            job current;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&]() { return stop || !jobs.empty(); });
                if (stop && jobs.empty()) {
                    return;
                }
                current = std::move(jobs.front());
                jobs.pop_front();
            }
            if (current.info != nullptr && current.cache != nullptr) {
                std::string error;
                if (llama_infinitum_moe_ggml_pack_enabled() && llama_infinitum_moe_ggml_pack_prefetch_enabled()) {
                    llama_infinitum_moe_prefetch_selected_ggml_pack_pages(
                        *current.info, current.layer_index, current.expert_ids, error);
                } else if (llama_infinitum_moe_ggml_pack_enabled() && llama_infinitum_moe_ggml_pack_slots_enabled() &&
                        llama_infinitum_moe_backend_kind_from_env() == llama_infinitum_moe_backend_kind::vulkan) {
                    llama_infinitum_moe_prefetch_selected_gpu_experts(
                        *current.info, current.layer_index, current.expert_ids, error);
                } else {
                    llama_infinitum_moe_prefetch_selected_experts(
                        *current.info, *current.cache, current.layer_index, current.expert_ids, error);
                }
            }
        }
    }
};

void llama_infinitum_moe_prefetch_selected_experts_async(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        const std::vector<int> & expert_ids) {
    if (!llama_infinitum_moe_prefetch_enabled()) {
        return;
    }
    static llama_infinitum_moe_prefetch_worker worker;
    worker.enqueue(info, cache, layer_index, expert_ids);
}

static bool llama_infinitum_compute_loaded_expert_mlp(
        const llama_infinitum_loaded_expert_mlp & expert,
        const float * hidden_data,
        int hidden_size,
        float * output,
        float * act,
        const llama_infinitum_quantized_input * shared_quantized_hidden = nullptr,
        const float * hidden_pair_table = nullptr,
        llama_infinitum_moe_compute_profile * profile = nullptr,
        const llama_infinitum_packed_expert_mlp * packed_expert = nullptr,
        int row_threads = 1,
        const llama_infinitum_mxfp4_runtime_flags * runtime_flags = nullptr) {
    if (hidden_data == nullptr || hidden_size <= 0 || output == nullptr || act == nullptr) {
        return false;
    }
    const int gate_up_rows = llama_infinitum_bf16_row_count(*expert.gate_up_bias);
    const int down_rows = llama_infinitum_bf16_row_count(*expert.down_bias);
    if (gate_up_rows != 2 * hidden_size || down_rows != hidden_size) {
        return false;
    }
    const int input_block_count = llama_infinitum_mxfp4_block_count_for_hidden(hidden_size);
    if (input_block_count <= 0) {
        return false;
    }
    const std::uint8_t * gate_up_blocks = llama_infinitum_loaded_slice_data(*expert.gate_up_blocks);
    const std::int8_t * gate_up_sdot_values = expert.gate_up_blocks->sdot_values.empty() ? nullptr : expert.gate_up_blocks->sdot_values.data();
    const std::int8_t * gate_up_kmajor_values = expert.gate_up_blocks_kmajor != nullptr ?
        reinterpret_cast<const std::int8_t *>(llama_infinitum_loaded_slice_data(*expert.gate_up_blocks_kmajor)) :
        (packed_expert != nullptr && !packed_expert->gate_up_kmajor_values.empty() ?
        packed_expert->gate_up_kmajor_values.data() :
        (expert.gate_up_blocks->sdot_kmajor_values.empty() ? nullptr : expert.gate_up_blocks->sdot_kmajor_values.data()));
    const std::uint8_t * gate_up_scales = llama_infinitum_loaded_slice_data(*expert.gate_up_scales);
    const std::uint8_t * gate_up_bias = llama_infinitum_loaded_slice_data(*expert.gate_up_bias);
    const std::uint8_t * down_blocks = llama_infinitum_loaded_slice_data(*expert.down_blocks);
    const std::int8_t * down_sdot_values = expert.down_blocks->sdot_values.empty() ? nullptr : expert.down_blocks->sdot_values.data();
    const std::int8_t * down_kmajor_values = expert.down_blocks_kmajor != nullptr ?
        reinterpret_cast<const std::int8_t *>(llama_infinitum_loaded_slice_data(*expert.down_blocks_kmajor)) :
        (packed_expert != nullptr && !packed_expert->down_kmajor_values.empty() ?
        packed_expert->down_kmajor_values.data() :
        (expert.down_blocks->sdot_kmajor_values.empty() ? nullptr : expert.down_blocks->sdot_kmajor_values.data()));
    const std::uint8_t * down_scales = llama_infinitum_loaded_slice_data(*expert.down_scales);
    const float * down_table_scales = expert.down_scales->table_scale_values.empty() ?
        nullptr : expert.down_scales->table_scale_values.data();
    const std::uint8_t * down_bias = llama_infinitum_loaded_slice_data(*expert.down_bias);
    const llama_infinitum_mxfp4_runtime_flags local_runtime_flags =
        runtime_flags != nullptr ? *runtime_flags : llama_infinitum_mxfp4_runtime_flags_from_env();
    const bool use_sdot_gate_up = local_runtime_flags.use_sdot_gate_up;
    const bool use_sdot_down = local_runtime_flags.use_sdot_down;
    const bool use_cpu_table_gate_up = local_runtime_flags.use_input_pair_table && !use_sdot_gate_up && hidden_pair_table != nullptr;
    const bool use_cpu_table_down = local_runtime_flags.use_scaled_pair_table && !use_sdot_down;
    llama_infinitum_quantized_input quantized_hidden;
    llama_infinitum_quantized_input quantized_act;
    const llama_infinitum_quantized_input * hidden_for_sdot = shared_quantized_hidden;
    if (use_sdot_gate_up) {
        if (hidden_for_sdot == nullptr) {
            llama_infinitum_quantize_input_32(hidden_data, hidden_size, quantized_hidden);
            hidden_for_sdot = &quantized_hidden;
        }
    }

    if (profile != nullptr) {
        std::vector<float> gate_values(hidden_size);
        std::vector<float> up_values(hidden_size);
        auto start = std::chrono::steady_clock::now();
        int i = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
        if (!use_sdot_gate_up) {
            for (; i + 3 < hidden_size; i += 4) {
                float dots[8] = {};
                llama_infinitum_mxfp4_dot_rows8_neon(gate_up_blocks, gate_up_scales, 2 * i, hidden_data, dots);
                for (int j = 0; j < 4; ++j) {
                    const int index = i + j;
                    gate_values[index] = dots[2 * j] + llama_infinitum_bf16_at(gate_up_bias, 2 * index);
                    up_values[index] = dots[2 * j + 1] + llama_infinitum_bf16_at(gate_up_bias, 2 * index + 1);
                }
            }
        }
#endif
        for (; i < hidden_size; ++i) {
            float gate_dot = 0.0f;
            float up_dot = 0.0f;
            if (use_cpu_table_gate_up) {
                llama_infinitum_mxfp4_dot_row_pair_input_table(
                    gate_up_blocks, gate_up_scales, hidden_pair_table, 2 * i, input_block_count,
                    gate_dot, up_dot);
            } else {
                llama_infinitum_mxfp4_dot_row_pair(
                    gate_up_blocks, gate_up_sdot_values, gate_up_scales, 2 * i, input_block_count,
                    hidden_data, hidden_for_sdot, use_sdot_gate_up, gate_dot, up_dot);
            }
            gate_values[i] = gate_dot + llama_infinitum_bf16_at(gate_up_bias, 2 * i);
            up_values[i] = up_dot + llama_infinitum_bf16_at(gate_up_bias, 2 * i + 1);
        }
        auto after_gate_up = std::chrono::steady_clock::now();
        for (int i = 0; i < hidden_size; ++i) {
            float gate = std::min(gate_values[i], 7.0f);
            const float up = std::max(-7.0f, std::min(up_values[i], 7.0f));
            const float sigmoid = 1.0f / (1.0f + std::exp(-1.702f * gate));
            act[i] = gate * sigmoid * (up + 1.0f);
        }
        if (use_sdot_down) {
            llama_infinitum_quantize_input_32(act, hidden_size, quantized_act);
        }
        auto after_activation = std::chrono::steady_clock::now();
        bool finite = true;
        int row = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
        if (!use_sdot_down) {
            for (; row + 7 < hidden_size; row += 8) {
                float values[8] = {};
                llama_infinitum_mxfp4_dot_rows8_neon(down_blocks, down_scales, row, act, values);
                for (int j = 0; j < 8; ++j) {
                    const float value = values[j] + llama_infinitum_bf16_at(down_bias, row + j);
                    finite = finite && std::isfinite(value);
                    output[row + j] = value;
                }
            }
        }
#endif
#if !(defined(__ARM_NEON) && defined(__aarch64__))
        if (!use_sdot_down) {
            for (; row + 7 < hidden_size; row += 8) {
                float values[8] = {};
                if (use_cpu_table_down) {
                    llama_infinitum_mxfp4_dot_rows8_table(down_blocks, down_scales, down_table_scales, row, input_block_count, act, values);
                } else {
                    llama_infinitum_mxfp4_dot_rows8_scalar(down_blocks, down_scales, row, input_block_count, act, values);
                }
                for (int j = 0; j < 8; ++j) {
                    const float value = values[j] + llama_infinitum_bf16_at(down_bias, row + j);
                    finite = finite && std::isfinite(value);
                    output[row + j] = value;
                }
            }
        }
#endif
        for (; row < hidden_size; ++row) {
            const float value =
                llama_infinitum_mxfp4_dot_row_for_sdot(down_blocks, down_sdot_values, down_scales, row, input_block_count,
                act, &quantized_act, use_sdot_down)
                + llama_infinitum_bf16_at(down_bias, row);
            finite = finite && std::isfinite(value);
            output[row] = value;
        }
        auto after_down = std::chrono::steady_clock::now();
        profile->gate_up_ms += llama_infinitum_elapsed_ms(start, after_gate_up);
        profile->activation_ms += llama_infinitum_elapsed_ms(after_gate_up, after_activation);
        profile->down_ms += llama_infinitum_elapsed_ms(after_activation, after_down);
        return finite;
    }

    if (profile == nullptr && row_threads > 1 && !use_sdot_gate_up && !use_sdot_down) {
#if defined(__ARM_NEON) && defined(__aarch64__)
        static llama_infinitum_worker_pool row_worker_pool;
        constexpr int groups_per_task = 8;
        const int gate_group_count = hidden_size / 4;
        const int down_group_count = hidden_size / 8;
        const int gate_tasks = (gate_group_count + groups_per_task - 1) / groups_per_task;
        row_worker_pool.run(row_threads, gate_tasks, [&](int task_index) {
            const int group_begin = task_index * groups_per_task;
            const int group_end = std::min(gate_group_count, group_begin + groups_per_task);
            for (int group = group_begin; group < group_end; ++group) {
                const int activation_index = group * 4;
                float dots[8] = {};
                llama_infinitum_mxfp4_dot_rows8_neon(gate_up_blocks, gate_up_scales, 2 * activation_index, hidden_data, dots);
                for (int j = 0; j < 4; ++j) {
                    const int index = activation_index + j;
                    float gate = dots[2 * j] + llama_infinitum_bf16_at(gate_up_bias, 2 * index);
                    float up = dots[2 * j + 1] + llama_infinitum_bf16_at(gate_up_bias, 2 * index + 1);
                    gate = std::min(gate, 7.0f);
                    up = std::max(-7.0f, std::min(up, 7.0f));
                    const float sigmoid = 1.0f / (1.0f + std::exp(-1.702f * gate));
                    act[index] = gate * sigmoid * (up + 1.0f);
                }
            }
        });

        bool finite = true;
        const int down_tasks = (down_group_count + groups_per_task - 1) / groups_per_task;
        row_worker_pool.run(row_threads, down_tasks, [&](int task_index) {
            const int group_begin = task_index * groups_per_task;
            const int group_end = std::min(down_group_count, group_begin + groups_per_task);
            for (int group = group_begin; group < group_end; ++group) {
                const int row = group * 8;
                float values[8] = {};
                llama_infinitum_mxfp4_dot_rows8_neon(down_blocks, down_scales, row, act, values);
                for (int i = 0; i < 8; ++i) {
                    output[row + i] = values[i] + llama_infinitum_bf16_at(down_bias, row + i);
                }
            }
        });
        for (int row = 0; row < hidden_size; ++row) {
            finite = finite && std::isfinite(output[row]);
        }
        return finite;
#endif
    }

    int activation_index = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
    if (use_sdot_gate_up && hidden_for_sdot != nullptr && (gate_up_sdot_values == nullptr || gate_up_kmajor_values != nullptr)) {
        for (; activation_index + 3 < hidden_size; activation_index += 4) {
            float dots[8] = {};
            if (gate_up_kmajor_values != nullptr) {
                llama_infinitum_mxfp4_dot_rows8_sdot_kmajor(gate_up_kmajor_values, gate_up_scales, 2 * activation_index, *hidden_for_sdot, dots);
            } else {
                llama_infinitum_mxfp4_dot_rows8_sdot(gate_up_blocks, gate_up_scales, 2 * activation_index, *hidden_for_sdot, dots);
            }
            for (int j = 0; j < 4; ++j) {
                const int index = activation_index + j;
                float gate = dots[2 * j] + llama_infinitum_bf16_at(gate_up_bias, 2 * index);
                float up = dots[2 * j + 1] + llama_infinitum_bf16_at(gate_up_bias, 2 * index + 1);
                gate = std::min(gate, 7.0f);
                up = std::max(-7.0f, std::min(up, 7.0f));
                const float sigmoid = 1.0f / (1.0f + std::exp(-1.702f * gate));
                act[index] = gate * sigmoid * (up + 1.0f);
            }
        }
    }
    if (!use_sdot_gate_up) {
        for (; activation_index + 3 < hidden_size; activation_index += 4) {
            float dots[8] = {};
            llama_infinitum_mxfp4_dot_rows8_neon(gate_up_blocks, gate_up_scales, 2 * activation_index, hidden_data, dots);
            for (int j = 0; j < 4; ++j) {
                const int index = activation_index + j;
                float gate = dots[2 * j] + llama_infinitum_bf16_at(gate_up_bias, 2 * index);
                float up = dots[2 * j + 1] + llama_infinitum_bf16_at(gate_up_bias, 2 * index + 1);
                gate = std::min(gate, 7.0f);
                up = std::max(-7.0f, std::min(up, 7.0f));
                const float sigmoid = 1.0f / (1.0f + std::exp(-1.702f * gate));
                act[index] = gate * sigmoid * (up + 1.0f);
            }
        }
    }
#endif
    for (; activation_index < hidden_size; ++activation_index) {
        float gate_dot = 0.0f;
        float up_dot = 0.0f;
        if (use_cpu_table_gate_up) {
            llama_infinitum_mxfp4_dot_row_pair_input_table(
                gate_up_blocks, gate_up_scales, hidden_pair_table, 2 * activation_index, input_block_count,
                gate_dot, up_dot);
        } else {
            llama_infinitum_mxfp4_dot_row_pair(
                gate_up_blocks, gate_up_sdot_values, gate_up_scales, 2 * activation_index, input_block_count,
                hidden_data, hidden_for_sdot, use_sdot_gate_up, gate_dot, up_dot);
        }
        float gate = gate_dot + llama_infinitum_bf16_at(gate_up_bias, 2 * activation_index);
        float up = up_dot + llama_infinitum_bf16_at(gate_up_bias, 2 * activation_index + 1);
        gate = std::min(gate, 7.0f);
        up = std::max(-7.0f, std::min(up, 7.0f));
        const float sigmoid = 1.0f / (1.0f + std::exp(-1.702f * gate));
        act[activation_index] = gate * sigmoid * (up + 1.0f);
    }
    if (use_sdot_down) {
        llama_infinitum_quantize_input_32(act, hidden_size, quantized_act);
    }
    bool finite = true;
    int row = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
    if (use_sdot_down && (down_sdot_values == nullptr || down_kmajor_values != nullptr)) {
        for (; row + 7 < hidden_size; row += 8) {
            float values[8] = {};
            if (down_kmajor_values != nullptr) {
                llama_infinitum_mxfp4_dot_rows8_sdot_kmajor(down_kmajor_values, down_scales, row, quantized_act, values);
            } else {
                llama_infinitum_mxfp4_dot_rows8_sdot(down_blocks, down_scales, row, quantized_act, values);
            }
            for (int i = 0; i < 8; ++i) {
                const float value = values[i] + llama_infinitum_bf16_at(down_bias, row + i);
                finite = finite && std::isfinite(value);
                output[row + i] = value;
            }
        }
    }
    if (!use_sdot_down) {
        for (; row + 7 < hidden_size; row += 8) {
            float values[8] = {};
            llama_infinitum_mxfp4_dot_rows8_neon(down_blocks, down_scales, row, act, values);
            for (int i = 0; i < 8; ++i) {
                const float value = values[i] + llama_infinitum_bf16_at(down_bias, row + i);
                finite = finite && std::isfinite(value);
                output[row + i] = value;
            }
        }
    }
#endif
#if !(defined(__ARM_NEON) && defined(__aarch64__))
    if (!use_sdot_down) {
        for (; row + 7 < hidden_size; row += 8) {
            float values[8] = {};
            if (use_cpu_table_down) {
                llama_infinitum_mxfp4_dot_rows8_table(down_blocks, down_scales, down_table_scales, row, input_block_count, act, values);
            } else {
                llama_infinitum_mxfp4_dot_rows8_scalar(down_blocks, down_scales, row, input_block_count, act, values);
            }
            for (int i = 0; i < 8; ++i) {
                const float value = values[i] + llama_infinitum_bf16_at(down_bias, row + i);
                finite = finite && std::isfinite(value);
                output[row + i] = value;
            }
        }
    }
#endif
    for (; row < hidden_size; ++row) {
        const float value =
            llama_infinitum_mxfp4_dot_row_for_sdot(down_blocks, down_sdot_values, down_scales, row, input_block_count,
            act, &quantized_act, use_sdot_down)
            + llama_infinitum_bf16_at(down_bias, row);
        finite = finite && std::isfinite(value);
        output[row] = value;
    }
    return finite;
}

llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_expert_mlp_cpu(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        int expert_id,
        const std::vector<float> & hidden) {
    llama_infinitum_moe_expert_mlp_result result;
    result.layer_index = layer_index;
    result.expert_id = expert_id;
    result.backend = "cpu";
    const int expected_hidden_size = info.hidden_size > 0 ? info.hidden_size : static_cast<int>(hidden.size());
    if (hidden.empty() || static_cast<int>(hidden.size()) != expected_hidden_size) {
        result.error = "hidden size does not match expert index shape";
        return result;
    }

    llama_infinitum_loaded_expert_mlp expert;
    std::string error;
    const llama_infinitum_moe_cache_stats before_stats = cache.stats();
    const auto load_start = std::chrono::steady_clock::now();
    if (!llama_infinitum_load_expert_mlp(info, cache, layer_index, expert_id, expert, error)) {
        result.error = error;
        return result;
    }
    const auto load_end = std::chrono::steady_clock::now();
    const llama_infinitum_moe_cache_stats after_load_stats = cache.stats();

    result.loaded_bytes = expert.loaded_bytes();
    result.cache_hits_delta = after_load_stats.cache_hits - before_stats.cache_hits;
    result.cache_misses_delta = after_load_stats.cache_misses - before_stats.cache_misses;
    result.cache_evictions_delta = after_load_stats.evictions - before_stats.evictions;
    result.loaded_bytes_delta = after_load_stats.loaded_bytes - before_stats.loaded_bytes;
    result.touched_bytes_delta = after_load_stats.touched_bytes - before_stats.touched_bytes;
    result.mapped_bytes_delta = after_load_stats.mapped_bytes - before_stats.mapped_bytes;
    result.copied_bytes_delta = after_load_stats.copied_bytes - before_stats.copied_bytes;
    result.prepacked_bytes_delta = after_load_stats.prepacked_bytes - before_stats.prepacked_bytes;
    result.resident_bytes = after_load_stats.resident_bytes;
    result.process_resident_bytes = after_load_stats.process_resident_bytes;
    result.load_ms = llama_infinitum_elapsed_ms(load_start, load_end);

    llama_infinitum_moe_compute_profile profile;
    const int hidden_size = static_cast<int>(hidden.size());
    result.output.assign(hidden_size, 0.0f);
    std::vector<float> act(hidden_size);
    const auto compute_start = std::chrono::steady_clock::now();
    const bool finite_output = llama_infinitum_compute_loaded_expert_mlp(
        expert, hidden.data(), hidden_size, result.output.data(), act.data(), nullptr, nullptr, &profile);
    const auto compute_end = std::chrono::steady_clock::now();
    result.gate_up_ms = profile.gate_up_ms;
    result.activation_ms = profile.activation_ms;
    result.down_ms = profile.down_ms;
    result.total_compute_ms = llama_infinitum_elapsed_ms(compute_start, compute_end);
    double norm = 0.0;
    bool finite = true;
    for (const float value : result.output) {
        finite = finite && std::isfinite(value);
        norm += double(value) * double(value);
    }
    result.output_norm = float(std::sqrt(norm));
    result.ok = finite && finite_output;
    if (!finite) {
        result.error = "non-finite selected expert output";
    }
    return result;
}

class llama_infinitum_ggml_gpu_backend {
public:
    bool get(ggml_backend_t & out_backend, ggml_backend_dev_t & out_device, std::string & error) {
        std::lock_guard<std::mutex> lock(mutex);
        if (backend != nullptr && device != nullptr) {
            out_backend = backend;
            out_device = device;
            return true;
        }
        if (init_attempted) {
            error = init_error;
            return false;
        }
        init_attempted = true;
        device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (device == nullptr) {
            device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        }
        if (device == nullptr) {
            init_error = "no GPU/IGPU ggml backend device is registered";
            error = init_error;
            return false;
        }
        backend = ggml_backend_dev_init(device, nullptr);
        if (backend == nullptr) {
            init_error = "failed to initialize ggml GPU backend";
            error = init_error;
            device = nullptr;
            return false;
        }
        out_backend = backend;
        out_device = device;
        return true;
    }

    std::mutex & compute_mutex() {
        return compute_lock;
    }

private:
    std::mutex mutex;
    std::mutex compute_lock;
    bool init_attempted = false;
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    std::string init_error;
};

static llama_infinitum_ggml_gpu_backend & llama_infinitum_ggml_gpu_backend_get() {
    static llama_infinitum_ggml_gpu_backend state;
    return state;
}

struct llama_infinitum_ggml_gpu_expert_graph {
    int layer_index = -1;
    int expert_id = -1;
    std::size_t bytes = 0;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * hidden = nullptr;
    ggml_tensor * out = nullptr;

    llama_infinitum_ggml_gpu_expert_graph() = default;
    llama_infinitum_ggml_gpu_expert_graph(const llama_infinitum_ggml_gpu_expert_graph &) = delete;
    llama_infinitum_ggml_gpu_expert_graph & operator=(const llama_infinitum_ggml_gpu_expert_graph &) = delete;

    ~llama_infinitum_ggml_gpu_expert_graph() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

static std::shared_ptr<llama_infinitum_ggml_gpu_expert_graph> llama_infinitum_ggml_gpu_create_expert_graph(
        const llama_infinitum_ggml_packed_expert_mlp & expert,
        int hidden_size,
        ggml_backend_t backend,
        ggml_backend_dev_t device,
        std::string & error) {
    constexpr std::size_t ctx_mem = 4ull * 1024ull * 1024ull;
    ggml_init_params params = {};
    params.mem_size = ctx_mem;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        error = "failed to initialize ggml context for Vulkan expert";
        return nullptr;
    }

    ggml_tensor * gate_w = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, hidden_size, hidden_size);
    ggml_tensor * up_w = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, hidden_size, hidden_size);
    ggml_tensor * down_w = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, hidden_size, hidden_size);
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 1);
    ggml_tensor * gate_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 1);
    ggml_tensor * up_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 1);
    ggml_tensor * down_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 1);
    ggml_tensor * gate = ggml_add(ctx, ggml_mul_mat(ctx, gate_w, hidden), gate_b);
    ggml_tensor * up = ggml_add(ctx, ggml_mul_mat(ctx, up_w, hidden), up_b);
    ggml_tensor * act = ggml_swiglu_oai(ctx, gate, up, 1.702f, 7.0f);
    ggml_tensor * out = ggml_add(ctx, ggml_mul_mat(ctx, down_w, act), down_b);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    if (!ggml_backend_dev_supports_op(device, out)) {
        error = "GPU backend does not support the selected MXFP4 expert graph";
        ggml_free(ctx);
        return nullptr;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        error = "failed to allocate GPU tensors for selected expert";
        ggml_free(ctx);
        return nullptr;
    }

    ggml_backend_tensor_set(gate_w, expert.gate_blocks.data(), 0, expert.gate_blocks.size());
    ggml_backend_tensor_set(up_w, expert.up_blocks.data(), 0, expert.up_blocks.size());
    ggml_backend_tensor_set(down_w, expert.down_blocks.data(), 0, expert.down_blocks.size());
    ggml_backend_tensor_set(gate_b, expert.gate_bias.data(), 0, expert.gate_bias.size() * sizeof(float));
    ggml_backend_tensor_set(up_b, expert.up_bias.data(), 0, expert.up_bias.size() * sizeof(float));
    ggml_backend_tensor_set(down_b, expert.down_bias.data(), 0, expert.down_bias.size() * sizeof(float));

    auto graph_state = std::make_shared<llama_infinitum_ggml_gpu_expert_graph>();
    graph_state->layer_index = expert.layer_index;
    graph_state->expert_id = expert.expert_id;
    graph_state->bytes = ggml_backend_buffer_get_size(buffer);
    graph_state->ctx = ctx;
    graph_state->buffer = buffer;
    graph_state->graph = graph;
    graph_state->hidden = hidden;
    graph_state->out = out;
    return graph_state;
}

class llama_infinitum_ggml_gpu_expert_cache {
public:
    std::shared_ptr<llama_infinitum_ggml_gpu_expert_graph> get_or_create(
            const llama_infinitum_ggml_packed_expert_mlp & expert,
            int hidden_size,
            ggml_backend_t backend,
            ggml_backend_dev_t device,
            std::string & error) {
        std::lock_guard<std::mutex> lock(mutex);
        max_bytes = llama_infinitum_moe_gpu_cache_bytes_or_default();
        const std::uint64_t key = key_for(expert.layer_index, expert.expert_id);
        auto found = items.find(key);
        if (found != items.end()) {
            touch(key);
            return found->second;
        }

        auto graph = llama_infinitum_ggml_gpu_create_expert_graph(expert, hidden_size, backend, device, error);
        if (graph == nullptr) {
            return nullptr;
        }
        if (graph->bytes > max_bytes) {
            error = "selected expert GPU graph exceeds GPU cache budget";
            return nullptr;
        }

        used_bytes += static_cast<std::uint64_t>(graph->bytes);
        items[key] = graph;
        order.push_back(key);
        evict_if_needed(key);
        return graph;
    }

private:
    std::mutex mutex;
    std::uint64_t max_bytes = 4096ull * 1024ull * 1024ull;
    std::uint64_t used_bytes = 0;
    std::unordered_map<std::uint64_t, std::shared_ptr<llama_infinitum_ggml_gpu_expert_graph>> items;
    std::list<std::uint64_t> order;

    static std::uint64_t key_for(int layer_index, int expert_id) {
        return (std::uint64_t(std::uint32_t(layer_index)) << 32) | std::uint32_t(expert_id);
    }

    void touch(std::uint64_t key) {
        order.remove(key);
        order.push_back(key);
    }

    void evict_if_needed(std::uint64_t protected_key) {
        while (used_bytes > max_bytes && order.size() > 1) {
            const std::uint64_t victim = order.front();
            order.pop_front();
            if (victim == protected_key) {
                order.push_back(victim);
                continue;
            }
            auto found = items.find(victim);
            if (found == items.end()) {
                continue;
            }
            used_bytes -= static_cast<std::uint64_t>(found->second->bytes);
            items.erase(found);
        }
    }
};

static bool llama_infinitum_ggml_gpu_compute_expert(
        const llama_infinitum_ggml_packed_expert_mlp & expert,
        const float * hidden_data,
        int hidden_size,
        float * output,
        std::string & error) {
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    llama_infinitum_ggml_gpu_backend & state = llama_infinitum_ggml_gpu_backend_get();
    if (!state.get(backend, device, error)) {
        return false;
    }

    std::lock_guard<std::mutex> compute_guard(state.compute_mutex());
    static llama_infinitum_ggml_gpu_expert_cache gpu_cache;
    std::shared_ptr<llama_infinitum_ggml_gpu_expert_graph> graph =
        gpu_cache.get_or_create(expert, hidden_size, backend, device, error);
    if (graph == nullptr) {
        return false;
    }

    ggml_backend_tensor_set(graph->hidden, hidden_data, 0, std::size_t(hidden_size) * sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(backend, graph->graph);
    if (status != GGML_STATUS_SUCCESS) {
        error = "GPU backend failed computing selected expert graph";
        return false;
    }
    ggml_backend_tensor_get(graph->out, output, 0, std::size_t(hidden_size) * sizeof(float));
    return true;
}

struct llama_infinitum_ggml_gpu_expert_tensors {
    int layer_index = -1;
    int expert_id = -1;
    std::size_t bytes = 0;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * gate_w = nullptr;
    ggml_tensor * up_w = nullptr;
    ggml_tensor * down_w = nullptr;
    ggml_tensor * gate_b = nullptr;
    ggml_tensor * up_b = nullptr;
    ggml_tensor * down_b = nullptr;

    llama_infinitum_ggml_gpu_expert_tensors() = default;
    llama_infinitum_ggml_gpu_expert_tensors(const llama_infinitum_ggml_gpu_expert_tensors &) = delete;
    llama_infinitum_ggml_gpu_expert_tensors & operator=(const llama_infinitum_ggml_gpu_expert_tensors &) = delete;

    ~llama_infinitum_ggml_gpu_expert_tensors() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

static std::shared_ptr<llama_infinitum_ggml_gpu_expert_tensors> llama_infinitum_ggml_gpu_create_expert_tensors(
        const llama_infinitum_ggml_packed_expert_mlp & expert,
        int hidden_size,
        ggml_backend_t backend,
        std::string & error) {
    constexpr std::size_t ctx_mem = 4ull * 1024ull * 1024ull;
    ggml_init_params params = {};
    params.mem_size = ctx_mem;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        error = "failed to initialize ggml context for Vulkan expert tensors";
        return nullptr;
    }

    ggml_tensor * gate_w = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, hidden_size, hidden_size);
    ggml_tensor * up_w = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, hidden_size, hidden_size);
    ggml_tensor * down_w = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, hidden_size, hidden_size);
    ggml_tensor * gate_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 1);
    ggml_tensor * up_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 1);
    ggml_tensor * down_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 1);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        error = "failed to allocate GPU tensors for selected expert weights";
        ggml_free(ctx);
        return nullptr;
    }

    ggml_backend_tensor_set(gate_w, expert.gate_blocks.data(), 0, expert.gate_blocks.size());
    ggml_backend_tensor_set(up_w, expert.up_blocks.data(), 0, expert.up_blocks.size());
    ggml_backend_tensor_set(down_w, expert.down_blocks.data(), 0, expert.down_blocks.size());
    ggml_backend_tensor_set(gate_b, expert.gate_bias.data(), 0, expert.gate_bias.size() * sizeof(float));
    ggml_backend_tensor_set(up_b, expert.up_bias.data(), 0, expert.up_bias.size() * sizeof(float));
    ggml_backend_tensor_set(down_b, expert.down_bias.data(), 0, expert.down_bias.size() * sizeof(float));

    auto tensors = std::make_shared<llama_infinitum_ggml_gpu_expert_tensors>();
    tensors->layer_index = expert.layer_index;
    tensors->expert_id = expert.expert_id;
    tensors->bytes = ggml_backend_buffer_get_size(buffer);
    tensors->ctx = ctx;
    tensors->buffer = buffer;
    tensors->gate_w = gate_w;
    tensors->up_w = up_w;
    tensors->down_w = down_w;
    tensors->gate_b = gate_b;
    tensors->up_b = up_b;
    tensors->down_b = down_b;
    return tensors;
}

class llama_infinitum_ggml_gpu_expert_tensor_cache {
public:
    std::shared_ptr<llama_infinitum_ggml_gpu_expert_tensors> get_or_create(
            const llama_infinitum_ggml_packed_expert_mlp & expert,
            int hidden_size,
            ggml_backend_t backend,
            std::string & error) {
        std::lock_guard<std::mutex> lock(mutex);
        max_bytes = llama_infinitum_moe_gpu_cache_bytes_or_default();
        const std::uint64_t key = key_for(expert.layer_index, expert.expert_id);
        auto found = items.find(key);
        if (found != items.end()) {
            touch(key);
            return found->second;
        }

        auto tensors = llama_infinitum_ggml_gpu_create_expert_tensors(expert, hidden_size, backend, error);
        if (tensors == nullptr) {
            return nullptr;
        }
        if (tensors->bytes > max_bytes) {
            error = "selected expert GPU tensor set exceeds GPU cache budget";
            return nullptr;
        }

        used_bytes += static_cast<std::uint64_t>(tensors->bytes);
        items[key] = tensors;
        order.push_back(key);
        evict_if_needed(key);
        return tensors;
    }

private:
    std::mutex mutex;
    std::uint64_t max_bytes = 4096ull * 1024ull * 1024ull;
    std::uint64_t used_bytes = 0;
    std::unordered_map<std::uint64_t, std::shared_ptr<llama_infinitum_ggml_gpu_expert_tensors>> items;
    std::list<std::uint64_t> order;

    static std::uint64_t key_for(int layer_index, int expert_id) {
        return (std::uint64_t(std::uint32_t(layer_index)) << 32) | std::uint32_t(expert_id);
    }

    void touch(std::uint64_t key) {
        order.remove(key);
        order.push_back(key);
    }

    void evict_if_needed(std::uint64_t protected_key) {
        while (used_bytes > max_bytes && order.size() > 1) {
            const std::uint64_t victim = order.front();
            order.pop_front();
            if (victim == protected_key) {
                order.push_back(victim);
                continue;
            }
            auto found = items.find(victim);
            if (found == items.end()) {
                continue;
            }
            used_bytes -= static_cast<std::uint64_t>(found->second->bytes);
            items.erase(found);
        }
    }
};

class llama_infinitum_ggml_gpu_layer_slot_cache {
public:
    llama_infinitum_ggml_gpu_layer_slot_cache() : cache_uid(next_cache_uid.fetch_add(1, std::memory_order_relaxed)) {}

    ~llama_infinitum_ggml_gpu_layer_slot_cache() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }

    bool init(
            int layer,
            int hidden,
            int requested_slots,
            bool requested_global_residency,
            ggml_backend_t backend,
            std::string & error) {
        const int target_slots = std::max(1, requested_slots);
        if (ctx != nullptr) {
            if (hidden_size != hidden || slots != target_slots || global_residency != requested_global_residency || slots <= 0) {
                error = "GPU layer slot cache shape changed after initialization";
                return false;
            }
            if (!global_residency && layer_index != layer) {
                reset_slots_for_layer(layer);
            } else {
                layer_index = layer;
            }
            return true;
        }
        layer_index = layer;
        global_residency = requested_global_residency;
        hidden_size = hidden;
        slots = target_slots;
        block_count = llama_infinitum_mxfp4_block_count_for_hidden(hidden_size);
        if (block_count <= 0) {
            error = "invalid hidden block count for GPU layer slot cache";
            return false;
        }
        expert_matrix_bytes = std::size_t(hidden_size) * std::size_t(block_count) * 17ull;
        gate_up_matrix_bytes = expert_matrix_bytes * 2ull;

        constexpr std::size_t ctx_mem = 4ull * 1024ull * 1024ull;
        ggml_init_params params = {};
        params.mem_size = ctx_mem;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            error = "failed to initialize ggml context for GPU layer slot cache";
            return false;
        }

        gate_up_w = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, hidden_size, hidden_size * 2, slots);
        down_w = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, hidden_size, hidden_size, slots);
        gate_up_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size * 2, slots);
        down_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, slots);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buffer == nullptr) {
            error = "failed to allocate GPU layer slot cache";
            return false;
        }

        bytes_allocated = ggml_backend_buffer_get_size(buffer);
        slot_keys.assign(slots, empty_slot_key);
        slot_predicted.assign(slots, 0);
        slot_hit_counts.assign(slots, 0);
        slot_last_used.assign(slots, 0);
        return true;
    }

    bool ensure_expert(
            const llama_infinitum_ggml_packed_expert_mlp & expert,
            ggml_backend_t backend,
            int & out_slot,
            std::string & error,
            bool predicted_residency = false) {
        const std::uint64_t key = expert_key(expert);
        auto found = key_to_slot.find(key);
        if (found != key_to_slot.end()) {
            ++slot_hits;
            out_slot = found->second;
            if (!predicted_residency && out_slot >= 0 && out_slot < static_cast<int>(slot_hit_counts.size())) {
                slot_hit_counts[out_slot] = std::min<std::uint32_t>(slot_hit_counts[out_slot] + 1u, 1000000u);
            }
            touch(out_slot, predicted_residency);
            return true;
        }
        ++slot_misses;

        int slot = find_free_slot();
        if (slot < 0) {
            slot = global_residency ? choose_global_victim_slot() : (lru_slots.empty() ? 0 : lru_slots.front());
            if (slot < 0) {
                slot = 0;
            }
            lru_slots.remove(slot);
            const std::uint64_t previous_key = slot_keys[slot];
            if (previous_key != empty_slot_key) {
                key_to_slot.erase(previous_key);
                ++slot_evictions;
            }
            slot_keys[slot] = empty_slot_key;
            if (slot < static_cast<int>(slot_predicted.size())) {
                slot_predicted[slot] = 0;
            }
            if (slot < static_cast<int>(slot_hit_counts.size())) {
                slot_hit_counts[slot] = 0;
            }
            if (slot < static_cast<int>(slot_last_used.size())) {
                slot_last_used[slot] = 0;
            }
        }

        if (!upload(slot, expert, backend, error)) {
            return false;
        }
        slot_keys[slot] = key;
        key_to_slot[key] = slot;
        if (slot >= 0 && slot < static_cast<int>(slot_hit_counts.size())) {
            slot_hit_counts[slot] = predicted_residency ? 0u : 1u;
        }
        touch(slot, predicted_residency);
        out_slot = slot;
        return true;
    }

    ggml_tensor * gate_up_weights() const { return gate_up_w; }
    ggml_tensor * down_weights() const { return down_w; }
    ggml_tensor * gate_up_biases() const { return gate_up_b; }
    ggml_tensor * down_biases() const { return down_b; }
    std::uint64_t hits() const { return slot_hits; }
    std::uint64_t misses() const { return slot_misses; }
    std::uint64_t evictions() const { return slot_evictions; }
    std::uint64_t uid() const { return cache_uid; }
    std::size_t allocated_bytes() const { return bytes_allocated; }

private:
    int layer_index = -1;
    int hidden_size = 0;
    int slots = 0;
    int block_count = 0;
    bool global_residency = false;
    std::size_t expert_matrix_bytes = 0;
    std::size_t gate_up_matrix_bytes = 0;
    std::size_t bytes_allocated = 0;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * gate_up_w = nullptr;
    ggml_tensor * down_w = nullptr;
    ggml_tensor * gate_up_b = nullptr;
    ggml_tensor * down_b = nullptr;
    std::vector<std::uint64_t> slot_keys;
    std::vector<std::uint8_t> slot_predicted;
    std::vector<std::uint32_t> slot_hit_counts;
    std::vector<std::uint64_t> slot_last_used;
    std::unordered_map<std::uint64_t, int> key_to_slot;
    std::list<int> lru_slots;
    std::uint64_t access_clock = 0;
    std::uint64_t slot_hits = 0;
    std::uint64_t slot_misses = 0;
    std::uint64_t slot_evictions = 0;
    std::uint64_t cache_uid = 0;
    static constexpr std::uint64_t empty_slot_key = UINT64_MAX;
    static std::atomic<std::uint64_t> next_cache_uid;

    std::uint64_t expert_key(const llama_infinitum_ggml_packed_expert_mlp & expert) const {
        if (!global_residency) {
            return std::uint64_t(std::uint32_t(expert.expert_id));
        }
        return (std::uint64_t(std::uint32_t(expert.layer_index)) << 32) |
            std::uint64_t(std::uint32_t(expert.expert_id));
    }

    void reset_slots_for_layer(int layer) {
        layer_index = layer;
        std::fill(slot_keys.begin(), slot_keys.end(), empty_slot_key);
        std::fill(slot_predicted.begin(), slot_predicted.end(), 0);
        std::fill(slot_hit_counts.begin(), slot_hit_counts.end(), 0);
        std::fill(slot_last_used.begin(), slot_last_used.end(), 0);
        key_to_slot.clear();
        lru_slots.clear();
    }

    int find_free_slot() const {
        for (int i = 0; i < static_cast<int>(slot_keys.size()); ++i) {
            if (slot_keys[i] == empty_slot_key) {
                return i;
            }
        }
        return -1;
    }

    int choose_global_victim_slot() const {
        int best_slot = -1;
        std::int64_t best_score = INT64_MAX;
        for (int i = 0; i < static_cast<int>(slot_keys.size()); ++i) {
            if (slot_keys[i] == empty_slot_key) {
                return i;
            }
            const std::uint64_t last_used = i < static_cast<int>(slot_last_used.size()) ? slot_last_used[i] : 0;
            const std::uint64_t age = access_clock > last_used ? access_clock - last_used : 0;
            const std::uint32_t hits = i < static_cast<int>(slot_hit_counts.size()) ? slot_hit_counts[i] : 0;
            const bool predicted = i < static_cast<int>(slot_predicted.size()) && slot_predicted[i] != 0;
            std::int64_t score = predicted ? 1000000 : 0;
            score += static_cast<std::int64_t>(std::min<std::uint32_t>(hits, 1000u)) * 1000;
            score -= static_cast<std::int64_t>(std::min<std::uint64_t>(age, 100000ull));
            if (score < best_score) {
                best_score = score;
                best_slot = i;
            }
        }
        return best_slot;
    }

    void touch(int slot, bool predicted_residency) {
        if (slot < 0) {
            return;
        }
        ++access_clock;
        if (slot < static_cast<int>(slot_last_used.size())) {
            slot_last_used[slot] = access_clock;
        }
        if (slot < static_cast<int>(slot_predicted.size())) {
            slot_predicted[slot] = predicted_residency ? 1 : 0;
        }
        lru_slots.remove(slot);
        lru_slots.push_back(slot);
    }

    bool upload(int slot, const llama_infinitum_ggml_packed_expert_mlp & expert, ggml_backend_t backend, std::string & error) {
        if (expert.gate_blocks_size() != expert_matrix_bytes ||
                expert.up_blocks_size() != expert_matrix_bytes ||
                expert.down_blocks_size() != expert_matrix_bytes ||
                expert.gate_bias_size() != hidden_size ||
                expert.up_bias_size() != hidden_size ||
                expert.down_bias_size() != hidden_size) {
            error = "selected expert tensor shapes do not match GPU layer slot cache";
            return false;
        }

        thread_local std::vector<std::uint8_t> gate_up_scratch;
        thread_local std::vector<float> gate_up_bias_scratch;
        const std::uint8_t * gate_up_data = nullptr;
        if (expert.has_contiguous_gate_up()) {
            gate_up_data = expert.gate_blocks_data();
        } else {
            gate_up_scratch.resize(gate_up_matrix_bytes);
            std::memcpy(gate_up_scratch.data(), expert.gate_blocks_data(), expert_matrix_bytes);
            std::memcpy(gate_up_scratch.data() + expert_matrix_bytes, expert.up_blocks_data(), expert_matrix_bytes);
            gate_up_data = gate_up_scratch.data();
        }

        const float * gate_up_bias_data = nullptr;
        if (expert.has_contiguous_gate_up_bias()) {
            gate_up_bias_data = expert.gate_bias_data();
        } else {
            gate_up_bias_scratch.resize(std::size_t(hidden_size) * 2);
            std::memcpy(gate_up_bias_scratch.data(), expert.gate_bias_data(), std::size_t(hidden_size) * sizeof(float));
            std::memcpy(gate_up_bias_scratch.data() + hidden_size, expert.up_bias_data(), std::size_t(hidden_size) * sizeof(float));
            gate_up_bias_data = gate_up_bias_scratch.data();
        }

        ggml_backend_tensor_set_async(backend, gate_up_w, gate_up_data, std::size_t(slot) * gate_up_matrix_bytes, gate_up_matrix_bytes);
        ggml_backend_tensor_set_async(backend, down_w, expert.down_blocks_data(), std::size_t(slot) * expert_matrix_bytes, expert_matrix_bytes);
        ggml_backend_tensor_set_async(backend, gate_up_b, gate_up_bias_data, std::size_t(slot) * std::size_t(hidden_size) * 2ull * sizeof(float), std::size_t(hidden_size) * 2ull * sizeof(float));
        ggml_backend_tensor_set_async(backend, down_b, expert.down_bias_data(), std::size_t(slot) * std::size_t(hidden_size) * sizeof(float), std::size_t(hidden_size) * sizeof(float));
        return true;
    }
};

std::atomic<std::uint64_t> llama_infinitum_ggml_gpu_layer_slot_cache::next_cache_uid { 1 };

class llama_infinitum_ggml_gpu_layer_slot_cache_manager {
public:
    llama_infinitum_ggml_gpu_layer_slot_cache & get_layer(int layer_index) {
        if (llama_infinitum_moe_gpu_global_slots_enabled()) {
            if (global_layer == nullptr) {
                global_layer = std::make_unique<llama_infinitum_ggml_gpu_layer_slot_cache>();
            }
            global_layer_index = layer_index;
            return *global_layer;
        }
        if (llama_infinitum_moe_gpu_stream_layers_enabled()) {
            if (streaming_layer == nullptr) {
                streaming_layer = std::make_unique<llama_infinitum_ggml_gpu_layer_slot_cache>();
            }
            streaming_layer_index = layer_index;
            return *streaming_layer;
        }
        auto found = layers.find(layer_index);
        if (found == layers.end()) {
            if (!evict_if_needed(layer_index)) {
                if (transient_layer == nullptr) {
                    transient_layer = std::make_unique<llama_infinitum_ggml_gpu_layer_slot_cache>();
                }
                llama_infinitum_debug_log("using transient GPU expert layer cache layer=%d limit=%d",
                    layer_index,
                    llama_infinitum_moe_gpu_layer_cache_limit_from_env());
                return *transient_layer;
            }
            auto inserted = layers.emplace(layer_index, std::make_unique<llama_infinitum_ggml_gpu_layer_slot_cache>());
            touch(layer_index);
            return *inserted.first->second;
        }
        touch(layer_index);
        return *found->second;
    }

private:
    int global_layer_index = -1;
    std::unique_ptr<llama_infinitum_ggml_gpu_layer_slot_cache> global_layer;
    int streaming_layer_index = -1;
    std::unique_ptr<llama_infinitum_ggml_gpu_layer_slot_cache> streaming_layer;
    std::unique_ptr<llama_infinitum_ggml_gpu_layer_slot_cache> transient_layer;
    std::unordered_map<int, std::unique_ptr<llama_infinitum_ggml_gpu_layer_slot_cache>> layers;
    std::list<int> layer_lru;

    void touch(int layer_index) {
        layer_lru.remove(layer_index);
        layer_lru.push_back(layer_index);
    }

    bool evict_if_needed(int requested_layer) {
        const int limit = llama_infinitum_moe_gpu_layer_cache_limit_from_env();
        if (limit <= 0) {
            return true;
        }
        while (static_cast<int>(layers.size()) >= limit && !layer_lru.empty()) {
            auto victim_it = layer_lru.end();
            for (auto it = layer_lru.begin(); it != layer_lru.end(); ++it) {
                if (*it < requested_layer) {
                    victim_it = it;
                    break;
                }
            }
            if (victim_it == layer_lru.end()) {
                return false;
            }
            const int victim = *victim_it;
            layer_lru.erase(victim_it);
            auto found = layers.find(victim);
            if (found != layers.end()) {
                llama_infinitum_debug_log("evicting GPU expert layer cache layer=%d limit=%d bytes=%llu",
                    victim,
                    limit,
                    static_cast<unsigned long long>(found->second->allocated_bytes()));
                layers.erase(found);
                return true;
            }
        }
        return static_cast<int>(layers.size()) < limit;
    }
};

static llama_infinitum_ggml_gpu_layer_slot_cache_manager & llama_infinitum_ggml_gpu_layer_slot_cache_manager_get() {
    static llama_infinitum_ggml_gpu_layer_slot_cache_manager manager;
    return manager;
}

bool llama_infinitum_moe_prefetch_selected_gpu_experts(
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        const std::vector<int> & expert_ids,
        std::string & error) {
    error.clear();
    if (expert_ids.empty()) {
        return true;
    }
    if (!llama_infinitum_moe_ggml_pack_enabled() || !llama_infinitum_moe_ggml_pack_slots_enabled()) {
        error = "GPU expert prefetch requires GGML expert pack slots";
        return false;
    }
    const llama_infinitum_moe_backend_kind backend_kind = llama_infinitum_moe_backend_kind_from_env();
    if (backend_kind != llama_infinitum_moe_backend_kind::vulkan &&
            backend_kind != llama_infinitum_moe_backend_kind::fused_arena) {
        error = "GPU expert prefetch requires Vulkan expert backend";
        return false;
    }

    const int hidden_size = info.hidden_size;
    const int block_count = llama_infinitum_mxfp4_block_count_for_hidden(hidden_size);
    if (hidden_size <= 0 || block_count <= 0) {
        error = "invalid hidden shape for GPU expert prefetch";
        return false;
    }

    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    llama_infinitum_ggml_gpu_backend & state = llama_infinitum_ggml_gpu_backend_get();
    if (!state.get(backend, device, error)) {
        return false;
    }
    (void) device;

    llama_infinitum_ggml_pack_expert_cache & pack_cache = llama_infinitum_ggml_pack_expert_cache_get();
    std::vector<std::shared_ptr<const llama_infinitum_ggml_packed_expert_mlp>> packed_experts;
    packed_experts.reserve(expert_ids.size());
    for (const int expert_id : expert_ids) {
        bool duplicate = false;
        for (const std::shared_ptr<const llama_infinitum_ggml_packed_expert_mlp> & packed : packed_experts) {
            if (packed->expert_id == expert_id) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }
        auto packed = pack_cache.get_or_load(info, layer_index, expert_id, hidden_size, block_count, error);
        if (packed == nullptr) {
            return false;
        }
        packed_experts.push_back(std::move(packed));
    }

    std::unique_lock<std::mutex> compute_guard(state.compute_mutex(), std::try_to_lock);
    if (!compute_guard.owns_lock()) {
        return true;
    }
    llama_infinitum_ggml_gpu_layer_slot_cache & layer_cache =
        llama_infinitum_ggml_gpu_layer_slot_cache_manager_get().get_layer(layer_index);
    if (!layer_cache.init(
            layer_index,
            hidden_size,
            llama_infinitum_moe_gpu_selected_slots_from_env(),
            llama_infinitum_moe_gpu_global_slots_enabled(),
            backend,
            error)) {
        return false;
    }

    for (const std::shared_ptr<const llama_infinitum_ggml_packed_expert_mlp> & packed : packed_experts) {
        int slot = -1;
        if (!layer_cache.ensure_expert(*packed, backend, slot, error, true)) {
            return false;
        }
    }
    return true;
}

class llama_infinitum_ggml_gpu_layer_compute_graph {
public:
    llama_infinitum_ggml_gpu_layer_compute_graph() = default;
    llama_infinitum_ggml_gpu_layer_compute_graph(const llama_infinitum_ggml_gpu_layer_compute_graph &) = delete;
    llama_infinitum_ggml_gpu_layer_compute_graph & operator=(const llama_infinitum_ggml_gpu_layer_compute_graph &) = delete;

    ~llama_infinitum_ggml_gpu_layer_compute_graph() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }

    bool init(
            llama_infinitum_ggml_gpu_layer_slot_cache & layer_cache,
            int layer,
            int hidden,
            int selected_count,
            bool use_q8_input,
            bool use_f16_input,
            ggml_backend_t backend,
            ggml_backend_dev_t device,
            std::string & error) {
        if (ctx != nullptr) {
            return layer_index == layer && hidden_size == hidden && expert_count == selected_count &&
                q8_input == use_q8_input && f16_input == use_f16_input;
        }

        layer_index = layer;
        hidden_size = hidden;
        expert_count = selected_count;
        q8_input = use_q8_input;
        f16_input = use_f16_input && !use_q8_input;

        constexpr std::size_t ctx_mem = 16ull * 1024ull * 1024ull;
        ggml_init_params params = {};
        params.mem_size = ctx_mem;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            error = "failed to initialize ggml context for cached Vulkan mul_mat_id graph";
            return false;
        }

        hidden_tensor = ggml_new_tensor_3d(ctx, q8_input ? GGML_TYPE_Q8_1 : (f16_input ? GGML_TYPE_F16 : GGML_TYPE_F32), hidden_size, 1, 1);
        ids_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, expert_count, 1);
        weights_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, expert_count, 1);

        ggml_tensor * gate_up = ggml_mul_mat_id(ctx, layer_cache.gate_up_weights(), hidden_tensor, ids_tensor);
        gate_up = ggml_add_id(ctx, gate_up, layer_cache.gate_up_biases(), ids_tensor);
        ggml_tensor * gate = ggml_view_3d(ctx, gate_up, hidden_size, expert_count, 1, gate_up->nb[1], gate_up->nb[2], 0);
        ggml_tensor * up = ggml_view_3d(ctx, gate_up, hidden_size, expert_count, 1, gate_up->nb[1], gate_up->nb[2], std::size_t(hidden_size) * gate_up->nb[0]);
        ggml_tensor * act = ggml_swiglu_oai(ctx, gate, up, 1.702f, 7.0f);
        ggml_tensor * experts_out = ggml_mul_mat_id(ctx, layer_cache.down_weights(), act, ids_tensor);
        experts_out = ggml_add_id(ctx, experts_out, layer_cache.down_biases(), ids_tensor);
        experts_out = ggml_mul(ctx, experts_out, weights_tensor);

        sum_tensor = nullptr;
        for (int i = 0; i < expert_count; ++i) {
            ggml_tensor * expert_view = ggml_view_2d(ctx, experts_out, hidden_size, 1, experts_out->nb[2], std::size_t(i) * experts_out->nb[1]);
            sum_tensor = sum_tensor == nullptr ? expert_view : ggml_add(ctx, sum_tensor, expert_view);
        }

        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, sum_tensor);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (!ggml_backend_dev_supports_op(device, ggml_graph_node(graph, i))) {
                error = "GPU backend does not support a node in the cached mul_mat_id selected expert graph";
                return false;
            }
        }

        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buffer == nullptr) {
            error = "failed to allocate GPU tensors for cached Vulkan mul_mat_id graph";
            return false;
        }
        return true;
    }

    bool compute(
            const float * hidden_data,
            const std::int32_t * local_ids,
            const float * local_weights,
            ggml_backend_t backend,
            float * output,
            std::string & error,
            double * input_upload_ms = nullptr,
            double * graph_compute_ms = nullptr,
            double * output_get_ms = nullptr) {
        if (ctx == nullptr || buffer == nullptr || graph == nullptr || hidden_tensor == nullptr ||
                ids_tensor == nullptr || weights_tensor == nullptr || sum_tensor == nullptr) {
            error = "cached Vulkan mul_mat_id graph is not initialized";
            return false;
        }

        const auto input_start = std::chrono::steady_clock::now();
        if (q8_input) {
            thread_local std::vector<std::uint8_t> hidden_q8_scratch;
            const std::size_t hidden_q8_bytes = ggml_row_size(GGML_TYPE_Q8_1, hidden_size);
            hidden_q8_scratch.resize(hidden_q8_bytes);
            const ggml_type_traits * q8_traits = ggml_get_type_traits(GGML_TYPE_Q8_1);
            if (q8_traits == nullptr || q8_traits->from_float_ref == nullptr) {
                error = "GGML_TYPE_Q8_1 quantizer is unavailable";
                return false;
            }
            q8_traits->from_float_ref(hidden_data, hidden_q8_scratch.data(), hidden_size);
            ggml_backend_tensor_set_async(backend, hidden_tensor, hidden_q8_scratch.data(), 0, hidden_q8_bytes);
        } else if (f16_input) {
            thread_local std::vector<ggml_fp16_t> hidden_f16_scratch;
            hidden_f16_scratch.resize(std::size_t(hidden_size));
            ggml_fp32_to_fp16_row(hidden_data, hidden_f16_scratch.data(), hidden_size);
            ggml_backend_tensor_set_async(backend, hidden_tensor, hidden_f16_scratch.data(), 0, hidden_f16_scratch.size() * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set_async(backend, hidden_tensor, hidden_data, 0, std::size_t(hidden_size) * sizeof(float));
        }
        ggml_backend_tensor_set_async(backend, ids_tensor, local_ids, 0, std::size_t(expert_count) * sizeof(std::int32_t));
        ggml_backend_tensor_set_async(backend, weights_tensor, local_weights, 0, std::size_t(expert_count) * sizeof(float));
        const auto input_end = std::chrono::steady_clock::now();

        const auto graph_start = std::chrono::steady_clock::now();
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        const auto graph_end = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            error = "GPU backend failed computing cached Vulkan mul_mat_id selected expert graph";
            return false;
        }

        const auto output_start = std::chrono::steady_clock::now();
        ggml_backend_tensor_get(sum_tensor, output, 0, std::size_t(hidden_size) * sizeof(float));
        const auto output_end = std::chrono::steady_clock::now();
        if (input_upload_ms != nullptr) {
            *input_upload_ms = llama_infinitum_elapsed_ms(input_start, input_end);
        }
        if (graph_compute_ms != nullptr) {
            *graph_compute_ms = llama_infinitum_elapsed_ms(graph_start, graph_end);
        }
        if (output_get_ms != nullptr) {
            *output_get_ms = llama_infinitum_elapsed_ms(output_start, output_end);
        }
        return true;
    }

private:
    int layer_index = -1;
    int hidden_size = 0;
    int expert_count = 0;
    bool q8_input = false;
    bool f16_input = false;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * hidden_tensor = nullptr;
    ggml_tensor * ids_tensor = nullptr;
    ggml_tensor * weights_tensor = nullptr;
    ggml_tensor * sum_tensor = nullptr;
};

class llama_infinitum_ggml_gpu_layer_compute_graph_manager {
public:
    llama_infinitum_ggml_gpu_layer_compute_graph & get_or_create(
            llama_infinitum_ggml_gpu_layer_slot_cache & layer_cache,
            int layer_index,
            int hidden_size,
            int expert_count,
            bool q8_input,
            bool f16_input,
            ggml_backend_t backend,
            ggml_backend_dev_t device,
            std::string & error) {
        const std::uint64_t layer_component = llama_infinitum_moe_gpu_global_slots_enabled() ?
            0ull : (llama_infinitum_moe_gpu_stream_layers_enabled() ? 0ull : std::uint64_t(std::uint32_t(layer_index)));
        const std::uint64_t cache_component = (llama_infinitum_moe_gpu_global_slots_enabled() ||
                llama_infinitum_moe_gpu_layer_cache_limit_from_env() > 0) ?
            (layer_cache.uid() & 0xFFFFull) : 0ull;
        const std::uint64_t key =
            (layer_component << 40) |
            (cache_component << 16) |
            (std::uint64_t(std::uint16_t(expert_count)) << 2) |
            (q8_input ? 1ull : 0ull) |
            (f16_input ? 2ull : 0ull);
        if (llama_infinitum_moe_gpu_stream_layers_enabled()) {
            if (streaming_key != key || streaming_graph == nullptr) {
                auto graph = std::make_unique<llama_infinitum_ggml_gpu_layer_compute_graph>();
                if (!graph->init(layer_cache, layer_index, hidden_size, expert_count, q8_input, f16_input, backend, device, error)) {
                    auto failed = std::move(graph);
                    (void) failed;
                    return invalid_graph;
                }
                streaming_key = key;
                streaming_graph = std::move(graph);
            }
            return *streaming_graph;
        }
        auto found = graphs.find(key);
        if (found == graphs.end()) {
            auto graph = std::make_unique<llama_infinitum_ggml_gpu_layer_compute_graph>();
            if (!graph->init(layer_cache, layer_index, hidden_size, expert_count, q8_input, f16_input, backend, device, error)) {
                auto failed = std::move(graph);
                (void) failed;
                return invalid_graph;
            }
            auto inserted = graphs.emplace(key, std::move(graph));
            return *inserted.first->second;
        }
        return *found->second;
    }

private:
    std::uint64_t streaming_key = UINT64_MAX;
    std::unique_ptr<llama_infinitum_ggml_gpu_layer_compute_graph> streaming_graph;
    std::unordered_map<std::uint64_t, std::unique_ptr<llama_infinitum_ggml_gpu_layer_compute_graph>> graphs;
    llama_infinitum_ggml_gpu_layer_compute_graph invalid_graph;
};

class llama_infinitum_gemma4_q4_gpu_layer_slot_cache {
public:
    ~llama_infinitum_gemma4_q4_gpu_layer_slot_cache() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }

    bool init(
            const llama_infinitum_moe_index_info & info,
            int layer,
            int hidden,
            int gate_up,
            int down,
            int requested_slots,
            int min_slots,
            ggml_backend_t backend,
            std::string & error) {
        if (ctx != nullptr) {
            return layer_index == layer && hidden_size == hidden && gate_up_rows == gate_up &&
                down_rows == down && slots > 0;
        }
        if (hidden <= 0 || gate_up <= 0 || down <= 0 || gate_up % 2 != 0) {
            error = "invalid Gemma4 Q4 slot cache shape";
            return false;
        }
        layer_index = layer;
        hidden_size = hidden;
        gate_up_rows = gate_up;
        down_rows = down;
        expert_count = llama_infinitum_moe_expert_count(info);
        source_type = llama_infinitum_gemma4_pack_type_from_env();
        resident_type = source_type == GGML_TYPE_F16 || llama_infinitum_gemma4_q4_slot_f16_enabled() ? GGML_TYPE_F16 : GGML_TYPE_Q4_0;
        if (source_type == GGML_TYPE_Q8_0 && resident_type != GGML_TYPE_F16) {
            resident_type = GGML_TYPE_Q8_0;
        }
        gate_up_source_bytes = ggml_row_size(source_type, hidden_size) * std::size_t(gate_up_rows);
        down_source_bytes = ggml_row_size(source_type, down_rows) * std::size_t(hidden_size);
        gate_up_resident_bytes = ggml_row_size(resident_type, hidden_size) * std::size_t(gate_up_rows);
        down_resident_bytes = ggml_row_size(resident_type, down_rows) * std::size_t(hidden_size);
        gate_up_layer_bytes = gate_up_source_bytes * std::size_t(expert_count);
        down_layer_bytes = down_source_bytes * std::size_t(expert_count);
        if (expert_count <= 0 || gate_up_source_bytes == 0 || down_source_bytes == 0 ||
                gate_up_resident_bytes == 0 || down_resident_bytes == 0) {
            error = "invalid Gemma4 Q4 expert pack geometry";
            return false;
        }
        const int layer_count = std::max(1, llama_infinitum_moe_layer_count(info));
        const std::size_t slot_bytes = gate_up_resident_bytes + down_resident_bytes;
        const std::uint64_t configured_gpu_cache = llama_infinitum_moe_gpu_cache_bytes_from_env();
        const std::uint64_t auto_budget = configured_gpu_cache != 0 ?
            configured_gpu_cache : (2048ull * 1024ull * 1024ull);
        const int auto_slots = slot_bytes == 0 ? min_slots :
            static_cast<int>(std::max<std::uint64_t>(1, auto_budget / std::uint64_t(layer_count) / std::uint64_t(slot_bytes)));
        slots = requested_slots > 0 ? requested_slots : auto_slots;
        slots = std::max(slots, std::max(1, min_slots));
        slots = std::min(slots, expert_count);
        if (llama_infinitum_debug_enabled()) {
            llama_infinitum_debug_log(
                "gemma4 slots layer=%d slots=%d requested=%d min=%d budget_mb=%llu slot_bytes=%zu layers=%d source=%s resident=%s",
                layer_index, slots, requested_slots, min_slots,
                static_cast<unsigned long long>(auto_budget / (1024ull * 1024ull)),
                slot_bytes, layer_count, ggml_type_name(source_type), ggml_type_name(resident_type));
        }

        std::string mapping_error;
        mapping = llama_infinitum_moe_ggml_pack_mapping_for(llama_infinitum_moe_ggml_pack_path(info), mapping_error);
        if (mapping == nullptr) {
            error = "failed to mmap Gemma4 Q4 expert pack: " + mapping_error;
            return false;
        }

        const std::size_t layer_base = std::size_t(layer_index) * (gate_up_layer_bytes + down_layer_bytes);
        if (layer_base + gate_up_layer_bytes + down_layer_bytes > mapping->file->size()) {
            error = "Gemma4 Q4 expert pack is smaller than expected for requested layer";
            return false;
        }

        constexpr std::size_t ctx_mem = 4ull * 1024ull * 1024ull;
        ggml_init_params params = {};
        params.mem_size = ctx_mem;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            error = "failed to initialize ggml context for Gemma4 Q4 GPU layer slot cache";
            return false;
        }

        gate_up_w = ggml_new_tensor_3d(ctx, resident_type, hidden_size, gate_up_rows, slots);
        down_w = ggml_new_tensor_3d(ctx, resident_type, down_rows, hidden_size, slots);
        gate_up_slots.resize(slots);
        down_slots.resize(slots);
        for (int s = 0; s < slots; ++s) {
            gate_up_slots[s] = ggml_new_tensor_2d(ctx, resident_type, hidden_size, gate_up_rows);
            down_slots[s] = ggml_new_tensor_2d(ctx, resident_type, down_rows, hidden_size);
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buffer == nullptr) {
            error = "failed to allocate Gemma4 Q4 GPU layer slot cache";
            return false;
        }
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        slot_expert_ids.assign(slots, -1);
        return true;
    }

    bool ensure_expert(int expert_id, int & out_slot, std::string & error) {
        std::vector<int> expert_ids = { expert_id };
        std::vector<std::int32_t> slots_out(1);
        if (!ensure_experts(expert_ids, slots_out, error)) {
            return false;
        }
        out_slot = slots_out[0];
        return true;
    }

    bool ensure_experts(
            const std::vector<int> & expert_ids,
            std::vector<std::int32_t> & out_slots,
            std::string & error) {
        if (out_slots.size() < expert_ids.size()) {
            out_slots.resize(expert_ids.size());
        }

        std::vector<pending_upload> pending;
        pending.reserve(expert_ids.size());

        for (std::size_t i = 0; i < expert_ids.size(); ++i) {
            const int expert_id = expert_ids[i];
            auto found = expert_to_slot.find(expert_id);
            if (found != expert_to_slot.end()) {
                out_slots[i] = found->second;
                touch(found->second);
                ++stats_hits;
                continue;
            }
            if (expert_id < 0 || expert_id >= expert_count) {
                error = "Gemma4 selected expert id is out of range";
                return false;
            }
            ++stats_misses;

            int slot = find_free_slot();
            if (slot < 0) {
                slot = lru_slots.empty() ? 0 : lru_slots.front();
                lru_slots.remove(slot);
                const int previous_expert = slot_expert_ids[slot];
                if (previous_expert >= 0) {
                    expert_to_slot.erase(previous_expert);
                    ++stats_evictions;
                }
            }

            slot_expert_ids[slot] = -2;
            out_slots[i] = slot;
            pending.push_back({ expert_id, slot });
        }

        if (!pending.empty() && !upload_many(pending, error)) {
            for (const pending_upload & item : pending) {
                if (item.slot >= 0 && item.slot < static_cast<int>(slot_expert_ids.size()) &&
                        slot_expert_ids[item.slot] == -2) {
                    slot_expert_ids[item.slot] = -1;
                }
            }
            return false;
        }

        for (const pending_upload & item : pending) {
            slot_expert_ids[item.slot] = item.expert_id;
            expert_to_slot[item.expert_id] = item.slot;
            touch(item.slot);
        }
        return true;
    }

    bool ensure_expert_legacy(int expert_id, int & out_slot, std::string & error) {
        auto found = expert_to_slot.find(expert_id);
        if (found != expert_to_slot.end()) {
            out_slot = found->second;
            touch(out_slot);
            ++stats_hits;
            return true;
        }
        if (expert_id < 0 || expert_id >= expert_count) {
            error = "Gemma4 selected expert id is out of range";
            return false;
        }
        ++stats_misses;

        int slot = find_free_slot();
        if (slot < 0) {
            slot = lru_slots.empty() ? 0 : lru_slots.front();
            lru_slots.remove(slot);
            const int previous_expert = slot_expert_ids[slot];
            if (previous_expert >= 0) {
                expert_to_slot.erase(previous_expert);
                ++stats_evictions;
            }
        }

        if (!upload(slot, expert_id, error)) {
            return false;
        }
        slot_expert_ids[slot] = expert_id;
        expert_to_slot[expert_id] = slot;
        touch(slot);
        out_slot = slot;
        return true;
    }

    ggml_tensor * gate_up_weights() const { return gate_up_w; }
    ggml_tensor * down_weights() const { return down_w; }
    ggml_tensor * gate_up_slot_tensor(int slot) const { return slot >= 0 && slot < static_cast<int>(gate_up_slots.size()) ? gate_up_slots[slot] : nullptr; }
    ggml_tensor * down_slot_tensor(int slot) const { return slot >= 0 && slot < static_cast<int>(down_slots.size()) ? down_slots[slot] : nullptr; }
    int gate_up_row_count() const { return gate_up_rows; }
    int down_row_count() const { return down_rows; }
    std::size_t gate_up_row_stride() const { return static_cast<std::size_t>(gate_up_w->nb[1]); }
    std::size_t down_row_stride() const { return static_cast<std::size_t>(down_w->nb[1]); }
    std::size_t gate_up_slot_stride() const { return static_cast<std::size_t>(gate_up_w->nb[2]); }
    std::size_t down_slot_stride() const { return static_cast<std::size_t>(down_w->nb[2]); }
    std::uint64_t cache_hits() const { return stats_hits; }
    std::uint64_t cache_misses() const { return stats_misses; }
    std::uint64_t cache_evictions() const { return stats_evictions; }
    std::uint64_t uploaded_bytes() const { return stats_uploaded_bytes; }
    std::uint64_t resident_bytes() const { return std::uint64_t(slots) * std::uint64_t(gate_up_resident_bytes + down_resident_bytes); }

private:
    int layer_index = -1;
    int hidden_size = 0;
    int gate_up_rows = 0;
    int down_rows = 0;
    int expert_count = 0;
    int slots = 0;
    ggml_type source_type = GGML_TYPE_Q4_0;
    ggml_type resident_type = GGML_TYPE_Q4_0;
    std::size_t gate_up_source_bytes = 0;
    std::size_t down_source_bytes = 0;
    std::size_t gate_up_resident_bytes = 0;
    std::size_t down_resident_bytes = 0;
    std::size_t gate_up_layer_bytes = 0;
    std::size_t down_layer_bytes = 0;
    std::shared_ptr<llama_infinitum_ggml_pack_mapping> mapping;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * gate_up_w = nullptr;
    ggml_tensor * down_w = nullptr;
    std::vector<ggml_tensor *> gate_up_slots;
    std::vector<ggml_tensor *> down_slots;
    std::vector<int> slot_expert_ids;
    std::unordered_map<int, int> expert_to_slot;
    std::list<int> lru_slots;
    std::uint64_t stats_hits = 0;
    std::uint64_t stats_misses = 0;
    std::uint64_t stats_evictions = 0;
    std::uint64_t stats_uploaded_bytes = 0;
    std::vector<std::uint8_t> gate_up_batch_scratch;
    std::vector<std::uint8_t> down_batch_scratch;
    std::vector<float> dequant_row_scratch;
    std::vector<ggml_fp16_t> f16_upload_scratch;

    struct pending_upload {
        int expert_id = -1;
        int slot = -1;
    };

    int find_free_slot() const {
        for (int i = 0; i < static_cast<int>(slot_expert_ids.size()); ++i) {
            if (slot_expert_ids[i] < 0) {
                return i;
            }
        }
        return -1;
    }

    void touch(int slot) {
        lru_slots.remove(slot);
        lru_slots.push_back(slot);
    }

    void upload_quantized_as_f16(
            ggml_tensor * tensor,
            const std::uint8_t * source,
            ggml_type type,
            int rows,
            int cols,
            int slot,
            std::size_t slot_stride) {
        const ggml_type_traits * traits = ggml_get_type_traits(type);
        const std::size_t row_bytes = ggml_row_size(type, cols);
        dequant_row_scratch.resize(std::size_t(cols));
        f16_upload_scratch.resize(std::size_t(rows) * std::size_t(cols));
        for (int row = 0; row < rows; ++row) {
            const std::uint8_t * quant_row = source + std::size_t(row) * row_bytes;
            float * f32_row = dequant_row_scratch.data();
            traits->to_float(quant_row, f32_row, cols);
            ggml_fp32_to_fp16_row(
                    f32_row,
                    f16_upload_scratch.data() + std::size_t(row) * std::size_t(cols),
                    cols);
        }
        ggml_backend_tensor_set(
                tensor,
                f16_upload_scratch.data(),
                std::size_t(slot) * slot_stride,
                f16_upload_scratch.size() * sizeof(ggml_fp16_t));
    }

    bool upload(int slot, int expert_id, std::string & error) {
        if (mapping == nullptr || mapping->mapping == nullptr) {
            error = "Gemma4 Q4 expert pack mapping is not initialized";
            return false;
        }
        const std::size_t layer_base = std::size_t(layer_index) * (gate_up_layer_bytes + down_layer_bytes);
        const std::size_t gate_up_offset = layer_base + std::size_t(expert_id) * gate_up_source_bytes;
        const std::size_t down_offset = layer_base + gate_up_layer_bytes + std::size_t(expert_id) * down_source_bytes;
        if (down_offset + down_source_bytes > mapping->file->size()) {
            error = "Gemma4 Q4 expert slot upload would read past pack end";
            return false;
        }
        const auto * base = static_cast<const std::uint8_t *>(mapping->mapping->addr());
        if ((source_type == GGML_TYPE_Q4_0 || source_type == GGML_TYPE_Q8_0) && resident_type == GGML_TYPE_F16) {
            upload_quantized_as_f16(gate_up_w, base + gate_up_offset, source_type, gate_up_rows, hidden_size, slot, gate_up_slot_stride());
            upload_quantized_as_f16(down_w, base + down_offset, source_type, hidden_size, down_rows, slot, down_slot_stride());
            if (slot >= 0 && slot < static_cast<int>(gate_up_slots.size()) && gate_up_slots[slot] != nullptr) {
                upload_quantized_as_f16(gate_up_slots[slot], base + gate_up_offset, source_type, gate_up_rows, hidden_size, 0, 0);
                upload_quantized_as_f16(down_slots[slot], base + down_offset, source_type, hidden_size, down_rows, 0, 0);
            }
        } else {
            ggml_backend_tensor_set(gate_up_w, base + gate_up_offset, std::size_t(slot) * gate_up_slot_stride(), gate_up_source_bytes);
            ggml_backend_tensor_set(down_w, base + down_offset, std::size_t(slot) * down_slot_stride(), down_source_bytes);
            if (slot >= 0 && slot < static_cast<int>(gate_up_slots.size()) && gate_up_slots[slot] != nullptr) {
                ggml_backend_tensor_set(gate_up_slots[slot], base + gate_up_offset, 0, gate_up_source_bytes);
                ggml_backend_tensor_set(down_slots[slot], base + down_offset, 0, down_source_bytes);
            }
        }
        stats_uploaded_bytes += std::uint64_t(gate_up_source_bytes + down_source_bytes);
        return true;
    }

    bool upload_many(std::vector<pending_upload> pending, std::string & error) {
        if (mapping == nullptr || mapping->mapping == nullptr) {
            error = "Gemma4 Q4 expert pack mapping is not initialized";
            return false;
        }
        if (pending.empty()) {
            return true;
        }
        if ((source_type == GGML_TYPE_Q4_0 || source_type == GGML_TYPE_Q8_0) && resident_type == GGML_TYPE_F16) {
            for (const pending_upload & item : pending) {
                if (!upload(item.slot, item.expert_id, error)) {
                    return false;
                }
            }
            return true;
        }
        std::sort(pending.begin(), pending.end(), [](const pending_upload & a, const pending_upload & b) {
            return a.slot < b.slot;
        });

        const auto * base = static_cast<const std::uint8_t *>(mapping->mapping->addr());
        const std::size_t layer_base = std::size_t(layer_index) * (gate_up_layer_bytes + down_layer_bytes);
        std::size_t begin = 0;
        while (begin < pending.size()) {
            std::size_t end = begin + 1;
            while (end < pending.size() && pending[end].slot == pending[end - 1].slot + 1) {
                ++end;
            }

            const std::size_t count = end - begin;
            gate_up_batch_scratch.resize(count * gate_up_source_bytes);
            down_batch_scratch.resize(count * down_source_bytes);
            for (std::size_t i = 0; i < count; ++i) {
                const int expert_id = pending[begin + i].expert_id;
                const std::size_t gate_up_offset = layer_base + std::size_t(expert_id) * gate_up_source_bytes;
                const std::size_t down_offset = layer_base + gate_up_layer_bytes + std::size_t(expert_id) * down_source_bytes;
                if (down_offset + down_source_bytes > mapping->file->size()) {
                    error = "Gemma4 Q4 expert slot batch upload would read past pack end";
                    return false;
                }
                std::memcpy(gate_up_batch_scratch.data() + i * gate_up_source_bytes, base + gate_up_offset, gate_up_source_bytes);
                std::memcpy(down_batch_scratch.data() + i * down_source_bytes, base + down_offset, down_source_bytes);
            }

            const int start_slot = pending[begin].slot;
            const bool packed_slot_strides =
                gate_up_slot_stride() == gate_up_source_bytes &&
                down_slot_stride() == down_source_bytes;
            if (packed_slot_strides) {
                ggml_backend_tensor_set(
                        gate_up_w,
                        gate_up_batch_scratch.data(),
                        std::size_t(start_slot) * gate_up_slot_stride(),
                        count * gate_up_source_bytes);
                ggml_backend_tensor_set(
                        down_w,
                        down_batch_scratch.data(),
                        std::size_t(start_slot) * down_slot_stride(),
                        count * down_source_bytes);
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const int slot = pending[begin + i].slot;
                    ggml_backend_tensor_set(
                            gate_up_w,
                            gate_up_batch_scratch.data() + i * gate_up_source_bytes,
                            std::size_t(slot) * gate_up_slot_stride(),
                            gate_up_source_bytes);
                    ggml_backend_tensor_set(
                            down_w,
                            down_batch_scratch.data() + i * down_source_bytes,
                            std::size_t(slot) * down_slot_stride(),
                            down_source_bytes);
                }
            }
            stats_uploaded_bytes += std::uint64_t(count) * std::uint64_t(gate_up_source_bytes + down_source_bytes);
            begin = end;
        }
        return true;
    }
};

class llama_infinitum_gemma4_q4_gpu_layer_slot_cache_manager {
public:
    llama_infinitum_gemma4_q4_gpu_layer_slot_cache & get_layer(int layer_index) {
        auto found = layers.find(layer_index);
        if (found == layers.end()) {
            auto inserted = layers.emplace(layer_index, std::make_unique<llama_infinitum_gemma4_q4_gpu_layer_slot_cache>());
            return *inserted.first->second;
        }
        return *found->second;
    }

private:
    std::unordered_map<int, std::unique_ptr<llama_infinitum_gemma4_q4_gpu_layer_slot_cache>> layers;
};

class llama_infinitum_gemma4_q4_gpu_layer_compute_graph {
public:
    llama_infinitum_gemma4_q4_gpu_layer_compute_graph() = default;
    llama_infinitum_gemma4_q4_gpu_layer_compute_graph(const llama_infinitum_gemma4_q4_gpu_layer_compute_graph &) = delete;
    llama_infinitum_gemma4_q4_gpu_layer_compute_graph & operator=(const llama_infinitum_gemma4_q4_gpu_layer_compute_graph &) = delete;

    ~llama_infinitum_gemma4_q4_gpu_layer_compute_graph() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }

    bool init(
            llama_infinitum_gemma4_q4_gpu_layer_slot_cache & layer_cache,
            int layer,
            int hidden,
            int selected_count,
            ggml_backend_t backend,
            ggml_backend_dev_t device,
            std::string & error) {
        if (ctx != nullptr) {
            return layer_index == layer && hidden_size == hidden && expert_count == selected_count;
        }

        layer_index = layer;
        hidden_size = hidden;
        expert_count = selected_count;
        const int gate_up_rows = layer_cache.gate_up_row_count();
        const int down_rows = layer_cache.down_row_count();
        if (gate_up_rows != 2 * down_rows || down_rows <= 0) {
            error = "Gemma4 Q4 selected expert graph has invalid gate/up/down rows";
            return false;
        }

        constexpr std::size_t ctx_mem = 16ull * 1024ull * 1024ull;
        ggml_init_params params = {};
        params.mem_size = ctx_mem;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            error = "failed to initialize ggml context for Gemma4 Q4 selected expert graph";
            return false;
        }

        hidden_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_size, 1, 1);
        ids_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, expert_count, 1);
        weights_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, expert_count, 1);

        ggml_tensor * gate_up = ggml_mul_mat_id(ctx, layer_cache.gate_up_weights(), hidden_tensor, ids_tensor);
        ggml_tensor * gate = ggml_view_3d(ctx, gate_up, down_rows, expert_count, 1, gate_up->nb[1], gate_up->nb[2], 0);
        ggml_tensor * up = ggml_view_3d(ctx, gate_up, down_rows, expert_count, 1, gate_up->nb[1], gate_up->nb[2], std::size_t(down_rows) * gate_up->nb[0]);
        ggml_tensor * act = ggml_geglu_split(ctx, gate, up);
        ggml_tensor * experts_out = ggml_mul_mat_id(ctx, layer_cache.down_weights(), act, ids_tensor);
        experts_out = ggml_mul(ctx, experts_out, weights_tensor);

        sum_tensor = nullptr;
        for (int i = 0; i < expert_count; ++i) {
            ggml_tensor * expert_view = ggml_view_2d(ctx, experts_out, hidden_size, 1, experts_out->nb[2], std::size_t(i) * experts_out->nb[1]);
            sum_tensor = sum_tensor == nullptr ? expert_view : ggml_add(ctx, sum_tensor, expert_view);
        }

        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, sum_tensor);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (!ggml_backend_dev_supports_op(device, ggml_graph_node(graph, i))) {
                error = "GPU backend does not support a node in the Gemma4 Q4 selected expert graph";
                return false;
            }
        }

        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buffer == nullptr) {
            error = "failed to allocate GPU tensors for Gemma4 Q4 selected expert graph";
            return false;
        }
        return true;
    }

    bool compute(
            const float * hidden_data,
            const std::int32_t * local_ids,
            const float * local_weights,
            ggml_backend_t backend,
            float * output,
            std::string & error) {
        if (ctx == nullptr || buffer == nullptr || graph == nullptr || hidden_tensor == nullptr ||
                ids_tensor == nullptr || weights_tensor == nullptr || sum_tensor == nullptr) {
            error = "Gemma4 Q4 selected expert graph is not initialized";
            return false;
        }

        ggml_backend_tensor_set(hidden_tensor, hidden_data, 0, std::size_t(hidden_size) * sizeof(float));
        ggml_backend_tensor_set(ids_tensor, local_ids, 0, std::size_t(expert_count) * sizeof(std::int32_t));
        ggml_backend_tensor_set(weights_tensor, local_weights, 0, std::size_t(expert_count) * sizeof(float));

        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            error = "GPU backend failed computing Gemma4 Q4 selected expert graph";
            return false;
        }

        ggml_backend_tensor_get_async(backend, sum_tensor, output, 0, std::size_t(hidden_size) * sizeof(float));
        ggml_backend_synchronize(backend);
        return true;
    }

private:
    int layer_index = -1;
    int hidden_size = 0;
    int expert_count = 0;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * hidden_tensor = nullptr;
    ggml_tensor * ids_tensor = nullptr;
    ggml_tensor * weights_tensor = nullptr;
    ggml_tensor * sum_tensor = nullptr;
};

class llama_infinitum_gemma4_q4_gpu_layer_compute_graph_manager {
public:
    llama_infinitum_gemma4_q4_gpu_layer_compute_graph & get_or_create(
            llama_infinitum_gemma4_q4_gpu_layer_slot_cache & layer_cache,
            int layer_index,
            int hidden_size,
            int expert_count,
            ggml_backend_t backend,
            ggml_backend_dev_t device,
            std::string & error) {
        const std::uint64_t key =
            (std::uint64_t(std::uint32_t(layer_index)) << 32) |
            std::uint64_t(std::uint32_t(expert_count));
        auto found = graphs.find(key);
        if (found == graphs.end()) {
            auto graph = std::make_unique<llama_infinitum_gemma4_q4_gpu_layer_compute_graph>();
            if (!graph->init(layer_cache, layer_index, hidden_size, expert_count, backend, device, error)) {
                auto failed = std::move(graph);
                (void) failed;
                return invalid_graph;
            }
            auto inserted = graphs.emplace(key, std::move(graph));
            return *inserted.first->second;
        }
        return *found->second;
    }

private:
    std::unordered_map<std::uint64_t, std::unique_ptr<llama_infinitum_gemma4_q4_gpu_layer_compute_graph>> graphs;
    llama_infinitum_gemma4_q4_gpu_layer_compute_graph invalid_graph;
};

static bool llama_infinitum_gemma4_q4_gpu_compute_selected_experts_views(
        llama_infinitum_gemma4_q4_gpu_layer_slot_cache & layer_cache,
        const float * hidden_data,
        int hidden_size,
        const std::int32_t * local_ids,
        const float * local_weights,
        int expert_count,
        ggml_backend_t backend,
        ggml_backend_dev_t device,
        float * output,
        std::string & error) {
    if (hidden_data == nullptr || output == nullptr || local_ids == nullptr || local_weights == nullptr ||
            hidden_size <= 0 || expert_count <= 0) {
        error = "Gemma4 Q4 slot-view graph received invalid buffers";
        return false;
    }

    const int gate_up_rows = layer_cache.gate_up_row_count();
    const int down_rows = layer_cache.down_row_count();
    if (gate_up_rows != 2 * down_rows || down_rows <= 0) {
        error = "Gemma4 Q4 slot-view graph has invalid gate/up/down rows";
        return false;
    }

    constexpr std::size_t ctx_mem = 16ull * 1024ull * 1024ull;
    ggml_init_params params = {};
    params.mem_size = ctx_mem;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        error = "failed to initialize ggml context for Gemma4 Q4 slot-view graph";
        return false;
    }

    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 1);
    ggml_tensor * sum = nullptr;
    for (int i = 0; i < expert_count; ++i) {
        const int slot = local_ids[i];
        if (slot < 0) {
            error = "Gemma4 Q4 slot-view graph received negative slot id";
            ggml_free(ctx);
            return false;
        }

        ggml_tensor * gate_up_w_slot = layer_cache.gate_up_slot_tensor(slot);
        ggml_tensor * down_w_slot = layer_cache.down_slot_tensor(slot);
        if (gate_up_w_slot == nullptr || down_w_slot == nullptr) {
            error = "Gemma4 Q4 slot-view graph: per-slot tensor not allocated";
            ggml_free(ctx);
            return false;
        }

        ggml_tensor * gate_up = ggml_mul_mat(ctx, gate_up_w_slot, hidden);
        ggml_tensor * gate = ggml_view_2d(ctx, gate_up, down_rows, 1, gate_up->nb[1], 0);
        ggml_tensor * up = ggml_view_2d(ctx, gate_up, down_rows, 1, gate_up->nb[1], std::size_t(down_rows) * gate_up->nb[0]);
        ggml_tensor * act = ggml_geglu_split(ctx, gate, up);
        ggml_tensor * out = ggml_mul_mat(ctx, down_w_slot, act);
        ggml_tensor * weighted = ggml_scale(ctx, out, local_weights[i]);
        sum = sum == nullptr ? weighted : ggml_add(ctx, sum, weighted);
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sum);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!ggml_backend_dev_supports_op(device, ggml_graph_node(graph, i))) {
            error = "GPU backend does not support a node in the Gemma4 Q4 slot-view graph";
            ggml_free(ctx);
            return false;
        }
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        error = "failed to allocate GPU tensors for Gemma4 Q4 slot-view graph";
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(hidden, hidden_data, 0, std::size_t(hidden_size) * sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        error = "GPU backend failed computing Gemma4 Q4 slot-view graph";
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(sum, output, 0, std::size_t(hidden_size) * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

static int llama_infinitum_gemma4_direct_f16_workers_from_env(
        const char * name,
        int fallback,
        int max_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return std::max(1, std::min(fallback, max_value));
    }
    char * end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        parsed = fallback;
    }
    return std::max(1, std::min<int>(static_cast<int>(parsed), max_value));
}

static bool llama_infinitum_gemma4_direct_f16_fp16_accum_enabled() {
    static const bool enabled = llama_infinitum_env_enabled("LLAMA_INFINITUM_GEMMA4_DIRECT_F16_FP16_ACCUM");
    return enabled;
}

static int llama_infinitum_gemma4_direct_f16_fast_gelu_mode() {
    static const int mode = []() {
        const char * value = std::getenv("LLAMA_INFINITUM_GEMMA4_DIRECT_F16_FAST_GELU");
        if (value == nullptr || value[0] == '\0' || value[0] == '0') {
            return 0;
        }
        if (std::strcmp(value, "hard") == 0) {
            return 2;
        }
        if (std::strcmp(value, "linear") == 0) {
            return 3;
        }
        return 1;
    }();
    return mode;
}

static float llama_infinitum_gemma4_gelu_f32(float x) {
    const int fast_mode = llama_infinitum_gemma4_direct_f16_fast_gelu_mode();
    if (fast_mode == 3) {
        const float hard_gate = std::max(0.0f, std::min(1.0f, 0.5f + 0.25f * x));
        return x * hard_gate;
    }
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float gelu_coef = 0.044715f;
    const float x2 = x * x;
    const float y = sqrt_2_over_pi * x * (1.0f + gelu_coef * x2);
    if (fast_mode == 2) {
        const float hard_gate = std::max(0.0f, std::min(1.0f, 0.5f + 0.25f * y));
        return x * hard_gate;
    }
    if (fast_mode == 1) {
        const float clamped = std::max(-4.0f, std::min(4.0f, y));
        const float y2 = clamped * clamped;
        const float tanh_approx = clamped * (27.0f + y2) / (27.0f + 9.0f * y2);
        return 0.5f * x * (1.0f + std::max(-1.0f, std::min(1.0f, tanh_approx)));
    }
    return 0.5f * x * (1.0f + std::tanh(y));
}

static float llama_infinitum_gemma4_f16_dot(
        const ggml_fp16_t * row,
        const ggml_fp16_t * vector,
        int n) {
    float value = 0.0f;
#if defined(__ARM_NEON) && defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    const auto * x = reinterpret_cast<const __fp16 *>(row);
    const auto * y = reinterpret_cast<const __fp16 *>(vector);
    if (llama_infinitum_gemma4_direct_f16_fp16_accum_enabled()) {
        float16x8_t acc0 = vdupq_n_f16(0.0f);
        float16x8_t acc1 = vdupq_n_f16(0.0f);
        float16x8_t acc2 = vdupq_n_f16(0.0f);
        float16x8_t acc3 = vdupq_n_f16(0.0f);
        int i = 0;
        for (; i + 31 < n; i += 32) {
            const float16x8_t x0 = vld1q_f16(x + i + 0);
            const float16x8_t y0 = vld1q_f16(y + i + 0);
            const float16x8_t x1 = vld1q_f16(x + i + 8);
            const float16x8_t y1 = vld1q_f16(y + i + 8);
            const float16x8_t x2 = vld1q_f16(x + i + 16);
            const float16x8_t y2 = vld1q_f16(y + i + 16);
            const float16x8_t x3 = vld1q_f16(x + i + 24);
            const float16x8_t y3 = vld1q_f16(y + i + 24);
            acc0 = vfmaq_f16(acc0, x0, y0);
            acc1 = vfmaq_f16(acc1, x1, y1);
            acc2 = vfmaq_f16(acc2, x2, y2);
            acc3 = vfmaq_f16(acc3, x3, y3);
        }
        const float16x8_t acc16 = vaddq_f16(vaddq_f16(acc0, acc1), vaddq_f16(acc2, acc3));
        float32x4_t acc32 = vaddq_f32(
                vcvt_f32_f16(vget_low_f16(acc16)),
                vcvt_f32_f16(vget_high_f16(acc16)));
        value = vaddvq_f32(acc32);
        for (; i < n; ++i) {
            value += ggml_fp16_to_fp32(row[i]) * ggml_fp16_to_fp32(vector[i]);
        }
        return value;
    }
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 31 < n; i += 32) {
        const float16x8_t x0 = vld1q_f16(x + i + 0);
        const float16x8_t y0 = vld1q_f16(y + i + 0);
        const float16x8_t x1 = vld1q_f16(x + i + 8);
        const float16x8_t y1 = vld1q_f16(y + i + 8);
        const float16x8_t x2 = vld1q_f16(x + i + 16);
        const float16x8_t y2 = vld1q_f16(y + i + 16);
        const float16x8_t x3 = vld1q_f16(x + i + 24);
        const float16x8_t y3 = vld1q_f16(y + i + 24);
        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(x0)), vcvt_f32_f16(vget_low_f16(y0)));
        acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(x0)), vcvt_f32_f16(vget_high_f16(y0)));
        acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(x1)), vcvt_f32_f16(vget_low_f16(y1)));
        acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_high_f16(x1)), vcvt_f32_f16(vget_high_f16(y1)));
        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(x2)), vcvt_f32_f16(vget_low_f16(y2)));
        acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(x2)), vcvt_f32_f16(vget_high_f16(y2)));
        acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(x3)), vcvt_f32_f16(vget_low_f16(y3)));
        acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_high_f16(x3)), vcvt_f32_f16(vget_high_f16(y3)));
    }
    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    value = vaddvq_f32(acc0);
    for (; i < n; ++i) {
        value += ggml_fp16_to_fp32(row[i]) * ggml_fp16_to_fp32(vector[i]);
    }
#else
    for (int i = 0; i < n; ++i) {
        value += ggml_fp16_to_fp32(row[i]) * ggml_fp16_to_fp32(vector[i]);
    }
#endif
    return value;
}

static const char * llama_infinitum_gemma4_f16_dot_backend_name() {
#if defined(__ARM_NEON) && defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return llama_infinitum_gemma4_direct_f16_fp16_accum_enabled() ? "neon-fp16-fp16acc" : "neon-fp16-f32acc";
#else
    return "scalar-f32acc";
#endif
}

static inline void llama_infinitum_gemma4_prefetch_read(const void * ptr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 3);
#else
    (void) ptr;
#endif
}

static bool llama_infinitum_gemma4_direct_f16_parallel_retry_enabled() {
    return llama_infinitum_env_enabled("LLAMA_INFINITUM_GEMMA4_DIRECT_F16_PARALLEL_RETRY");
}

static bool llama_infinitum_gemma4_direct_f16_madvise_enabled() {
    static const bool enabled = llama_infinitum_env_enabled("LLAMA_INFINITUM_GEMMA4_DIRECT_F16_MADVISE");
    return enabled;
}

static void llama_infinitum_gemma4_direct_f16_madvise(const void * ptr, std::size_t len) {
#if defined(__linux__) && defined(MADV_WILLNEED)
    if (!llama_infinitum_gemma4_direct_f16_madvise_enabled() || ptr == nullptr || len == 0) {
        return;
    }
    static const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return;
    }
    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(ptr);
    const std::uintptr_t page = addr & ~std::uintptr_t(page_size - 1);
    const std::size_t adjust = static_cast<std::size_t>(addr - page);
    (void) madvise(reinterpret_cast<void *>(page), len + adjust, MADV_WILLNEED);
#else
    (void) ptr;
    (void) len;
#endif
}

static std::uint64_t llama_infinitum_gemma4_direct_f16_cache_bytes_from_env() {
    static const std::uint64_t bytes = []() {
        const char * value = std::getenv("LLAMA_INFINITUM_GEMMA4_DIRECT_F16_CACHE_MB");
        std::uint64_t mb = 0;
        if (value != nullptr && value[0] != '\0') {
            for (const char * p = value; *p >= '0' && *p <= '9'; ++p) {
                mb = mb * 10 + static_cast<std::uint64_t>(*p - '0');
            }
        }
        return mb * 1024ull * 1024ull;
    }();
    return bytes;
}

struct llama_infinitum_gemma4_direct_f16_cache_entry {
    int layer_index = -1;
    int expert_id = -1;
    std::size_t gate_up_bytes = 0;
    std::vector<std::uint8_t> bytes;

    const ggml_fp16_t * gate_up() const {
        return reinterpret_cast<const ggml_fp16_t *>(bytes.data());
    }

    const ggml_fp16_t * down() const {
        return reinterpret_cast<const ggml_fp16_t *>(bytes.data() + gate_up_bytes);
    }
};

class llama_infinitum_gemma4_direct_f16_cache {
public:
    std::shared_ptr<const llama_infinitum_gemma4_direct_f16_cache_entry> get_or_load(
            int layer_index,
            int expert_id,
            const std::uint8_t * pack_base,
            std::size_t gate_up_offset,
            std::size_t down_offset,
            std::size_t gate_up_bytes,
            std::size_t down_bytes) {
        max_bytes = llama_infinitum_gemma4_direct_f16_cache_bytes_from_env();
        const std::size_t entry_bytes = gate_up_bytes + down_bytes;
        if (max_bytes == 0 || pack_base == nullptr || entry_bytes == 0 || entry_bytes > max_bytes) {
            return nullptr;
        }

        const std::uint64_t key = key_for(layer_index, expert_id);
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto found = items.find(key);
            if (found != items.end()) {
                ++cache_hits;
                touch_locked(key);
                return found->second;
            }
            ++cache_misses;
        }

        auto entry = std::make_shared<llama_infinitum_gemma4_direct_f16_cache_entry>();
        entry->layer_index = layer_index;
        entry->expert_id = expert_id;
        entry->gate_up_bytes = gate_up_bytes;
        entry->bytes.resize(entry_bytes);
        std::memcpy(entry->bytes.data(), pack_base + gate_up_offset, gate_up_bytes);
        std::memcpy(entry->bytes.data() + gate_up_bytes, pack_base + down_offset, down_bytes);

        std::lock_guard<std::mutex> lock(mutex);
        auto found = items.find(key);
        if (found != items.end()) {
            ++cache_hits;
            touch_locked(key);
            return found->second;
        }
        used_bytes += std::uint64_t(entry_bytes);
        items[key] = entry;
        order.push_back(key);
        evict_if_needed_locked(key);
        return entry;
    }

    std::uint64_t hits() const {
        std::lock_guard<std::mutex> lock(mutex);
        return cache_hits;
    }

    std::uint64_t misses() const {
        std::lock_guard<std::mutex> lock(mutex);
        return cache_misses;
    }

    std::uint64_t evictions() const {
        std::lock_guard<std::mutex> lock(mutex);
        return cache_evictions;
    }

    std::uint64_t resident_bytes() const {
        std::lock_guard<std::mutex> lock(mutex);
        return used_bytes;
    }

private:
    mutable std::mutex mutex;
    std::uint64_t max_bytes = 0;
    std::uint64_t used_bytes = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t cache_misses = 0;
    std::uint64_t cache_evictions = 0;
    std::unordered_map<std::uint64_t, std::shared_ptr<const llama_infinitum_gemma4_direct_f16_cache_entry>> items;
    std::list<std::uint64_t> order;

    static std::uint64_t key_for(int layer_index, int expert_id) {
        return (std::uint64_t(std::uint32_t(layer_index)) << 32) | std::uint32_t(expert_id);
    }

    void touch_locked(std::uint64_t key) {
        order.remove(key);
        order.push_back(key);
    }

    void evict_if_needed_locked(std::uint64_t protected_key) {
        while (used_bytes > max_bytes && order.size() > 1) {
            const std::uint64_t victim = order.front();
            order.pop_front();
            if (victim == protected_key) {
                order.push_back(victim);
                continue;
            }
            auto found = items.find(victim);
            if (found == items.end()) {
                continue;
            }
            used_bytes -= static_cast<std::uint64_t>(found->second->bytes.size());
            items.erase(found);
            ++cache_evictions;
        }
    }
};

static llama_infinitum_gemma4_direct_f16_cache & llama_infinitum_gemma4_direct_f16_cache_get() {
    static llama_infinitum_gemma4_direct_f16_cache cache;
    return cache;
}

// ---------------------------------------------------------------------------
// Q8_0 expert cache — avoids mmap page faults from UFS flash (~500 MB/s)
// by keeping frequently-used experts in RAM.
// Default 2 GB budget fits all 128 experts of one layer + hot spares.
// ---------------------------------------------------------------------------

static std::uint64_t gemma4_direct_q8_cache_bytes_from_env();

struct gemma4_direct_q8_cache_entry {
    int layer_index = -1;
    int expert_id = -1;
    std::size_t gate_up_bytes = 0;
    std::vector<std::uint8_t> bytes;

    const void * gate_up() const { return bytes.data(); }
    const void * down() const { return bytes.data() + gate_up_bytes; }
};

class gemma4_direct_q8_cache {
public:
    std::shared_ptr<const gemma4_direct_q8_cache_entry> get_or_load(
            int layer_index,
            int expert_id,
            const std::uint8_t * pack_base,
            std::size_t gate_up_offset,
            std::size_t down_offset,
            std::size_t gate_up_bytes,
            std::size_t down_bytes) {
        max_bytes = gemma4_direct_q8_cache_bytes_from_env();
        const std::size_t entry_bytes = gate_up_bytes + down_bytes;
        if (max_bytes == 0 || pack_base == nullptr || entry_bytes == 0) {
            return nullptr;
        }

        const std::uint64_t key = key_for(layer_index, expert_id);
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto found = items.find(key);
            if (found != items.end()) {
                ++cache_hits;
                touch_locked(key);
                return found->second;
            }
            ++cache_misses;
        }

        if (entry_bytes > max_bytes) {
            return nullptr;
        }

        auto entry = std::make_shared<gemma4_direct_q8_cache_entry>();
        entry->layer_index = layer_index;
        entry->expert_id = expert_id;
        entry->gate_up_bytes = gate_up_bytes;
        entry->bytes.resize(entry_bytes);

#if defined(__linux__) && defined(MADV_WILLNEED)
        madvise(const_cast<std::uint8_t *>(pack_base + gate_up_offset), gate_up_bytes, MADV_WILLNEED);
        madvise(const_cast<std::uint8_t *>(pack_base + down_offset), down_bytes, MADV_WILLNEED);
#endif
        std::memcpy(entry->bytes.data(), pack_base + gate_up_offset, gate_up_bytes);
        std::memcpy(entry->bytes.data() + gate_up_bytes, pack_base + down_offset, down_bytes);

        std::lock_guard<std::mutex> lock(mutex);
        auto found = items.find(key);
        if (found != items.end()) {
            ++cache_hits;
            touch_locked(key);
            return found->second;
        }
        used_bytes += std::uint64_t(entry_bytes);
        items[key] = entry;
        order.push_back(key);
        evict_if_needed_locked(key);
        return entry;
    }

    std::uint64_t hits() const     { std::lock_guard<std::mutex> lock(mutex); return cache_hits; }
    std::uint64_t misses() const   { std::lock_guard<std::mutex> lock(mutex); return cache_misses; }
    std::uint64_t evictions() const { std::lock_guard<std::mutex> lock(mutex); return cache_evictions; }
    std::uint64_t resident_bytes() const { std::lock_guard<std::mutex> lock(mutex); return used_bytes; }

private:
    mutable std::mutex mutex;
    std::uint64_t max_bytes = 0;
    std::uint64_t used_bytes = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t cache_misses = 0;
    std::uint64_t cache_evictions = 0;
    std::unordered_map<std::uint64_t, std::shared_ptr<const gemma4_direct_q8_cache_entry>> items;
    std::list<std::uint64_t> order;

    static std::uint64_t key_for(int layer_index, int expert_id) {
        return (std::uint64_t(std::uint32_t(layer_index)) << 32) | std::uint32_t(expert_id);
    }

    void touch_locked(std::uint64_t key) {
        order.remove(key);
        order.push_back(key);
    }

    void evict_if_needed_locked(std::uint64_t protected_key) {
        while (used_bytes > max_bytes && !order.empty()) {
            auto it = order.begin();
            if (*it == protected_key && order.size() <= 1) break;
            if (*it == protected_key) {
                std::uint64_t skipped = *it;
                order.erase(it);
                order.push_back(skipped);
                it = order.begin();
            }
            auto found = items.find(*it);
            if (found == items.end()) {
                order.erase(it);
                continue;
            }
            used_bytes -= static_cast<std::uint64_t>(found->second->bytes.size());
            items.erase(found);
            order.erase(it);
            ++cache_evictions;
        }
    }
};

static std::uint64_t gemma4_direct_q8_cache_bytes_from_env() {
    static const std::uint64_t bytes = []() {
        const char * value = std::getenv("LLAMA_INFINITUM_GEMMA4_DIRECT_Q8_CACHE_MB");
        std::uint64_t mb = 0;
        if (value != nullptr && value[0] != '\0') {
            for (const char * p = value; *p >= '0' && *p <= '9'; ++p)
                mb = mb * 10 + static_cast<std::uint64_t>(*p - '0');
        }
        if (mb == 0) mb = 2048; // default 2 GB
        return mb * 1024ull * 1024ull;
    }();
    return bytes;
}

static gemma4_direct_q8_cache & gemma4_direct_q8_cache_get() {
    static gemma4_direct_q8_cache cache;
    return cache;
}

// Async prefetch: submit layer N+1 experts to background thread while computing layer N
struct gemma4_q8_prefetch_request {
    int layer_index;
    std::vector<int> expert_ids;
    const std::uint8_t * pack_base;
    std::size_t layer_base;
    std::size_t gate_up_source_bytes;
    std::size_t gate_up_layer_bytes;
    std::size_t down_source_bytes;
};

static void gemma4_q8_prefetch_worker() {
    // best-effort: drain requests and warm cache
}

static bool llama_infinitum_gemma4_f16_compute_one_expert(
        const ggml_fp16_t * gate_up,
        const ggml_fp16_t * down,
        int hidden_size,
        int down_rows,
        const ggml_fp16_t * hidden_f16,
        float weight,
        float * output,
        bool collect_profile,
        llama_infinitum_moe_compute_profile * profile,
        int row_threads,
        int * first_bad_row = nullptr,
        int * first_bad_stage = nullptr,
        float * first_bad_value = nullptr) {
    if (gate_up == nullptr || down == nullptr || hidden_f16 == nullptr || output == nullptr ||
            hidden_size <= 0 || down_rows <= 0) {
        return false;
    }

    thread_local std::vector<float> activation_scratch;
    thread_local std::vector<ggml_fp16_t> activation_f16_scratch;
    activation_scratch.resize(std::size_t(down_rows));
    activation_f16_scratch.resize(std::size_t(down_rows));
    float * activation = activation_scratch.data();
    ggml_fp16_t * activation_f16 = activation_f16_scratch.data();

    if (row_threads <= 1) {
        const auto gate_start = collect_profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        for (int row = 0; row < down_rows; ++row) {
            constexpr int prefetch_rows = 4;
            if (row + prefetch_rows < down_rows) {
                llama_infinitum_gemma4_prefetch_read(
                        gate_up + std::size_t(row + prefetch_rows) * std::size_t(hidden_size));
                llama_infinitum_gemma4_prefetch_read(
                        gate_up + std::size_t(row + prefetch_rows + down_rows) * std::size_t(hidden_size));
            }
            const ggml_fp16_t * gate_row = gate_up + std::size_t(row) * std::size_t(hidden_size);
            const ggml_fp16_t * up_row = gate_up + std::size_t(row + down_rows) * std::size_t(hidden_size);
            const float gate = llama_infinitum_gemma4_f16_dot(gate_row, hidden_f16, hidden_size);
            const float up = llama_infinitum_gemma4_f16_dot(up_row, hidden_f16, hidden_size);
            activation[row] = llama_infinitum_gemma4_gelu_f32(gate) * up;
            if (!std::isfinite(activation[row]) && first_bad_row != nullptr && *first_bad_row < 0) {
                *first_bad_row = row;
                if (first_bad_stage != nullptr) {
                    *first_bad_stage = 1;
                }
                if (first_bad_value != nullptr) {
                    *first_bad_value = activation[row];
                }
            }
        }
        const auto activation_start = collect_profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        ggml_fp32_to_fp16_row(activation, activation_f16, down_rows);
        const auto down_start = collect_profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

        bool finite = true;
        for (int row = 0; row < hidden_size; ++row) {
            constexpr int prefetch_rows = 4;
            if (row + prefetch_rows < hidden_size) {
                llama_infinitum_gemma4_prefetch_read(
                        down + std::size_t(row + prefetch_rows) * std::size_t(down_rows));
            }
            const ggml_fp16_t * down_row = down + std::size_t(row) * std::size_t(down_rows);
            const float value = llama_infinitum_gemma4_f16_dot(down_row, activation_f16, down_rows) * weight;
            if (!std::isfinite(value)) {
                finite = false;
                if (first_bad_row != nullptr && *first_bad_row < 0) {
                    *first_bad_row = row;
                    if (first_bad_stage != nullptr) {
                        *first_bad_stage = 2;
                    }
                    if (first_bad_value != nullptr) {
                        *first_bad_value = value;
                    }
                }
            }
            output[row] = value;
        }

        if (collect_profile && profile != nullptr) {
            const auto end = std::chrono::steady_clock::now();
            profile->gate_up_ms += llama_infinitum_elapsed_ms(gate_start, activation_start);
            profile->activation_ms += llama_infinitum_elapsed_ms(activation_start, down_start);
            profile->down_ms += llama_infinitum_elapsed_ms(down_start, end);
        }
        return finite;
    }

    auto run_rows = [&](int row_count, auto && fn) {
        const int worker_count = std::max(1, std::min(row_threads, row_count));
        if (worker_count <= 1 || row_count <= 1) {
            for (int row = 0; row < row_count; ++row) {
                fn(row);
            }
            return;
        }
        auto worker_fn = [&](int worker) {
            const int begin = (row_count * worker) / worker_count;
            const int end = (row_count * (worker + 1)) / worker_count;
            for (int row = begin; row < end; ++row) {
                fn(row);
            }
        };
        std::vector<std::thread> workers;
        workers.reserve(std::size_t(worker_count - 1));
        for (int worker = 1; worker < worker_count; ++worker) {
            workers.emplace_back(worker_fn, worker);
        }
        worker_fn(0);
        for (std::thread & worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    };

    const auto gate_start = collect_profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    run_rows(down_rows, [&](int row) {
        constexpr int prefetch_rows = 4;
        if (row + prefetch_rows < down_rows) {
            llama_infinitum_gemma4_prefetch_read(
                    gate_up + std::size_t(row + prefetch_rows) * std::size_t(hidden_size));
            llama_infinitum_gemma4_prefetch_read(
                    gate_up + std::size_t(row + prefetch_rows + down_rows) * std::size_t(hidden_size));
        }
        const ggml_fp16_t * gate_row = gate_up + std::size_t(row) * std::size_t(hidden_size);
        const ggml_fp16_t * up_row = gate_up + std::size_t(row + down_rows) * std::size_t(hidden_size);
        const float gate = llama_infinitum_gemma4_f16_dot(gate_row, hidden_f16, hidden_size);
        const float up = llama_infinitum_gemma4_f16_dot(up_row, hidden_f16, hidden_size);
        activation[row] = llama_infinitum_gemma4_gelu_f32(gate) * up;
        if (row_threads <= 1 && !std::isfinite(activation[row]) && first_bad_row != nullptr && *first_bad_row < 0) {
            *first_bad_row = row;
            if (first_bad_stage != nullptr) {
                *first_bad_stage = 1;
            }
            if (first_bad_value != nullptr) {
                *first_bad_value = activation[row];
            }
        }
    });
    const auto activation_start = collect_profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ggml_fp32_to_fp16_row(activation, activation_f16, down_rows);
    const auto down_start = collect_profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    std::atomic<int> finite{1};
    run_rows(hidden_size, [&](int row) {
        constexpr int prefetch_rows = 4;
        if (row + prefetch_rows < hidden_size) {
            llama_infinitum_gemma4_prefetch_read(
                    down + std::size_t(row + prefetch_rows) * std::size_t(down_rows));
        }
        const ggml_fp16_t * down_row = down + std::size_t(row) * std::size_t(down_rows);
        const float value = llama_infinitum_gemma4_f16_dot(down_row, activation_f16, down_rows) * weight;
        if (!std::isfinite(value)) {
            finite.store(0, std::memory_order_relaxed);
        }
        if (row_threads <= 1 && !std::isfinite(value) && first_bad_row != nullptr && *first_bad_row < 0) {
            *first_bad_row = row;
            if (first_bad_stage != nullptr) {
                *first_bad_stage = 2;
            }
            if (first_bad_value != nullptr) {
                *first_bad_value = value;
            }
        }
        output[row] = value;
    });

    if (collect_profile && profile != nullptr) {
        const auto end = std::chrono::steady_clock::now();
        profile->gate_up_ms += llama_infinitum_elapsed_ms(gate_start, activation_start);
        profile->activation_ms += llama_infinitum_elapsed_ms(activation_start, down_start);
        profile->down_ms += llama_infinitum_elapsed_ms(down_start, end);
    }
    return finite.load(std::memory_order_relaxed) != 0;
}

// ---------------------------------------------------------------------------
// Q8_0 / Q4_0 direct CPU path — fused dequant + dot with NEON
// Bypasses ggml graph compute for expert matmuls.
// ---------------------------------------------------------------------------

static bool llama_infinitum_gemma4_direct_q8_enabled() {
    static const bool enabled = llama_infinitum_env_enabled("LLAMA_INFINITUM_GEMMA4_DIRECT_Q8");
    return enabled;
}

static bool llama_infinitum_gemma4_direct_q8_fast_gelu_enabled() {
    static const bool enabled = llama_infinitum_env_enabled("LLAMA_INFINITUM_GEMMA4_DIRECT_Q8_FAST_GELU");
    return enabled;
}

// Block definitions matching ggml-common.h
struct gemma4_block_q8_0 { uint16_t d; int8_t qs[32]; };
struct gemma4_block_q4_0 { uint16_t d; uint8_t qs[16]; };

static float gemma4_fp16_to_f32(uint16_t h) {
#if defined(__ARM_NEON) && defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    __fp16 f16;
    std::memcpy(&f16, &h, 2);
    return static_cast<float>(f16);
#else
    const uint32_t e = (h & 0x7C00) >> 10;
    const uint32_t m = (h & 0x03FF) << 13;
    const uint32_t v = (h & 0x8000) << 16;
    if (e == 0) {
        float r;
        uint32_t bits = v | m;
        if (bits == 0) { r = 0.0f; }
        else { std::memcpy(&r, &bits, 4); r = std::ldexp(r, -24); }
        return r;
    }
    uint32_t r = v | ((e + 112) << 23) | m;
    float f;
    std::memcpy(&f, &r, 4);
    return f;
#endif
}

// NEON-optimised Q8_0 dot product: sum(dequant(block) * vec)
static float gemma4_q8_dot(const void * packed, const float * vec, int n) {
    const auto * blocks = static_cast<const gemma4_block_q8_0 *>(packed);
    float sum = 0.0f;
    int nb = n / 32;

#if defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);
    for (int i = 0; i < nb; i++) {
        const float d = gemma4_fp16_to_f32(blocks[i].d);
        const float32x4_t dv = vdupq_n_f32(d);
        const float * vp = vec + i * 32;
        for (int j = 0; j < 32; j += 8) {
            // Load 8 int8 → sign-extend to int16 → widen to int32 → convert to f32
            int8x8_t qs8 = vld1_s8(blocks[i].qs + j);
            int16x8_t qs16 = vmovl_s8(qs8);
            int32x4_t qi0 = vmovl_s16(vget_low_s16(qs16));
            int32x4_t qi1 = vmovl_s16(vget_high_s16(qs16));
            float32x4_t qf0 = vcvtq_f32_s32(qi0);
            float32x4_t qf1 = vcvtq_f32_s32(qi1);
            float32x4_t vf0 = vld1q_f32(vp + j);
            float32x4_t vf1 = vld1q_f32(vp + j + 4);
            sumv0 = vfmaq_f32(sumv0, vmulq_f32(qf0, dv), vf0);
            sumv1 = vfmaq_f32(sumv1, vmulq_f32(qf1, dv), vf1);
        }
    }
    float32x4_t sumv = vaddq_f32(sumv0, sumv1);
    sum = vaddvq_f32(sumv);
#else
    for (int i = 0; i < nb; i++) {
        float d = gemma4_fp16_to_f32(blocks[i].d);
        for (int j = 0; j < 32; j++)
            sum += blocks[i].qs[j] * d * vec[i * 32 + j];
    }
#endif

    // Remainder
    int base = nb * 32;
    for (int i = base; i < n; i++) {
        int bi = i / 32, bj = i % 32;
        float d = gemma4_fp16_to_f32(blocks[bi].d);
        sum += blocks[bi].qs[bj] * d * vec[i];
    }
    return sum;
}

// NEON-optimised Q4_0 dot product
static float gemma4_q4_dot(const void * packed, const float * vec, int n) {
    const auto * blocks = static_cast<const gemma4_block_q4_0 *>(packed);
    float sum = 0.0f;
    int nb = n / 32;

#if defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);
    const int8x16_t offset = vdupq_n_s8(-8);
    for (int i = 0; i < nb; i++) {
        const float d = gemma4_fp16_to_f32(blocks[i].d);
        const float32x4_t dv = vdupq_n_f32(d);
        const float * vp = vec + i * 32;
        // Load 16 nibbles, deinterleave low/high
        uint8x16_t qs_raw = vld1q_u8(blocks[i].qs);
        // low nibbles [0..15]
        uint8x16_t lo_u8 = vandq_u8(qs_raw, vdupq_n_u8(0x0F));
        // high nibbles [16..31]
        uint8x16_t hi_u8 = vshrq_n_u8(qs_raw, 4);
        // sub 8 offset
        int8x16_t lo_s8 = vsubq_s8(vreinterpretq_s8_u8(lo_u8), offset);
        int8x16_t hi_s8 = vsubq_s8(vreinterpretq_s8_u8(hi_u8), offset);
        // widen to int16 -> int32 -> f32
        int16x8_t lo16_0 = vmovl_s8(vget_low_s8(lo_s8));
        int16x8_t lo16_1 = vmovl_s8(vget_high_s8(lo_s8));
        int16x8_t hi16_0 = vmovl_s8(vget_low_s8(hi_s8));
        int16x8_t hi16_1 = vmovl_s8(vget_high_s8(hi_s8));
        int32x4_t lo32_00 = vmovl_s16(vget_low_s16(lo16_0));
        int32x4_t lo32_01 = vmovl_s16(vget_high_s16(lo16_0));
        int32x4_t lo32_10 = vmovl_s16(vget_low_s16(lo16_1));
        int32x4_t lo32_11 = vmovl_s16(vget_high_s16(lo16_1));
        int32x4_t hi32_00 = vmovl_s16(vget_low_s16(hi16_0));
        int32x4_t hi32_01 = vmovl_s16(vget_high_s16(hi16_0));
        int32x4_t hi32_10 = vmovl_s16(vget_low_s16(hi16_1));
        int32x4_t hi32_11 = vmovl_s16(vget_high_s16(hi16_1));
        // f32
        float32x4_t qf[8] = {
            vcvtq_f32_s32(lo32_00), vcvtq_f32_s32(lo32_01),
            vcvtq_f32_s32(lo32_10), vcvtq_f32_s32(lo32_11),
            vcvtq_f32_s32(hi32_00), vcvtq_f32_s32(hi32_01),
            vcvtq_f32_s32(hi32_10), vcvtq_f32_s32(hi32_11),
        };
        for (int j = 0; j < 8; j++) {
            float32x4_t vf = vld1q_f32(vp + j * 4);
            float32x4_t qd = vmulq_f32(qf[j], dv);
            if (j < 4) sumv0 = vfmaq_f32(sumv0, qd, vf);
            else       sumv1 = vfmaq_f32(sumv1, qd, vf);
        }
    }
    float32x4_t sumv = vaddq_f32(sumv0, sumv1);
    sum = vaddvq_f32(sumv);
#else
    for (int i = 0; i < nb; i++) {
        float d = gemma4_fp16_to_f32(blocks[i].d);
        for (int j = 0; j < 16; j++) {
            sum += ((blocks[i].qs[j] & 0x0F) - 8) * d * vec[i * 32 + j];
            sum += ((blocks[i].qs[j] >> 4)   - 8) * d * vec[i * 32 + j + 16];
        }
    }
#endif

    int base = nb * 32;
    for (int i = base; i < n; i++) {
        int bi = i / 32, bj = i % 32;
        float d = gemma4_fp16_to_f32(blocks[bi].d);
        int val = (bj >= 16)
            ? ((blocks[bi].qs[bj - 16] >> 4) - 8)
            : ((blocks[bi].qs[bj] & 0x0F) - 8);
        sum += val * d * vec[i];
    }
    return sum;
}

// Compute one expert using Q8_0 weights — fused dequant+dot, no intermediate buffers
static bool gemma4_q8_compute_one_expert(
        const void * gate_up_packed,     // Q8_0 packed
        const void * down_packed,        // Q8_0 packed
        int hidden_size,
        int down_rows,
        const float * hidden_f32,
        float weight,
        float * output,
        int row_threads,
        bool fast_gelu) {

    thread_local std::vector<float> activation_scratch;
    activation_scratch.resize(std::size_t(down_rows));
    float * activation = activation_scratch.data();

    if (row_threads <= 1) {
        // Gate+Up matmul: [down_rows*2, hidden_size] x [hidden_size] -> [down_rows*2]
        // gate in [0..down_rows), up in [down_rows..2*down_rows)
        constexpr int blk_bytes_q8 = 34; // 2B fp16 d + 32B int8
        for (int row = 0; row < down_rows; row++) {
            const void * gate_row = static_cast<const uint8_t *>(gate_up_packed) + std::size_t(row) * hidden_size / 32 * blk_bytes_q8;
            const void * up_row   = static_cast<const uint8_t *>(gate_up_packed) + std::size_t(row + down_rows) * hidden_size / 32 * blk_bytes_q8;
            float gate = gemma4_q8_dot(gate_row, hidden_f32, hidden_size);
            float up   = gemma4_q8_dot(up_row,   hidden_f32, hidden_size);
            if (fast_gelu) {
                float hard_gate = std::max(0.0f, std::min(1.0f, 0.5f + 0.25f * gate));
                activation[row] = gate * hard_gate * up;
            } else {
                activation[row] = llama_infinitum_gemma4_gelu_f32(gate) * up;
            }
        }

        // Down matmul: [hidden_size, down_rows] x [down_rows] -> [hidden_size]
        for (int row = 0; row < hidden_size; row++) {
            const void * down_row = static_cast<const uint8_t *>(down_packed) + std::size_t(row) * down_rows / 32 * blk_bytes_q8;
            output[row] = gemma4_q8_dot(down_row, activation, down_rows) * weight;
        }
    } else {
        // Multithreaded rows
        auto run_rows = [&](int row_count, auto && fn) {
            const int workers = std::max(1, std::min(row_threads, row_count));
            if (workers <= 1 || row_count <= 1) {
                for (int r = 0; r < row_count; r++) fn(r);
                return;
            }
            auto worker_fn = [&](int w) {
                int begin = (row_count * w) / workers;
                int end   = (row_count * (w + 1)) / workers;
                for (int r = begin; r < end; r++) fn(r);
            };
            std::vector<std::thread> wthreads;
            wthreads.reserve(std::size_t(workers - 1));
            for (int w = 1; w < workers; w++) wthreads.emplace_back(worker_fn, w);
            worker_fn(0);
            for (auto & t : wthreads) if (t.joinable()) t.join();
        };

        // Gate+Up matmul — parallel over rows
        constexpr int blk_bytes_q8 = 34;
        run_rows(down_rows, [&](int row) {
            const void * gate_row = static_cast<const uint8_t *>(gate_up_packed) + std::size_t(row) * hidden_size / 32 * blk_bytes_q8;
            const void * up_row   = static_cast<const uint8_t *>(gate_up_packed) + std::size_t(row + down_rows) * hidden_size / 32 * blk_bytes_q8;
            float gate = gemma4_q8_dot(gate_row, hidden_f32, hidden_size);
            float up   = gemma4_q8_dot(up_row,   hidden_f32, hidden_size);
            if (fast_gelu) {
                float hard_gate = std::max(0.0f, std::min(1.0f, 0.5f + 0.25f * gate));
                activation[row] = gate * hard_gate * up;
            } else {
                activation[row] = llama_infinitum_gemma4_gelu_f32(gate) * up;
            }
        });

        // Down matmul — parallel over hidden_size rows
        run_rows(hidden_size, [&](int row) {
            const void * down_row = static_cast<const uint8_t *>(down_packed) + std::size_t(row) * down_rows / 32 * blk_bytes_q8;
            output[row] = gemma4_q8_dot(down_row, activation, down_rows) * weight;
        });
    }
    return true;
}

// Compute one expert using Q4_0 weights
static bool gemma4_q4_compute_one_expert(
        const void * gate_up_packed,
        const void * down_packed,
        int hidden_size,
        int down_rows,
        const float * hidden_f32,
        float weight,
        float * output,
        int row_threads,
        bool fast_gelu) {

    thread_local std::vector<float> activation_scratch;
    activation_scratch.resize(std::size_t(down_rows));
    float * activation = activation_scratch.data();

    constexpr int blk_bytes_q4 = 18; // 2B fp16 d + 16B nibbles

    if (row_threads <= 1) {
        for (int row = 0; row < down_rows; row++) {
            const void * gate_row = static_cast<const uint8_t *>(gate_up_packed) + std::size_t(row) * hidden_size / 32 * blk_bytes_q4;
            const void * up_row   = static_cast<const uint8_t *>(gate_up_packed) + std::size_t(row + down_rows) * hidden_size / 32 * blk_bytes_q4;
            float gate = gemma4_q4_dot(gate_row, hidden_f32, hidden_size);
            float up   = gemma4_q4_dot(up_row,   hidden_f32, hidden_size);
            if (fast_gelu) {
                float hard_gate = std::max(0.0f, std::min(1.0f, 0.5f + 0.25f * gate));
                activation[row] = gate * hard_gate * up;
            } else {
                activation[row] = llama_infinitum_gemma4_gelu_f32(gate) * up;
            }
        }
        for (int row = 0; row < hidden_size; row++) {
            const void * down_row = static_cast<const uint8_t *>(down_packed) + std::size_t(row) * down_rows / 32 * blk_bytes_q4;
            output[row] = gemma4_q4_dot(down_row, activation, down_rows) * weight;
        }
    } else {
        auto run_rows = [&](int row_count, auto && fn) {
            const int workers = std::max(1, std::min(row_threads, row_count));
            if (workers <= 1 || row_count <= 1) {
                for (int r = 0; r < row_count; r++) fn(r);
                return;
            }
            auto worker_fn = [&](int w) {
                int begin = (row_count * w) / workers;
                int end   = (row_count * (w + 1)) / workers;
                for (int r = begin; r < end; r++) fn(r);
            };
            std::vector<std::thread> wthreads;
            wthreads.reserve(std::size_t(workers - 1));
            for (int w = 1; w < workers; w++) wthreads.emplace_back(worker_fn, w);
            worker_fn(0);
            for (auto & t : wthreads) if (t.joinable()) t.join();
        };

        run_rows(down_rows, [&](int row) {
            const void * gate_row = static_cast<const uint8_t *>(gate_up_packed) + std::size_t(row) * hidden_size / 32 * blk_bytes_q4;
            const void * up_row   = static_cast<const uint8_t *>(gate_up_packed) + std::size_t(row + down_rows) * hidden_size / 32 * blk_bytes_q4;
            float gate = gemma4_q4_dot(gate_row, hidden_f32, hidden_size);
            float up   = gemma4_q4_dot(up_row,   hidden_f32, hidden_size);
            activation[row] = fast_gelu
                ? gate * std::max(0.0f, std::min(1.0f, 0.5f + 0.25f * gate)) * up
                : llama_infinitum_gemma4_gelu_f32(gate) * up;
        });
        run_rows(hidden_size, [&](int row) {
            const void * down_row = static_cast<const uint8_t *>(down_packed) + std::size_t(row) * down_rows / 32 * blk_bytes_q4;
            output[row] = gemma4_q4_dot(down_row, activation, down_rows) * weight;
        });
    }
    return true;
}

static int gemma4_direct_pack_workers_from_env(const char * name, int fallback, int max_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
        return std::max(1, std::min(fallback, max_value));
    char * end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) parsed = fallback;
    return std::max(1, std::min<int>(static_cast<int>(parsed), max_value));
}

// Execute selected Q8_0 experts directly — bypasses ggml, uses NEON fused dequant+dot.
// Mirrors the F16 direct path structure but with Q8_0 weight format and F32 hidden state.
llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_gemma4_q8_pack_selected_experts_into(
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        const std::vector<int> & expert_ids,
        const std::vector<float> & expert_weights,
        const float * hidden,
        int hidden_size,
        float * output) {
    llama_infinitum_moe_expert_mlp_result result;
    result.layer_index = layer_index;
    result.expert_id = -1;
    result.backend = "cpu-gemma4-direct-q8-neon";

    if (!llama_infinitum_gemma4_direct_q8_enabled()) {
        result.error = "Gemma4 direct Q8 path disabled (set LLAMA_INFINITUM_GEMMA4_DIRECT_Q8=1)";
        return result;
    }
    if (hidden == nullptr || output == nullptr || hidden_size <= 0) {
        result.error = "Gemma4 direct Q8 path received invalid buffers";
        return result;
    }
    if (info.hidden_size > 0 && info.hidden_size != hidden_size) {
        result.error = "Gemma4 direct Q8 hidden size mismatch";
        return result;
    }
    if (info.gate_up_rows <= 0 || info.down_rows <= 0 || info.gate_up_rows != info.down_rows * 2) {
        result.error = "Gemma4 direct Q8 expert index shape invalid";
        return result;
    }
    if (expert_ids.empty()) {
        result.error = "Gemma4 direct Q8 selected expert list empty";
        return result;
    }

    const int experts = llama_infinitum_moe_expert_count(info);
    const std::size_t gate_up_source_bytes =
        ggml_row_size(GGML_TYPE_Q8_0, hidden_size) *
        std::size_t(info.gate_up_rows);
    const std::size_t down_source_bytes =
        ggml_row_size(GGML_TYPE_Q8_0, info.down_rows) *
        std::size_t(hidden_size);
    const std::size_t gate_up_layer_bytes = gate_up_source_bytes * std::size_t(experts);
    const std::size_t down_layer_bytes = down_source_bytes * std::size_t(experts);

    const std::string pack_path = llama_infinitum_moe_ggml_pack_path(info);
    std::string mapping_error;
    auto mapping = llama_infinitum_moe_ggml_pack_mapping_for(pack_path, mapping_error);
    if (mapping == nullptr || mapping->mapping == nullptr) {
        result.error = "failed to mmap Q8 expert pack: " + mapping_error;
        return result;
    }

    const auto * pack_base = static_cast<const std::uint8_t *>(mapping->mapping->addr());
    const std::size_t layer_base = std::size_t(layer_index) * (gate_up_layer_bytes + down_layer_bytes);

    // Default to 4 expert workers (max 8) and 1 row thread
    const int expert_workers = gemma4_direct_pack_workers_from_env(
        "LLAMA_INFINITUM_GEMMA4_DIRECT_Q8_EXPERT_WORKERS", 4,
        static_cast<int>(expert_ids.size()));
    const int row_threads = expert_workers > 1 ? 1 :
        gemma4_direct_pack_workers_from_env("LLAMA_INFINITUM_GEMMA4_DIRECT_Q8_ROW_THREADS", 1, 8);
    const bool fast_gelu = llama_infinitum_gemma4_direct_q8_fast_gelu_enabled();

    std::fill(output, output + hidden_size, 0.0f);

    gemma4_direct_q8_cache & q8_cache = gemma4_direct_q8_cache_get();
    const bool use_cache = gemma4_direct_q8_cache_bytes_from_env() != 0;

    auto resolve_weight_ptrs = [&](int expert_id,
            std::size_t gate_up_offset, std::size_t down_offset,
            std::shared_ptr<const gemma4_direct_q8_cache_entry> & cache_entry,
            const void * & gate_up_ptr, const void * & down_ptr) {
        if (use_cache) {
            cache_entry = q8_cache.get_or_load(
                layer_index, expert_id,
                pack_base, gate_up_offset, down_offset,
                gate_up_source_bytes, down_source_bytes);
        }
        if (cache_entry != nullptr) {
            gate_up_ptr = cache_entry->gate_up();
            down_ptr = cache_entry->down();
        } else {
            // Fallback to direct mmap with madvise
#if defined(__linux__) && defined(MADV_WILLNEED)
            madvise(const_cast<std::uint8_t *>(pack_base + gate_up_offset), gate_up_source_bytes, MADV_WILLNEED);
            madvise(const_cast<std::uint8_t *>(pack_base + down_offset), down_source_bytes, MADV_WILLNEED);
#endif
            gate_up_ptr = pack_base + gate_up_offset;
            down_ptr = pack_base + down_offset;
        }
    };

    const auto compute_start = std::chrono::steady_clock::now();

    if (expert_workers <= 1 || expert_ids.size() == 1) {
        thread_local std::vector<float> expert_output_scratch;
        expert_output_scratch.resize(std::size_t(hidden_size));
        for (std::size_t i = 0; i < expert_ids.size(); ++i) {
            const int expert_id = expert_ids[i];
            const float weight = i < expert_weights.size() ? expert_weights[i] : (1.0f / float(expert_ids.size()));
            const std::size_t gate_up_offset = layer_base + std::size_t(expert_id) * gate_up_source_bytes;
            const std::size_t down_offset = layer_base + gate_up_layer_bytes + std::size_t(expert_id) * down_source_bytes;

            std::shared_ptr<const gemma4_direct_q8_cache_entry> cache_entry;
            const void * gate_up_ptr = nullptr;
            const void * down_ptr = nullptr;
            resolve_weight_ptrs(expert_id, gate_up_offset, down_offset, cache_entry, gate_up_ptr, down_ptr);

            gemma4_q8_compute_one_expert(
                gate_up_ptr, down_ptr,
                hidden_size, info.down_rows,
                hidden, weight,
                expert_output_scratch.data(),
                row_threads, fast_gelu);

            for (int row = 0; row < hidden_size; ++row)
                output[row] += expert_output_scratch[row];
        }
    } else {
        thread_local std::vector<float> outputs_scratch;
        outputs_scratch.resize(expert_ids.size() * std::size_t(hidden_size));
        float * expert_outputs = outputs_scratch.data();

        std::atomic<int> next_expert{0};
        auto compute_expert = [&]() {
            while (true) {
                const int index = next_expert.fetch_add(1, std::memory_order_relaxed);
                if (index >= static_cast<int>(expert_ids.size())) break;
                const std::size_t i = std::size_t(index);
                const int expert_id = expert_ids[i];
                const float weight = i < expert_weights.size() ? expert_weights[i] : (1.0f / float(expert_ids.size()));
                const std::size_t gate_up_offset = layer_base + std::size_t(expert_id) * gate_up_source_bytes;
                const std::size_t down_offset = layer_base + gate_up_layer_bytes + std::size_t(expert_id) * down_source_bytes;

                std::shared_ptr<const gemma4_direct_q8_cache_entry> cache_entry;
                const void * gate_up_ptr = nullptr;
                const void * down_ptr = nullptr;
                resolve_weight_ptrs(expert_id, gate_up_offset, down_offset, cache_entry, gate_up_ptr, down_ptr);

                gemma4_q8_compute_one_expert(
                    gate_up_ptr, down_ptr,
                    hidden_size, info.down_rows,
                    hidden, weight,
                    expert_outputs + i * std::size_t(hidden_size),
                    1, fast_gelu);
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(std::size_t(std::max(0, expert_workers - 1)));
        for (int w = 1; w < expert_workers; ++w)
            workers.emplace_back(compute_expert);
        compute_expert();
        for (auto & w : workers) if (w.joinable()) w.join();

        for (std::size_t i = 0; i < expert_ids.size(); ++i) {
            const float * eout = expert_outputs + i * std::size_t(hidden_size);
            for (int row = 0; row < hidden_size; ++row)
                output[row] += eout[row];
        }
    }

    result.total_compute_ms = llama_infinitum_elapsed_ms(compute_start, std::chrono::steady_clock::now());
    result.cache_hits_delta = use_cache ? q8_cache.hits() : 0;
    result.cache_misses_delta = use_cache ? q8_cache.misses() : 0;
    result.resident_bytes = use_cache ? q8_cache.resident_bytes() : 0;

    if (llama_infinitum_debug_enabled()) {
        std::fprintf(stderr, "q8_direct: layer=%d experts=%zu compute_ms=%.1f hits=%llu misses=%llu resident_mb=%.1f\n",
            layer_index, expert_ids.size(),
            double(result.total_compute_ms),
            static_cast<unsigned long long>(result.cache_hits_delta),
            static_cast<unsigned long long>(result.cache_misses_delta),
            double(result.resident_bytes) / (1024.0 * 1024.0));
    }

    // Output diagnostics
    double norm = 0.0;
    bool all_finite = true;
    for (int row = 0; row < hidden_size; ++row) {
        const float value = output[row];
        all_finite = all_finite && std::isfinite(value);
        norm += double(value) * double(value);
    }
    result.output_norm = float(std::sqrt(norm));
    result.ok = all_finite;
    if (!all_finite) result.error = "non-finite Q8 direct output";
    return result;
}

// ---------------------------------------------------------------------------
// F16 direct path (unchanged)
// ---------------------------------------------------------------------------

llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_gemma4_f16_pack_selected_experts_into(
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        const std::vector<int> & expert_ids,
        const std::vector<float> & expert_weights,
        const float * hidden,
        int hidden_size,
        float * output) {
    llama_infinitum_moe_expert_mlp_result result;
    result.layer_index = layer_index;
    result.expert_id = -1;
    result.backend = std::string("cpu-gemma4-direct-f16-") + llama_infinitum_gemma4_f16_dot_backend_name();
    if (!llama_infinitum_gemma4_direct_f16_enabled()) {
        result.error = "Gemma4 direct F16 path is disabled";
        return result;
    }
    if (!llama_infinitum_gemma4_f16_pack_enabled()) {
        result.error = "Gemma4 direct F16 path requires LLAMA_INFINITUM_GEMMA4_GGML_EXPERT_PACK_TYPE=f16";
        return result;
    }
    if (hidden == nullptr || output == nullptr || hidden_size <= 0) {
        result.error = "Gemma4 direct F16 selected expert path received invalid buffers";
        return result;
    }
    if (info.hidden_size > 0 && info.hidden_size != hidden_size) {
        result.error = "Gemma4 direct F16 hidden size does not match expert index: hidden=" +
            std::to_string(hidden_size) + " index_hidden=" + std::to_string(info.hidden_size);
        return result;
    }
    if (info.gate_up_rows <= 0 || info.down_rows <= 0 || info.gate_up_rows != info.down_rows * 2) {
        result.error = "Gemma4 direct F16 expert index shape is invalid";
        return result;
    }
    if (expert_ids.empty()) {
        result.error = "Gemma4 direct F16 selected expert list is empty";
        return result;
    }

    const int experts = llama_infinitum_moe_expert_count(info);
    const std::size_t gate_up_source_bytes =
        ggml_row_size(GGML_TYPE_F16, hidden_size) *
        std::size_t(info.gate_up_rows);
    const std::size_t down_source_bytes =
        ggml_row_size(GGML_TYPE_F16, info.down_rows) *
        std::size_t(hidden_size);
    const std::size_t gate_up_layer_bytes = gate_up_source_bytes * std::size_t(experts);
    const std::size_t down_layer_bytes = down_source_bytes * std::size_t(experts);
    if (experts <= 0 || gate_up_source_bytes == 0 || down_source_bytes == 0) {
        result.error = "Gemma4 direct F16 expert pack geometry is invalid";
        return result;
    }

    const std::string pack_path = llama_infinitum_moe_ggml_pack_path(info);
    std::string mapping_error;
    auto mapping = llama_infinitum_moe_ggml_pack_mapping_for(pack_path, mapping_error);
    if (mapping == nullptr || mapping->mapping == nullptr) {
        result.error = "failed to mmap Gemma4 direct F16 expert pack: " + mapping_error;
        return result;
    }
    const std::size_t layer_base = std::size_t(layer_index) * (gate_up_layer_bytes + down_layer_bytes);
    if (layer_base + gate_up_layer_bytes + down_layer_bytes > mapping->file->size()) {
        result.error = "Gemma4 direct F16 expert pack is smaller than expected for requested layer";
        return result;
    }
    for (const int expert_id : expert_ids) {
        if (expert_id < 0 || expert_id >= experts) {
            result.error = "Gemma4 direct F16 selected expert id is out of range";
            return result;
        }
    }

    thread_local std::vector<ggml_fp16_t> hidden_f16_scratch;
    hidden_f16_scratch.resize(std::size_t(hidden_size));
    ggml_fp32_to_fp16_row(hidden, hidden_f16_scratch.data(), hidden_size);
    const ggml_fp16_t * hidden_f16_data = hidden_f16_scratch.data();

    const int fallback_workers = std::min<int>(4, static_cast<int>(expert_ids.size()));
    int expert_workers = llama_infinitum_gemma4_direct_f16_workers_from_env(
            "LLAMA_INFINITUM_GEMMA4_DIRECT_F16_EXPERT_WORKERS",
            fallback_workers,
            static_cast<int>(expert_ids.size()));
    if (expert_workers > 1 && !llama_infinitum_env_enabled("LLAMA_INFINITUM_GEMMA4_DIRECT_F16_UNSAFE_EXPERT_PARALLEL")) {
        expert_workers = 1;
    }
    int row_threads = llama_infinitum_gemma4_direct_f16_workers_from_env(
            "LLAMA_INFINITUM_GEMMA4_DIRECT_F16_ROW_THREADS",
            1,
            8);
    if (expert_workers > 1) {
        row_threads = 1;
    }
    const bool collect_profile = llama_infinitum_env_enabled("LLAMA_INFINITUM_PROFILE");
    const auto compute_start = std::chrono::steady_clock::now();
    const auto * pack_base = static_cast<const std::uint8_t *>(mapping->mapping->addr());
    const std::uint64_t touched_bytes =
        std::uint64_t(expert_ids.size()) *
        std::uint64_t(gate_up_source_bytes + down_source_bytes);
    llama_infinitum_gemma4_direct_f16_cache & direct_cache =
        llama_infinitum_gemma4_direct_f16_cache_get();
    const bool direct_cache_enabled = llama_infinitum_gemma4_direct_f16_cache_bytes_from_env() != 0;
    const std::uint64_t cache_hits_before = direct_cache_enabled ? direct_cache.hits() : 0;
    const std::uint64_t cache_misses_before = direct_cache_enabled ? direct_cache.misses() : 0;
    const std::uint64_t cache_evictions_before = direct_cache_enabled ? direct_cache.evictions() : 0;

    auto resolve_weight_ptrs = [&](int expert_id,
            std::size_t gate_up_offset,
            std::size_t down_offset,
            std::shared_ptr<const llama_infinitum_gemma4_direct_f16_cache_entry> & cache_entry,
            const ggml_fp16_t * & gate_up_ptr,
            const ggml_fp16_t * & down_ptr) {
        if (direct_cache_enabled) {
            cache_entry = direct_cache.get_or_load(
                    layer_index,
                    expert_id,
                    pack_base,
                    gate_up_offset,
                    down_offset,
                    gate_up_source_bytes,
                    down_source_bytes);
        }
        if (cache_entry != nullptr) {
            gate_up_ptr = cache_entry->gate_up();
            down_ptr = cache_entry->down();
        } else {
            const std::uint8_t * gate_up_raw = pack_base + gate_up_offset;
            const std::uint8_t * down_raw = pack_base + down_offset;
            llama_infinitum_gemma4_direct_f16_madvise(gate_up_raw, gate_up_source_bytes);
            llama_infinitum_gemma4_direct_f16_madvise(down_raw, down_source_bytes);
            gate_up_ptr = reinterpret_cast<const ggml_fp16_t *>(gate_up_raw);
            down_ptr = reinterpret_cast<const ggml_fp16_t *>(down_raw);
        }
    };

    std::fill(output, output + hidden_size, 0.0f);
    bool all_finite = true;

    if (expert_workers <= 1 || expert_ids.size() == 1) {
        thread_local std::vector<float> expert_output_scratch;
        expert_output_scratch.resize(std::size_t(hidden_size));
        llama_infinitum_moe_compute_profile profile_sum;
        int first_bad_expert = -1;
        int first_bad_row = -1;
        int first_bad_stage = 0;
        float first_bad_value = 0.0f;
        for (std::size_t i = 0; i < expert_ids.size(); ++i) {
            const int expert_id = expert_ids[i];
            const float weight = i < expert_weights.size() ? expert_weights[i] : (1.0f / float(expert_ids.size()));
            const std::size_t gate_up_offset = layer_base + std::size_t(expert_id) * gate_up_source_bytes;
            const std::size_t down_offset = layer_base + gate_up_layer_bytes + std::size_t(expert_id) * down_source_bytes;
            std::shared_ptr<const llama_infinitum_gemma4_direct_f16_cache_entry> cache_entry;
            const ggml_fp16_t * gate_up_ptr = nullptr;
            const ggml_fp16_t * down_ptr = nullptr;
            resolve_weight_ptrs(expert_id, gate_up_offset, down_offset, cache_entry, gate_up_ptr, down_ptr);
            llama_infinitum_moe_compute_profile profile;
            const bool finite = llama_infinitum_gemma4_f16_compute_one_expert(
                    gate_up_ptr,
                    down_ptr,
                    hidden_size,
                    info.down_rows,
                    hidden_f16_data,
                    weight,
                    expert_output_scratch.data(),
                    collect_profile,
                    &profile,
                    row_threads,
                    &first_bad_row,
                    &first_bad_stage,
                    &first_bad_value);
            all_finite = all_finite && finite;
            if (!finite && first_bad_expert < 0) {
                first_bad_expert = expert_id;
            }
            if (collect_profile) {
                profile_sum.gate_up_ms += profile.gate_up_ms;
                profile_sum.activation_ms += profile.activation_ms;
                profile_sum.down_ms += profile.down_ms;
            }
            for (int row = 0; row < hidden_size; ++row) {
                output[row] += expert_output_scratch[row];
            }
        }
        result.gate_up_ms = profile_sum.gate_up_ms;
        result.activation_ms = profile_sum.activation_ms;
        result.down_ms = profile_sum.down_ms;
        if (!all_finite) {
            result.error = "non-finite Gemma4 direct F16 selected experts output expert=" +
                std::to_string(first_bad_expert) + " row=" + std::to_string(first_bad_row) +
                " stage=" + std::to_string(first_bad_stage) + " value=" + std::to_string(first_bad_value);
        }
    } else {
        thread_local std::vector<float> outputs_scratch;
        thread_local std::vector<int> finite_scratch;
        thread_local std::vector<int> bad_rows_scratch;
        thread_local std::vector<int> bad_stages_scratch;
        thread_local std::vector<float> bad_values_scratch;
        thread_local std::vector<llama_infinitum_moe_compute_profile> profiles_scratch;
        outputs_scratch.resize(expert_ids.size() * std::size_t(hidden_size));
        finite_scratch.assign(expert_ids.size(), 1);
        bad_rows_scratch.assign(expert_ids.size(), -1);
        bad_stages_scratch.assign(expert_ids.size(), 0);
        bad_values_scratch.assign(expert_ids.size(), 0.0f);
        profiles_scratch.assign(collect_profile ? expert_ids.size() : 0, llama_infinitum_moe_compute_profile{});
        float * expert_outputs = outputs_scratch.data();
        int * finite = finite_scratch.data();
        int * bad_rows = bad_rows_scratch.data();
        int * bad_stages = bad_stages_scratch.data();
        float * bad_values = bad_values_scratch.data();
        llama_infinitum_moe_compute_profile * profiles = profiles_scratch.empty() ? nullptr : profiles_scratch.data();

        std::atomic<int> next_expert{0};
        auto compute_expert = [&]() {
            while (true) {
                const int index = next_expert.fetch_add(1, std::memory_order_relaxed);
                if (index >= static_cast<int>(expert_ids.size())) {
                    break;
                }
            const std::size_t i = std::size_t(index);
            const int expert_id = expert_ids[i];
            const float weight = i < expert_weights.size() ? expert_weights[i] : (1.0f / float(expert_ids.size()));
            const std::size_t gate_up_offset = layer_base + std::size_t(expert_id) * gate_up_source_bytes;
            const std::size_t down_offset = layer_base + gate_up_layer_bytes + std::size_t(expert_id) * down_source_bytes;
            std::shared_ptr<const llama_infinitum_gemma4_direct_f16_cache_entry> cache_entry;
            const ggml_fp16_t * gate_up_ptr = nullptr;
            const ggml_fp16_t * down_ptr = nullptr;
            resolve_weight_ptrs(expert_id, gate_up_offset, down_offset, cache_entry, gate_up_ptr, down_ptr);
            finite[i] = llama_infinitum_gemma4_f16_compute_one_expert(
                    gate_up_ptr,
                    down_ptr,
                    hidden_size,
                    info.down_rows,
                    hidden_f16_data,
                    weight,
                    expert_outputs + i * std::size_t(hidden_size),
                    collect_profile,
                    profiles != nullptr ? &profiles[i] : nullptr,
                    row_threads,
                    &bad_rows[i],
                    &bad_stages[i],
                    &bad_values[i]) ? 1 : 0;
            }
        };
        std::vector<std::thread> workers;
        workers.reserve(std::size_t(std::max(0, expert_workers - 1)));
        for (int worker = 1; worker < expert_workers; ++worker) {
            workers.emplace_back(compute_expert);
        }
        compute_expert();
        for (std::thread & worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        const bool retry_failed_parallel = llama_infinitum_gemma4_direct_f16_parallel_retry_enabled();
        const bool debug_parallel = llama_infinitum_debug_enabled();
        if (collect_profile) {
            for (const llama_infinitum_moe_compute_profile & profile : profiles_scratch) {
                result.gate_up_ms += profile.gate_up_ms;
                result.activation_ms += profile.activation_ms;
                result.down_ms += profile.down_ms;
            }
        }
        int first_bad_expert = -1;
        int first_bad_row = -1;
        int first_bad_stage = 0;
        float first_bad_value = 0.0f;
        for (std::size_t i = 0; i < expert_ids.size(); ++i) {
            if (finite[i] == 0 && debug_parallel) {
                const float * expert_output = expert_outputs + i * std::size_t(hidden_size);
                double expert_norm = 0.0;
                int nonfinite_rows = 0;
                int first_nonfinite_row = -1;
                for (int row = 0; row < hidden_size; ++row) {
                    const float value = expert_output[row];
                    if (!std::isfinite(value)) {
                        if (first_nonfinite_row < 0) {
                            first_nonfinite_row = row;
                        }
                        ++nonfinite_rows;
                    } else {
                        expert_norm += double(value) * double(value);
                    }
                }
                llama_infinitum_debug_log(
                        "gemma4 direct F16 parallel expert failed layer=%d expert=%d slot=%zu bad_row=%d bad_stage=%d bad_value=%g output_norm=%g output_nonfinite=%d output_first_nonfinite=%d",
                        layer_index,
                        expert_ids[i],
                        i,
                        bad_rows[i],
                        bad_stages[i],
                        double(bad_values[i]),
                        std::sqrt(expert_norm),
                        nonfinite_rows,
                        first_nonfinite_row);
            }
            if (finite[i] == 0 && retry_failed_parallel) {
                const int expert_id = expert_ids[i];
                const float weight = i < expert_weights.size() ? expert_weights[i] : (1.0f / float(expert_ids.size()));
                const std::size_t gate_up_offset = layer_base + std::size_t(expert_id) * gate_up_source_bytes;
                const std::size_t down_offset = layer_base + gate_up_layer_bytes + std::size_t(expert_id) * down_source_bytes;
                std::shared_ptr<const llama_infinitum_gemma4_direct_f16_cache_entry> cache_entry;
                const ggml_fp16_t * gate_up_ptr = nullptr;
                const ggml_fp16_t * down_ptr = nullptr;
                resolve_weight_ptrs(expert_id, gate_up_offset, down_offset, cache_entry, gate_up_ptr, down_ptr);
                llama_infinitum_moe_compute_profile retry_profile;
                finite[i] = llama_infinitum_gemma4_f16_compute_one_expert(
                        gate_up_ptr,
                        down_ptr,
                        hidden_size,
                        info.down_rows,
                        hidden_f16_data,
                        weight,
                        expert_outputs + i * std::size_t(hidden_size),
                        collect_profile,
                        &retry_profile,
                        1,
                        &bad_rows[i],
                        &bad_stages[i],
                        &bad_values[i]) ? 1 : 0;
                if (collect_profile) {
                    result.gate_up_ms += retry_profile.gate_up_ms;
                    result.activation_ms += retry_profile.activation_ms;
                    result.down_ms += retry_profile.down_ms;
                }
                if (debug_parallel) {
                    llama_infinitum_debug_log(
                            "gemma4 direct F16 parallel retry layer=%d expert=%d ok=%d bad_row=%d bad_stage=%d bad_value=%g",
                            layer_index,
                            expert_id,
                            finite[i],
                            bad_rows[i],
                            bad_stages[i],
                            double(bad_values[i]));
                }
            }
            all_finite = all_finite && finite[i] != 0;
            if (finite[i] == 0 && first_bad_expert < 0) {
                first_bad_expert = expert_ids[i];
                first_bad_row = bad_rows[i];
                first_bad_stage = bad_stages[i];
                first_bad_value = bad_values[i];
            }
            const float * expert_output = expert_outputs + i * std::size_t(hidden_size);
            for (int row = 0; row < hidden_size; ++row) {
                output[row] += expert_output[row];
            }
        }
        if (!all_finite) {
            result.error = "non-finite Gemma4 direct F16 selected experts output expert=" +
                std::to_string(first_bad_expert) + " row=" + std::to_string(first_bad_row) +
                " stage=" + std::to_string(first_bad_stage) + " value=" + std::to_string(first_bad_value);
        }
    }

    result.total_compute_ms = llama_infinitum_elapsed_ms(compute_start, std::chrono::steady_clock::now());
    result.touched_bytes_delta = touched_bytes;
    result.mapped_bytes_delta = touched_bytes;
    result.loaded_bytes_delta = 0;
    result.cache_hits_delta = direct_cache_enabled ? direct_cache.hits() - cache_hits_before : 0;
    result.cache_misses_delta = direct_cache_enabled ? direct_cache.misses() - cache_misses_before : 0;
    result.cache_evictions_delta = direct_cache_enabled ? direct_cache.evictions() - cache_evictions_before : 0;
    result.resident_bytes = direct_cache_enabled ? direct_cache.resident_bytes() : 0;

    double norm = 0.0;
    for (int row = 0; row < hidden_size; ++row) {
        const float value = output[row];
        all_finite = all_finite && std::isfinite(value);
        norm += double(value) * double(value);
    }
    result.output_norm = float(std::sqrt(norm));
    result.ok = all_finite;
    if (!all_finite && result.error.empty()) {
        result.error = "non-finite Gemma4 direct F16 selected experts output";
    }
    return result;
}

llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_gemma4_q4_pack_selected_experts_into(
        const llama_infinitum_moe_index_info & info,
        int layer_index,
        const std::vector<int> & expert_ids,
        const std::vector<float> & expert_weights,
        const float * hidden,
        int hidden_size,
        float * output) {
    llama_infinitum_moe_expert_mlp_result result;
    result.layer_index = layer_index;
    result.expert_id = -1;
    result.backend = "vulkan-gemma4-q4-slots";
    if (hidden == nullptr || output == nullptr || hidden_size <= 0) {
        result.error = "Gemma4 Q4 selected expert graph received invalid buffers";
        return result;
    }
    if (info.hidden_size > 0 && info.hidden_size != hidden_size) {
        result.error = "Gemma4 Q4 hidden size does not match expert index: hidden=" +
            std::to_string(hidden_size) + " index_hidden=" + std::to_string(info.hidden_size);
        return result;
    }
    if (info.gate_up_rows <= 0 || info.down_rows <= 0 || info.gate_up_rows != info.down_rows * 2) {
        result.error = "Gemma4 Q4 expert index shape is invalid";
        return result;
    }
    if (expert_ids.empty()) {
        result.error = "Gemma4 Q4 selected expert list is empty";
        return result;
    }

    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    std::string error;
    const bool use_cpu_backend = llama_infinitum_env_enabled("LLAMA_INFINITUM_GEMMA4_Q4_CPU_BACKEND");
    if (use_cpu_backend) {
        backend = ggml_backend_cpu_init();
        if (backend == nullptr) {
            result.error = "failed to init CPU backend";
            return result;
        }
        device = ggml_backend_get_device(backend);
        result.backend = "cpu-gemma4-q4-slots";
    } else {
        llama_infinitum_ggml_gpu_backend & state = llama_infinitum_ggml_gpu_backend_get();
        if (!state.get(backend, device, error)) {
            result.error = error;
            return result;
        }
    }

    std::vector<std::int32_t> local_ids(expert_ids.size());
    std::vector<float> local_weights(expert_ids.size());
    const auto compute_start = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> compute_lock;
        if (!use_cpu_backend) {
            llama_infinitum_ggml_gpu_backend & gpu_state = llama_infinitum_ggml_gpu_backend_get();
            compute_lock = std::unique_lock<std::mutex>(gpu_state.compute_mutex());
        }
        static llama_infinitum_gemma4_q4_gpu_layer_slot_cache_manager slot_cache_manager;
        llama_infinitum_gemma4_q4_gpu_layer_slot_cache & layer_cache = slot_cache_manager.get_layer(layer_index);
        if (!layer_cache.init(
                info,
                layer_index,
                hidden_size,
                info.gate_up_rows,
                info.down_rows,
                llama_infinitum_moe_gpu_layer_slots_requested_from_env(),
                static_cast<int>(expert_ids.size()),
                backend,
                error)) {
            result.error = error;
            return result;
        }
        const std::uint64_t hits_before = layer_cache.cache_hits();
        const std::uint64_t misses_before = layer_cache.cache_misses();
        const std::uint64_t evictions_before = layer_cache.cache_evictions();
        const std::uint64_t uploaded_before = layer_cache.uploaded_bytes();
        if (!layer_cache.ensure_experts(expert_ids, local_ids, error)) {
            result.error = error;
            return result;
        }
        for (std::size_t i = 0; i < expert_ids.size(); ++i) {
            local_weights[i] = i < expert_weights.size() ? expert_weights[i] : (1.0f / float(expert_ids.size()));
        }

        if (llama_infinitum_gemma4_q4_slot_views_enabled()) {
            result.backend = "vulkan-gemma4-q4-slot-views";
            if (!llama_infinitum_gemma4_q4_gpu_compute_selected_experts_views(
                    layer_cache,
                    hidden,
                    hidden_size,
                    local_ids.data(),
                    local_weights.data(),
                    static_cast<int>(expert_ids.size()),
                    backend,
                    device,
                    output,
                    error)) {
                result.error = error;
                return result;
            }
        } else {
            static llama_infinitum_gemma4_q4_gpu_layer_compute_graph_manager graph_manager;
            llama_infinitum_gemma4_q4_gpu_layer_compute_graph & graph = graph_manager.get_or_create(
                layer_cache,
                layer_index,
                hidden_size,
                static_cast<int>(expert_ids.size()),
                backend,
                device,
                error);
            if (!graph.compute(hidden, local_ids.data(), local_weights.data(), backend, output, error)) {
                result.error = error;
                return result;
            }
        }
        result.cache_hits_delta = layer_cache.cache_hits() - hits_before;
        result.cache_misses_delta = layer_cache.cache_misses() - misses_before;
        result.cache_evictions_delta = layer_cache.cache_evictions() - evictions_before;
        result.loaded_bytes_delta = layer_cache.uploaded_bytes() - uploaded_before;
        result.resident_bytes = layer_cache.resident_bytes();
    }

    result.total_compute_ms = llama_infinitum_elapsed_ms(compute_start, std::chrono::steady_clock::now());
    double norm = 0.0;
    bool all_finite = true;
    for (int row = 0; row < hidden_size; ++row) {
        const float value = output[row];
        all_finite = all_finite && std::isfinite(value);
        norm += double(value) * double(value);
    }
    result.output_norm = float(std::sqrt(norm));
    result.ok = all_finite;
    if (!all_finite) {
        result.error = "non-finite Gemma4 Q4 selected experts GPU output";
    }
    return result;
}

static bool llama_infinitum_ggml_gpu_compute_selected_experts_batched(
        const std::vector<std::shared_ptr<const llama_infinitum_ggml_packed_expert_mlp>> & experts,
        const std::vector<float> & expert_weights,
        const float * hidden_data,
        int hidden_size,
        float * output,
        std::string & error) {
    if (experts.empty()) {
        error = "selected expert list is empty";
        return false;
    }

    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    llama_infinitum_ggml_gpu_backend & state = llama_infinitum_ggml_gpu_backend_get();
    if (!state.get(backend, device, error)) {
        return false;
    }

    std::lock_guard<std::mutex> compute_guard(state.compute_mutex());
    static llama_infinitum_ggml_gpu_expert_tensor_cache tensor_cache;
    std::vector<std::shared_ptr<llama_infinitum_ggml_gpu_expert_tensors>> gpu_experts;
    gpu_experts.reserve(experts.size());
    for (const std::shared_ptr<const llama_infinitum_ggml_packed_expert_mlp> & expert : experts) {
        auto gpu_expert = tensor_cache.get_or_create(*expert, hidden_size, backend, error);
        if (gpu_expert == nullptr) {
            return false;
        }
        gpu_experts.push_back(std::move(gpu_expert));
    }

    constexpr std::size_t ctx_mem = 8ull * 1024ull * 1024ull;
    ggml_init_params params = {};
    params.mem_size = ctx_mem;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        error = "failed to initialize ggml context for batched Vulkan experts";
        return false;
    }

    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 1);
    ggml_tensor * sum = nullptr;
    for (std::size_t i = 0; i < gpu_experts.size(); ++i) {
        const llama_infinitum_ggml_gpu_expert_tensors & expert = *gpu_experts[i];
        ggml_tensor * gate = ggml_add(ctx, ggml_mul_mat(ctx, expert.gate_w, hidden), expert.gate_b);
        ggml_tensor * up = ggml_add(ctx, ggml_mul_mat(ctx, expert.up_w, hidden), expert.up_b);
        ggml_tensor * act = ggml_swiglu_oai(ctx, gate, up, 1.702f, 7.0f);
        ggml_tensor * out = ggml_add(ctx, ggml_mul_mat(ctx, expert.down_w, act), expert.down_b);
        const float weight = i < expert_weights.size() ? expert_weights[i] : (1.0f / float(gpu_experts.size()));
        ggml_tensor * weighted = ggml_scale(ctx, out, weight);
        sum = sum == nullptr ? weighted : ggml_add(ctx, sum, weighted);
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sum);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!ggml_backend_dev_supports_op(device, ggml_graph_node(graph, i))) {
            error = "GPU backend does not support a node in the batched selected expert graph";
            ggml_free(ctx);
            return false;
        }
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        error = "failed to allocate GPU tensors for batched selected expert graph";
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(hidden, hidden_data, 0, std::size_t(hidden_size) * sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        error = "GPU backend failed computing batched selected expert graph";
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(sum, output, 0, std::size_t(hidden_size) * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

static bool llama_infinitum_ggml_gpu_compute_selected_experts_id(
        const std::vector<std::shared_ptr<const llama_infinitum_ggml_packed_expert_mlp>> & experts,
        const std::vector<float> & expert_weights,
        const float * hidden_data,
        int hidden_size,
        float * output,
        std::string & error,
        double * expert_upload_ms = nullptr,
        double * graph_input_ms = nullptr,
        double * graph_compute_ms = nullptr,
        double * graph_output_ms = nullptr,
        std::uint64_t * gpu_slot_hits_delta = nullptr,
        std::uint64_t * gpu_slot_misses_delta = nullptr,
        std::uint64_t * gpu_slot_evictions_delta = nullptr) {
    const int expert_count = static_cast<int>(experts.size());
    if (expert_count <= 0) {
        error = "selected expert list is empty";
        return false;
    }

    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    llama_infinitum_ggml_gpu_backend & state = llama_infinitum_ggml_gpu_backend_get();
    if (!state.get(backend, device, error)) {
        return false;
    }

    std::vector<std::int32_t> local_ids(expert_count);
    std::vector<float> local_weights(expert_count);

    std::lock_guard<std::mutex> compute_guard(state.compute_mutex());
    llama_infinitum_ggml_gpu_layer_slot_cache & layer_cache =
        llama_infinitum_ggml_gpu_layer_slot_cache_manager_get().get_layer(experts[0]->layer_index);
    if (!layer_cache.init(
            experts[0]->layer_index,
            hidden_size,
            llama_infinitum_moe_gpu_selected_slots_from_env(),
            llama_infinitum_moe_gpu_global_slots_enabled(),
            backend,
            error)) {
        return false;
    }
    const std::uint64_t slot_hits_before = layer_cache.hits();
    const std::uint64_t slot_misses_before = layer_cache.misses();
    const std::uint64_t slot_evictions_before = layer_cache.evictions();

    const auto upload_start = std::chrono::steady_clock::now();
    for (int i = 0; i < expert_count; ++i) {
        const llama_infinitum_ggml_packed_expert_mlp & expert = *experts[i];
        if (expert.layer_index != experts[0]->layer_index) {
            error = "selected experts span multiple layers";
            return false;
        }
        int slot = -1;
        if (!layer_cache.ensure_expert(expert, backend, slot, error, false)) {
            return false;
        }
        local_ids[i] = slot;
        local_weights[i] = i < static_cast<int>(expert_weights.size()) ? expert_weights[i] : (1.0f / float(expert_count));
    }
    const auto upload_end = std::chrono::steady_clock::now();
    if (expert_upload_ms != nullptr) {
        *expert_upload_ms = llama_infinitum_elapsed_ms(upload_start, upload_end);
    }
    if (gpu_slot_hits_delta != nullptr) {
        *gpu_slot_hits_delta = layer_cache.hits() - slot_hits_before;
    }
    if (gpu_slot_misses_delta != nullptr) {
        *gpu_slot_misses_delta = layer_cache.misses() - slot_misses_before;
    }
    if (gpu_slot_evictions_delta != nullptr) {
        *gpu_slot_evictions_delta = layer_cache.evictions() - slot_evictions_before;
    }

    const bool q8_input = llama_infinitum_moe_vulkan_q8_input_enabled();
    const bool f16_input = llama_infinitum_moe_vulkan_f16_input_enabled();
    static llama_infinitum_ggml_gpu_layer_compute_graph_manager graph_manager;
    llama_infinitum_ggml_gpu_layer_compute_graph & graph = graph_manager.get_or_create(
        layer_cache,
        experts[0]->layer_index,
        hidden_size,
        expert_count,
        q8_input,
        f16_input,
        backend,
        device,
        error);
    return graph.compute(hidden_data, local_ids.data(), local_weights.data(), backend, output, error,
            graph_input_ms, graph_compute_ms, graph_output_ms);
}

static llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_selected_experts_vulkan_ptr(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        const std::vector<int> & expert_ids,
        const std::vector<float> & expert_weights,
        const float * hidden_data,
        int hidden_size,
        float * output) {
    llama_infinitum_moe_expert_mlp_result result;
    result.layer_index = layer_index;
    result.expert_id = -1;
    result.backend = "vulkan-ggml";
    const int expected_hidden_size = info.hidden_size > 0 ? info.hidden_size : hidden_size;
    if (hidden_data == nullptr || output == nullptr || hidden_size <= 0 || hidden_size != expected_hidden_size) {
        result.error = "hidden size does not match expert index shape";
        return result;
    }
    if (expert_ids.empty()) {
        result.error = "selected expert list is empty";
        return result;
    }
    const int block_count = llama_infinitum_mxfp4_block_count_for_hidden(hidden_size);
    if (block_count <= 0) {
        result.error = "invalid hidden block count";
        return result;
    }

    std::string error;
    std::vector<std::shared_ptr<const llama_infinitum_ggml_packed_expert_mlp>> packed_experts;
    packed_experts.reserve(expert_ids.size());

    const auto load_start = std::chrono::steady_clock::now();
    if (llama_infinitum_moe_ggml_pack_enabled() && llama_infinitum_moe_ggml_pack_slots_enabled()) {
        llama_infinitum_ggml_pack_expert_cache & pack_cache = llama_infinitum_ggml_pack_expert_cache_get();
        for (const int expert_id : expert_ids) {
            auto packed = pack_cache.get_or_load(info, layer_index, expert_id, hidden_size, block_count, error);
            if (packed == nullptr) {
                result.error = error;
                return result;
            }
            result.loaded_bytes += packed->bytes();
            packed_experts.push_back(std::move(packed));
        }
        result.loaded_bytes_delta = result.loaded_bytes;
        result.mapped_bytes_delta = result.loaded_bytes;
    } else {
        std::vector<llama_infinitum_loaded_expert_mlp> experts;
        experts.reserve(expert_ids.size());
        const llama_infinitum_moe_cache_stats before_stats = cache.stats();
        for (const int expert_id : expert_ids) {
            llama_infinitum_loaded_expert_mlp expert;
            if (!llama_infinitum_load_expert_mlp(info, cache, layer_index, expert_id, expert, error)) {
                result.error = error;
                return result;
            }
            result.loaded_bytes += expert.loaded_bytes();
            experts.push_back(std::move(expert));
        }
        const llama_infinitum_moe_cache_stats after_load_stats = cache.stats();
        result.cache_hits_delta = after_load_stats.cache_hits - before_stats.cache_hits;
        result.cache_misses_delta = after_load_stats.cache_misses - before_stats.cache_misses;
        result.cache_evictions_delta = after_load_stats.evictions - before_stats.evictions;
        result.loaded_bytes_delta = after_load_stats.loaded_bytes - before_stats.loaded_bytes;
        result.touched_bytes_delta = after_load_stats.touched_bytes - before_stats.touched_bytes;
        result.mapped_bytes_delta = after_load_stats.mapped_bytes - before_stats.mapped_bytes;
        result.copied_bytes_delta = after_load_stats.copied_bytes - before_stats.copied_bytes;
        result.prepacked_bytes_delta = after_load_stats.prepacked_bytes - before_stats.prepacked_bytes;
        result.resident_bytes = after_load_stats.resident_bytes;
        result.process_resident_bytes = after_load_stats.process_resident_bytes;

        static llama_infinitum_ggml_packed_expert_cache packed_cache;
        for (std::size_t i = 0; i < experts.size(); ++i) {
            auto packed = packed_cache.get_or_pack(layer_index, experts[i], hidden_size, block_count);
            if (packed == nullptr) {
                result.error = "failed to pack selected expert for GPU backend";
                return result;
            }
            packed_experts.push_back(std::move(packed));
        }
    }
    const auto load_end = std::chrono::steady_clock::now();
    result.load_ms = llama_infinitum_elapsed_ms(load_start, load_end);

    std::fill(output, output + hidden_size, 0.0f);
    thread_local std::vector<float> expert_output_scratch;
    expert_output_scratch.resize(hidden_size);
    float * expert_output = expert_output_scratch.data();
    const auto compute_start = std::chrono::steady_clock::now();
    bool all_finite = true;
    double expert_upload_ms = 0.0;
    double graph_input_ms = 0.0;
    double graph_compute_ms = 0.0;
    double graph_output_ms = 0.0;
    std::uint64_t gpu_slot_hits_delta = 0;
    std::uint64_t gpu_slot_misses_delta = 0;
    std::uint64_t gpu_slot_evictions_delta = 0;
    if (!llama_infinitum_ggml_gpu_compute_selected_experts_id(
            packed_experts, expert_weights, hidden_data, hidden_size, expert_output, error,
            &expert_upload_ms, &graph_input_ms, &graph_compute_ms, &graph_output_ms,
            &gpu_slot_hits_delta, &gpu_slot_misses_delta, &gpu_slot_evictions_delta)) {
        if (!llama_infinitum_ggml_gpu_compute_selected_experts_batched(
                packed_experts, expert_weights, hidden_data, hidden_size, expert_output, error)) {
            result.error = error;
            return result;
        }
    }
    for (int row = 0; row < hidden_size; ++row) {
        output[row] = expert_output[row];
    }
    const auto compute_end = std::chrono::steady_clock::now();
    result.total_compute_ms = llama_infinitum_elapsed_ms(compute_start, compute_end);
    result.expert_upload_ms = expert_upload_ms;
    result.graph_input_ms = graph_input_ms;
    result.graph_compute_ms = graph_compute_ms;
    result.graph_output_ms = graph_output_ms;
    result.gpu_slot_hits_delta = gpu_slot_hits_delta;
    result.gpu_slot_misses_delta = gpu_slot_misses_delta;
    result.gpu_slot_evictions_delta = gpu_slot_evictions_delta;
    result.accumulate_ms = 0.0;
    double norm = 0.0;
    for (int row = 0; row < hidden_size; ++row) {
        const float value = output[row];
        all_finite = all_finite && std::isfinite(value);
        norm += double(value) * double(value);
    }
    result.output_norm = float(std::sqrt(norm));
    result.ok = all_finite;
    if (!all_finite) {
        result.error = "non-finite selected experts GPU output";
    }
    return result;
}

static llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_selected_experts_fused_arena_ptr(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        const std::vector<int> & expert_ids,
        const std::vector<float> & expert_weights,
        const float * hidden_data,
        int hidden_size,
        float * output) {
    const std::string target = llama_infinitum_moe_fused_target_from_env();
    (void) target;

    llama_infinitum_moe_expert_mlp_result result =
        llama_infinitum_moe_execute_selected_experts_vulkan_ptr(
            info, cache, layer_index, expert_ids, expert_weights, hidden_data, hidden_size, output);
    if (result.ok) {
        result.backend = "fused_arena-fallback-vulkan";
        result.backend_fallback = true;
    } else if (result.backend.empty()) {
        result.backend = "fused_arena-unavailable";
    }
    return result;
}

static llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_selected_experts_cpu_ptr(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        const std::vector<int> & expert_ids,
        const std::vector<float> & expert_weights,
        const float * hidden_data,
        int hidden_size,
        float * output,
        int expert_workers) {
    llama_infinitum_moe_expert_mlp_result result;
    result.layer_index = layer_index;
    result.expert_id = -1;
    result.backend = "cpu";
    const int expected_hidden_size = info.hidden_size > 0 ? info.hidden_size : hidden_size;
    if (hidden_data == nullptr || output == nullptr || hidden_size <= 0 || hidden_size != expected_hidden_size) {
        result.error = "hidden size does not match expert index shape";
        return result;
    }
    if (expert_ids.empty()) {
        result.error = "selected expert list is empty";
        return result;
    }

    std::vector<llama_infinitum_loaded_expert_mlp> experts;
    experts.reserve(expert_ids.size());
    std::string error;
    const llama_infinitum_moe_cache_stats before_stats = cache.stats();
    const auto load_start = std::chrono::steady_clock::now();
    for (const int expert_id : expert_ids) {
        llama_infinitum_loaded_expert_mlp expert;
        if (!llama_infinitum_load_expert_mlp(info, cache, layer_index, expert_id, expert, error)) {
            result.error = error;
            return result;
        }
        result.loaded_bytes += expert.loaded_bytes();
        experts.push_back(std::move(expert));
    }
    const auto load_end = std::chrono::steady_clock::now();
    const llama_infinitum_moe_cache_stats after_load_stats = cache.stats();
    result.cache_hits_delta = after_load_stats.cache_hits - before_stats.cache_hits;
    result.cache_misses_delta = after_load_stats.cache_misses - before_stats.cache_misses;
    result.cache_evictions_delta = after_load_stats.evictions - before_stats.evictions;
    result.loaded_bytes_delta = after_load_stats.loaded_bytes - before_stats.loaded_bytes;
    result.touched_bytes_delta = after_load_stats.touched_bytes - before_stats.touched_bytes;
    result.mapped_bytes_delta = after_load_stats.mapped_bytes - before_stats.mapped_bytes;
    result.copied_bytes_delta = after_load_stats.copied_bytes - before_stats.copied_bytes;
    result.prepacked_bytes_delta = after_load_stats.prepacked_bytes - before_stats.prepacked_bytes;
    result.resident_bytes = after_load_stats.resident_bytes;
    result.process_resident_bytes = after_load_stats.process_resident_bytes;
    result.load_ms = llama_infinitum_elapsed_ms(load_start, load_end);

    std::fill(output, output + hidden_size, 0.0f);
    const bool collect_profile = llama_infinitum_env_enabled("LLAMA_INFINITUM_PROFILE");
    const llama_infinitum_mxfp4_runtime_flags runtime_flags = llama_infinitum_mxfp4_runtime_flags_from_env();
    llama_infinitum_quantized_input shared_quantized_hidden;
    const llama_infinitum_quantized_input * shared_quantized_hidden_ptr = nullptr;
    if (runtime_flags.use_sdot_gate_up) {
        llama_infinitum_quantize_input_32(hidden_data, hidden_size, shared_quantized_hidden);
        shared_quantized_hidden_ptr = &shared_quantized_hidden;
    }
    thread_local std::vector<float> hidden_pair_table_scratch;
    const float * hidden_pair_table = nullptr;
    if (runtime_flags.use_input_pair_table && !runtime_flags.use_sdot_gate_up) {
        llama_infinitum_mxfp4_fill_input_pair_table(hidden_data, hidden_size, hidden_pair_table_scratch);
        hidden_pair_table = hidden_pair_table_scratch.empty() ? nullptr : hidden_pair_table_scratch.data();
    }
    std::vector<std::shared_ptr<const llama_infinitum_packed_expert_mlp>> packed_experts(experts.size());
    if (!collect_profile && shared_quantized_hidden_ptr != nullptr && runtime_flags.layer_packed_sdot) {
        static llama_infinitum_packed_expert_cache packed_cache;
        for (std::size_t i = 0; i < experts.size(); ++i) {
            packed_experts[i] = packed_cache.get_or_pack(layer_index, experts[i].expert_id, experts[i]);
        }
    }
    const int row_threads = llama_infinitum_expert_row_threads_from_env();
    const int worker_count = row_threads > 1 ? 1 : std::max(1, std::min<int>(expert_workers, int(experts.size())));
    const auto compute_start = std::chrono::steady_clock::now();
    if (worker_count == 1) {
        bool all_finite = true;
        thread_local std::vector<float> expert_output_scratch;
        thread_local std::vector<float> activation_scratch;
        expert_output_scratch.resize(hidden_size);
        activation_scratch.resize(hidden_size);
        float * expert_output = expert_output_scratch.data();
        float * activation = activation_scratch.data();
        llama_infinitum_moe_compute_profile profile_sum;
        for (std::size_t expert_index = 0; expert_index < experts.size(); ++expert_index) {
            llama_infinitum_moe_compute_profile profile;
            const bool ok = llama_infinitum_compute_loaded_expert_mlp(
                experts[expert_index],
                hidden_data,
                hidden_size,
                expert_output,
                activation,
                shared_quantized_hidden_ptr,
                hidden_pair_table,
                collect_profile ? &profile : nullptr,
                packed_experts[expert_index].get(),
                row_threads,
                &runtime_flags);
            all_finite = all_finite && ok;
            if (collect_profile) {
                profile_sum.gate_up_ms += profile.gate_up_ms;
                profile_sum.activation_ms += profile.activation_ms;
                profile_sum.down_ms += profile.down_ms;
            }
            const float weight = expert_index < expert_weights.size() ? expert_weights[expert_index] : (1.0f / float(experts.size()));
            const auto accumulate_start = collect_profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            for (int row = 0; row < hidden_size; ++row) {
                output[row] += expert_output[row] * weight;
            }
            if (collect_profile) {
                result.accumulate_ms += llama_infinitum_elapsed_ms(accumulate_start, std::chrono::steady_clock::now());
            }
        }
        const auto compute_end = std::chrono::steady_clock::now();
        result.total_compute_ms = llama_infinitum_elapsed_ms(compute_start, compute_end);
        result.gate_up_ms = profile_sum.gate_up_ms;
        result.activation_ms = profile_sum.activation_ms;
        result.down_ms = profile_sum.down_ms;

        double norm = 0.0;
        for (int row = 0; row < hidden_size; ++row) {
            const float value = output[row];
            all_finite = all_finite && std::isfinite(value);
            norm += double(value) * double(value);
        }
        result.output_norm = float(std::sqrt(norm));
        result.ok = all_finite;
        if (!all_finite) {
            result.error = "non-finite selected experts output";
        }
        return result;
    }

    thread_local std::vector<float> outputs_scratch;
    thread_local std::vector<float> activations_scratch;
    thread_local std::vector<std::uint8_t> finite_scratch;
    thread_local std::vector<llama_infinitum_moe_compute_profile> profiles_scratch;
    outputs_scratch.resize(experts.size() * hidden_size);
    activations_scratch.resize(experts.size() * hidden_size);
    finite_scratch.assign(experts.size(), 1);
    profiles_scratch.assign(collect_profile ? experts.size() : 0, llama_infinitum_moe_compute_profile{});
    float * outputs = outputs_scratch.data();
    float * activations = activations_scratch.data();
    std::uint8_t * finite = finite_scratch.data();
    llama_infinitum_moe_compute_profile * profiles = profiles_scratch.data();
    static llama_infinitum_worker_pool expert_worker_pool;
    expert_worker_pool.run(worker_count, int(experts.size()), [&](int index) {
            const std::size_t expert_index = std::size_t(index);
            finite[expert_index] = llama_infinitum_compute_loaded_expert_mlp(
                experts[expert_index],
                hidden_data,
                hidden_size,
                outputs + expert_index * hidden_size,
                activations + expert_index * hidden_size,
                shared_quantized_hidden_ptr,
                hidden_pair_table,
                collect_profile ? &profiles[expert_index] : nullptr,
                packed_experts[expert_index].get(),
                1,
                &runtime_flags) ? 1 : 0;
    });
    const auto compute_end = std::chrono::steady_clock::now();
    result.total_compute_ms = llama_infinitum_elapsed_ms(compute_start, compute_end);
    if (collect_profile) {
        for (std::size_t i = 0; i < experts.size(); ++i) {
            const llama_infinitum_moe_compute_profile & profile = profiles[i];
            result.gate_up_ms += profile.gate_up_ms;
            result.activation_ms += profile.activation_ms;
            result.down_ms += profile.down_ms;
        }
    }

    bool all_finite = true;
    const auto accumulate_start = collect_profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    for (std::size_t i = 0; i < experts.size(); ++i) {
        all_finite = all_finite && finite[i] != 0;
        const float weight = i < expert_weights.size() ? expert_weights[i] : (1.0f / float(experts.size()));
        const float * expert_output = outputs + i * hidden_size;
        for (int row = 0; row < hidden_size; ++row) {
            output[row] += expert_output[row] * weight;
        }
    }
    if (collect_profile) {
        result.accumulate_ms = llama_infinitum_elapsed_ms(accumulate_start, std::chrono::steady_clock::now());
    }

    double norm = 0.0;
    for (int row = 0; row < hidden_size; ++row) {
        const float value = output[row];
        all_finite = all_finite && std::isfinite(value);
        norm += double(value) * double(value);
    }
    result.output_norm = float(std::sqrt(norm));
    result.ok = all_finite;
    if (!all_finite) {
        result.error = "non-finite selected experts output";
    }
    return result;
}

llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_selected_experts_cpu(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        const std::vector<int> & expert_ids,
        const std::vector<float> & expert_weights,
        const std::vector<float> & hidden,
        int expert_workers) {
    std::vector<float> output(hidden.size(), 0.0f);
    llama_infinitum_moe_expert_mlp_result result = llama_infinitum_moe_execute_selected_experts_cpu_ptr(
            info,
            cache,
            layer_index,
            expert_ids,
            expert_weights,
            hidden.empty() ? nullptr : hidden.data(),
            static_cast<int>(hidden.size()),
            output.empty() ? nullptr : output.data(),
            expert_workers);
    result.output = std::move(output);
    return result;
}

llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_selected_experts_into(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        const std::vector<int> & expert_ids,
        const std::vector<float> & expert_weights,
        const float * hidden,
        int hidden_size,
        float * output,
        int expert_workers) {
    const llama_infinitum_moe_backend_kind backend = llama_infinitum_moe_backend_kind_from_env();
    if (backend == llama_infinitum_moe_backend_kind::vulkan ||
            backend == llama_infinitum_moe_backend_kind::fused_arena) {
        llama_infinitum_moe_expert_mlp_result gpu_result =
            backend == llama_infinitum_moe_backend_kind::fused_arena ?
                llama_infinitum_moe_execute_selected_experts_fused_arena_ptr(
                    info, cache, layer_index, expert_ids, expert_weights, hidden, hidden_size, output) :
                llama_infinitum_moe_execute_selected_experts_vulkan_ptr(
                    info, cache, layer_index, expert_ids, expert_weights, hidden, hidden_size, output);
        if (gpu_result.ok) {
            return gpu_result;
        }

        llama_infinitum_moe_expert_mlp_result fallback_result =
            llama_infinitum_moe_execute_selected_experts_cpu_ptr(
                info, cache, layer_index, expert_ids, expert_weights, hidden, hidden_size, output, expert_workers);
        fallback_result.backend = backend == llama_infinitum_moe_backend_kind::fused_arena ?
            "fused_arena-fallback-simd" : "vulkan-fallback-simd";
        fallback_result.backend_fallback = true;
        if (!gpu_result.error.empty()) {
            fallback_result.error = gpu_result.error;
        }
        return fallback_result;
    }

    llama_infinitum_moe_expert_mlp_result result =
        llama_infinitum_moe_execute_selected_experts_cpu_ptr(
            info, cache, layer_index, expert_ids, expert_weights, hidden, hidden_size, output, expert_workers);
    result.backend = llama_infinitum_moe_backend_name(backend);
    return result;
}

llama_infinitum_moe_expert_mlp_result llama_infinitum_moe_execute_selected_experts(
        const llama_infinitum_moe_index_info & info,
        llama_infinitum_moe_slice_cache & cache,
        int layer_index,
        const std::vector<int> & expert_ids,
        const std::vector<float> & expert_weights,
        const std::vector<float> & hidden,
        int expert_workers) {
    std::vector<float> output(hidden.size(), 0.0f);
    llama_infinitum_moe_expert_mlp_result result =
        llama_infinitum_moe_execute_selected_experts_into(
            info,
            cache,
            layer_index,
            expert_ids,
            expert_weights,
            hidden.empty() ? nullptr : hidden.data(),
            static_cast<int>(hidden.size()),
            output.empty() ? nullptr : output.data(),
            expert_workers);
    result.output = std::move(output);
    return result;
}
