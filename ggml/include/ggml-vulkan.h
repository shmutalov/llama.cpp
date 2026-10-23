#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

// ── Keryx zero-dup exports: expose resident weight tensors to an external
// same-process Vulkan compute kernel (see ggml-vulkan.cpp for details) ──────
GGML_BACKEND_API bool ggml_backend_vk_keryx_raw_handles(size_t dev_num, void ** vk_instance,
                                                        void ** vk_physical_device, void ** vk_device,
                                                        uint32_t * compute_queue_family_index);
GGML_BACKEND_API bool ggml_backend_vk_keryx_tensor_addr(const struct ggml_tensor * tensor,
                                                        uint64_t * gpu_addr, uint64_t * size_bytes);
GGML_BACKEND_API void ggml_backend_vk_keryx_queue_submit(size_t dev_num, const void * submit_info, void * fence);

// ── Keryx canonical trace (verifiable inference): BLAKE3 leaves of the q8_1 inputs and the outputs
// of every manifest product, and of the layer residuals, per traced position. Evaluations under
// trace must feed batches of at most 8 tokens and announce each with _ubatch before its decode.
// Every call returns 0 on success, a negative code otherwise (see ggml-vulkan.cpp). ─────────────
GGML_BACKEND_API int  ggml_backend_vk_keryx_trace_configure(size_t dev_num, const void * manifest, size_t manifest_len, uint32_t max_pos);
GGML_BACKEND_API void ggml_backend_vk_keryx_trace_release(size_t dev_num);
// Leaves per position (or max positions) of the configured trace, -1 when none is configured.
GGML_BACKEND_API int  ggml_backend_vk_keryx_trace_leaf_count(size_t dev_num);
GGML_BACKEND_API int  ggml_backend_vk_keryx_trace_max_pos(size_t dev_num);
GGML_BACKEND_API int  ggml_backend_vk_keryx_trace_begin(size_t dev_num);
GGML_BACKEND_API int  ggml_backend_vk_keryx_trace_ubatch(size_t dev_num, uint32_t pos_base, uint32_t n_tokens);
GGML_BACKEND_API int  ggml_backend_vk_keryx_trace_end(size_t dev_num);
GGML_BACKEND_API int  ggml_backend_vk_keryx_trace_read(size_t dev_num, uint8_t * leaves, size_t leaves_cap, uint32_t * ubatch,
                                                       size_t ubatch_cap, uint32_t * n_pos);
GGML_BACKEND_API int  ggml_backend_vk_keryx_trace_arm_captures(size_t dev_num, const uint32_t * pos, const uint32_t * leaf, uint32_t n,
                                                               uint32_t slot_bytes);
GGML_BACKEND_API int  ggml_backend_vk_keryx_trace_read_captures(size_t dev_num, uint8_t * out, size_t out_cap, uint32_t * lens, uint32_t n);
GGML_BACKEND_API const char * ggml_backend_vk_keryx_trace_error(size_t dev_num);


#ifdef  __cplusplus
}
#endif
