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


#ifdef  __cplusplus
}
#endif
