#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    6
#define RPC_PROTO_MINOR_VERSION    0
// [TAG_QSA_FUSED_SELECT] 0 -> 1: this fork inserts GGML_OP_QSA_TOP_K before GGML_OP_UNARY, so every
// op number from UNARY on is shifted by one relative to stock ggml, and rpc_tensor serialises
// tensor->op raw. The bump is what the assert below asks for, but be clear about what it does NOT
// buy: ggml_backend_rpc_buffer_type's handshake only rejects on major mismatch or a server minor
// ahead of the client (ggml-rpc.cpp:356), so the patch number is reported and logged, never
// enforced. A fork peer and a stock peer at major 6 / minor 0 still shake hands and would then
// disagree about every op past UNARY. Nothing here builds with GGML_RPC=ON, so this is a marker for
// whoever turns it on - if RPC ever ships from this fork the version policy needs a real decision
// (a major bump), not a patch.
#define RPC_PROTO_PATCH_VERSION    1

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 102, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
