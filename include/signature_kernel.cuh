/**
 * signature_kernel.cuh - WSPR Propagation Signature Engine Header
 *
 * GPU-accelerated path quality computation for propagation prediction.
 *
 * Part of: ki7mt-ai-lab-cuda (Sovereign CUDA Engine)
 * Target:  Blackwell sm_120 (RTX PRO 6000)
 *
 * Copyright (c) 2026 KI7MT - Greg Beam
 * License: GPL-3.0-or-later
 */

#ifndef WSPR_SIGNATURE_KERNEL_CUH
#define WSPR_SIGNATURE_KERNEL_CUH

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Kernel Declarations (for direct invocation)
// =============================================================================

/**
 * Compute path quality with full solar context
 *
 * @param n            Number of paths
 * @param tx_lat       Transmitter latitudes (degrees)
 * @param tx_lon       Transmitter longitudes (degrees)
 * @param rx_lat       Receiver latitudes (degrees)
 * @param rx_lon       Receiver longitudes (degrees)
 * @param solar_kp     Kp index at spot timestamp
 * @param solar_xray   X-ray flux at spot timestamp (W/m²)
 * @param out_quality  Output quality scores (0.0 - 1.0)
 * @param out_distance Output distances in km (optional, can be NULL)
 */
__global__ void compute_path_quality(
    int n,
    const float* __restrict__ tx_lat,
    const float* __restrict__ tx_lon,
    const float* __restrict__ rx_lat,
    const float* __restrict__ rx_lon,
    const float* __restrict__ solar_kp,
    const float* __restrict__ solar_xray,
    float* __restrict__ out_quality,
    float* __restrict__ out_distance
);

/**
 * Simplified kernel: Kp penalty only
 */
__global__ void compute_path_quality_simple(
    int n,
    const float* __restrict__ tx_lat,
    const float* __restrict__ tx_lon,
    const float* __restrict__ rx_lat,
    const float* __restrict__ rx_lon,
    const float* __restrict__ solar_k_index,
    float* __restrict__ out_quality
);

/**
 * Compute propagation signature embeddings (float4 vector output)
 *
 * Output embedding components:
 *   x: Normalized distance (0-1)
 *   y: Solar penalty factor (X-ray impact)
 *   z: Geomagnetic penalty factor (Kp impact)
 *   w: Final combined quality score
 *
 * @param n              Number of paths
 * @param tx_lat         Transmitter latitudes (degrees)
 * @param tx_lon         Transmitter longitudes (degrees)
 * @param rx_lat         Receiver latitudes (degrees)
 * @param rx_lon         Receiver longitudes (degrees)
 * @param solar_kp       Kp index at spot timestamp
 * @param solar_xray     X-ray flux at spot timestamp (W/m²)
 * @param out_embedding  Output float4 embeddings
 * @param out_distance   Output distances in km (optional, can be NULL)
 */
__global__ void compute_signature_embedding(
    int n,
    const float* __restrict__ tx_lat,
    const float* __restrict__ tx_lon,
    const float* __restrict__ rx_lat,
    const float* __restrict__ rx_lon,
    const float* __restrict__ solar_kp,
    const float* __restrict__ solar_xray,
    float4* __restrict__ out_embedding,
    float* __restrict__ out_distance
);

#ifdef __cplusplus
}
#endif

// =============================================================================
// C++ Host API (namespace wspr::signature)
// =============================================================================

#ifdef __cplusplus

namespace wspr {
namespace signature {

/**
 * Launch path quality kernel
 *
 * @param n           Number of paths
 * @param d_tx_lat    Device pointer: TX latitudes
 * @param d_tx_lon    Device pointer: TX longitudes
 * @param d_rx_lat    Device pointer: RX latitudes
 * @param d_rx_lon    Device pointer: RX longitudes
 * @param d_kp        Device pointer: Kp indices
 * @param d_xray      Device pointer: X-ray flux (can be nullptr for simple mode)
 * @param d_quality   Device pointer: Output quality scores
 * @param d_distance  Device pointer: Output distances (can be nullptr)
 * @param stream      CUDA stream for async execution (default: 0)
 */
void launch_path_quality(
    int n,
    const float* d_tx_lat,
    const float* d_tx_lon,
    const float* d_rx_lat,
    const float* d_rx_lon,
    const float* d_kp,
    const float* d_xray,
    float* d_quality,
    float* d_distance = nullptr,
    cudaStream_t stream = 0
);

// Constants exposed for host-side calculations
constexpr float EARTH_RADIUS_KM = 6371.0f;
constexpr float KP_QUIET_MAX = 2.0f;
constexpr float KP_UNSETTLED_MAX = 4.0f;
constexpr float XRAY_BLACKOUT_THRESHOLD = 1.0e-5f;

} // namespace signature
} // namespace wspr

#endif // __cplusplus

#endif // WSPR_SIGNATURE_KERNEL_CUH
