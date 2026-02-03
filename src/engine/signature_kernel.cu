/**
 * signature_kernel.cu - WSPR Propagation Signature Engine
 *
 * GPU-accelerated path quality computation for propagation prediction.
 * Combines great circle distance with geomagnetic conditions to produce
 * a "path quality" score that can be used for signature matching.
 *
 * Part of: ki7mt-ai-lab-cuda (Sovereign CUDA Engine)
 * Target:  Blackwell sm_120 (RTX PRO 6000)
 *
 * Mathematical Model:
 *   1. Haversine formula for great circle distance
 *   2. Geomagnetic penalty based on Kp index (0-9 scale)
 *   3. X-ray flux penalty for D-layer absorption events
 *
 * Copyright (c) 2026 KI7MT - Greg Beam
 * License: GPL-3.0-or-later
 */

#include <cuda_runtime.h>
#include <math.h>
#include "signature_kernel.cuh"

// =============================================================================
// Constants
// =============================================================================

// Earth radius in kilometers
constexpr float EARTH_RADIUS_KM = 6371.0f;

// Degrees to radians conversion
constexpr float DEG_TO_RAD = 3.14159265358979323846f / 180.0f;

// Kp index thresholds for geomagnetic conditions
// Kp 0-2: Quiet, Kp 3-4: Unsettled, Kp 5+: Storm
constexpr float KP_QUIET_MAX = 2.0f;
constexpr float KP_UNSETTLED_MAX = 4.0f;

// X-ray flux threshold for radio blackout (W/m², M-class flare)
constexpr float XRAY_BLACKOUT_THRESHOLD = 1.0e-5f;

// =============================================================================
// Device Functions
// =============================================================================

/**
 * Haversine formula for great circle distance
 *
 * @param lat1, lon1  Transmitter coordinates (degrees)
 * @param lat2, lon2  Receiver coordinates (degrees)
 * @return Distance in kilometers
 */
__device__ __forceinline__
float haversine_distance(float lat1, float lon1, float lat2, float lon2) {
    // Convert to radians
    float phi1 = lat1 * DEG_TO_RAD;
    float phi2 = lat2 * DEG_TO_RAD;
    float delta_phi = (lat2 - lat1) * DEG_TO_RAD;
    float delta_lambda = (lon2 - lon1) * DEG_TO_RAD;

    // Haversine formula
    float sin_dphi = sinf(delta_phi * 0.5f);
    float sin_dlam = sinf(delta_lambda * 0.5f);

    float a = sin_dphi * sin_dphi +
              cosf(phi1) * cosf(phi2) * sin_dlam * sin_dlam;

    float c = 2.0f * atan2f(sqrtf(a), sqrtf(1.0f - a));

    return EARTH_RADIUS_KM * c;
}

/**
 * Geomagnetic penalty factor based on Kp index
 *
 * Returns a multiplier (0.0 - 1.0) where:
 *   1.0 = Quiet conditions (Kp 0-2), no penalty
 *   0.5 = Unsettled (Kp 3-4), 50% penalty
 *   0.0 = Storm (Kp 5+), severe degradation
 *
 * @param kp  Planetary K-index (0-9)
 * @return Penalty multiplier
 */
__device__ __forceinline__
float geomagnetic_penalty(float kp) {
    if (kp <= KP_QUIET_MAX) {
        // Quiet: No penalty
        return 1.0f;
    } else if (kp <= KP_UNSETTLED_MAX) {
        // Unsettled: Linear degradation from 1.0 to 0.5
        return 1.0f - 0.25f * (kp - KP_QUIET_MAX);
    } else {
        // Storm: Exponential degradation
        // At Kp=5: 0.5, At Kp=7: ~0.12, At Kp=9: ~0.03
        return 0.5f * expf(-0.5f * (kp - KP_UNSETTLED_MAX));
    }
}

/**
 * X-ray flux penalty for D-layer absorption
 *
 * High X-ray flux causes D-layer ionization which absorbs HF signals.
 * Returns a multiplier (0.0 - 1.0).
 *
 * @param xray_flux  X-ray flux in W/m² (0.1-0.8nm band)
 * @return Penalty multiplier
 */
__device__ __forceinline__
float xray_penalty(float xray_flux) {
    if (xray_flux < 1.0e-6f) {
        // A-class or below: No penalty
        return 1.0f;
    } else if (xray_flux < 1.0e-5f) {
        // B/C-class: Minor degradation
        // Log scale interpolation
        float log_flux = log10f(xray_flux);
        return 1.0f - 0.1f * (log_flux + 6.0f); // -6 to -5 -> 0% to 10%
    } else {
        // M-class or above: Significant blackout
        // Exponential penalty
        float excess = xray_flux / XRAY_BLACKOUT_THRESHOLD;
        return fmaxf(0.1f, expf(-0.5f * excess));
    }
}

/**
 * Distance-based quality factor
 *
 * Normalizes distance to a 0-1 quality score.
 * Optimal WSPR distance is ~2000-5000 km for F2 propagation.
 *
 * @param distance_km  Great circle distance in km
 * @return Quality factor (0.0 - 1.0)
 */
__device__ __forceinline__
float distance_quality(float distance_km) {
    // Too short: Ground wave / E-skip, less interesting
    if (distance_km < 500.0f) {
        return 0.3f + 0.7f * (distance_km / 500.0f);
    }
    // Optimal F2 range: 2000-5000 km
    else if (distance_km < 2000.0f) {
        return 0.7f + 0.3f * ((distance_km - 500.0f) / 1500.0f);
    }
    else if (distance_km <= 5000.0f) {
        return 1.0f; // Peak quality
    }
    // Long path: Quality decreases with distance
    else if (distance_km <= 15000.0f) {
        return 1.0f - 0.3f * ((distance_km - 5000.0f) / 10000.0f);
    }
    // Very long path (approaching antipodal)
    else {
        return 0.7f - 0.2f * fminf(1.0f, (distance_km - 15000.0f) / 5000.0f);
    }
}

// =============================================================================
// Kernels
// =============================================================================

/**
 * Compute path quality scores for WSPR paths
 *
 * Combines distance, geomagnetic, and X-ray factors into a single
 * quality score (0.0 - 1.0) for each path.
 *
 * @param n            Number of paths to process
 * @param tx_lat       Transmitter latitudes (degrees)
 * @param tx_lon       Transmitter longitudes (degrees)
 * @param rx_lat       Receiver latitudes (degrees)
 * @param rx_lon       Receiver longitudes (degrees)
 * @param solar_kp     Kp index at spot timestamp
 * @param solar_xray   X-ray flux at spot timestamp (W/m², 0.1-0.8nm)
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Calculate great circle distance
    float dist = haversine_distance(
        tx_lat[idx], tx_lon[idx],
        rx_lat[idx], rx_lon[idx]
    );

    // Calculate quality factors
    float q_dist = distance_quality(dist);
    float q_geo = geomagnetic_penalty(solar_kp[idx]);
    float q_xray = xray_penalty(solar_xray[idx]);

    // Combined quality score
    // Multiplicative model: all factors must be good for high quality
    float quality = q_dist * q_geo * q_xray;

    // Output
    out_quality[idx] = quality;

    if (out_distance != nullptr) {
        out_distance[idx] = dist;
    }
}

/**
 * Compute propagation signature embeddings for WSPR paths
 *
 * Outputs a float4 vector per path for ML/k-NN search:
 *   x: Normalized distance (0-1, where 1 = 20000km max)
 *   y: Solar penalty factor (X-ray/D-layer absorption impact)
 *   z: Geomagnetic penalty factor (Kp storm impact)
 *   w: Final combined path quality score
 *
 * This decomposition allows searching for specific phenomena:
 *   - High solar impact events (low y, normal z)
 *   - Geomagnetic storm effects (normal y, low z)
 *   - Combined degradation (low y and z)
 *
 * @param n            Number of paths to process
 * @param tx_lat       Transmitter latitudes (degrees)
 * @param tx_lon       Transmitter longitudes (degrees)
 * @param rx_lat       Receiver latitudes (degrees)
 * @param rx_lon       Receiver longitudes (degrees)
 * @param solar_kp     Kp index at spot timestamp
 * @param solar_xray   X-ray flux at spot timestamp (W/m², 0.1-0.8nm)
 * @param out_embedding Output float4 embeddings
 * @param out_distance Output distances in km (optional, can be NULL)
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Calculate great circle distance
    float dist = haversine_distance(
        tx_lat[idx], tx_lon[idx],
        rx_lat[idx], rx_lon[idx]
    );

    // Calculate individual penalty factors
    float q_dist = distance_quality(dist);
    float q_xray = xray_penalty(solar_xray[idx]);
    float q_geo = geomagnetic_penalty(solar_kp[idx]);

    // Combined quality score
    float quality = q_dist * q_xray * q_geo;

    // Normalize distance to 0-1 (max ~20000km for antipodal)
    float norm_dist = fminf(dist / 20000.0f, 1.0f);

    // Output embedding vector
    out_embedding[idx] = make_float4(
        norm_dist,    // x: Normalized distance
        q_xray,       // y: Solar/X-ray penalty (1.0 = no impact, 0.0 = blackout)
        q_geo,        // z: Geomagnetic penalty (1.0 = quiet, 0.0 = severe storm)
        quality       // w: Final combined quality
    );

    if (out_distance != nullptr) {
        out_distance[idx] = dist;
    }
}

/**
 * Simplified kernel for basic path quality (Kp only)
 *
 * Matches Gemini's original prototype signature.
 */
__global__ void compute_path_quality_simple(
    int n,
    const float* __restrict__ tx_lat,
    const float* __restrict__ tx_lon,
    const float* __restrict__ rx_lat,
    const float* __restrict__ rx_lon,
    const float* __restrict__ solar_k_index,
    float* __restrict__ out_quality
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Calculate great circle distance
    float dist = haversine_distance(
        tx_lat[idx], tx_lon[idx],
        rx_lat[idx], rx_lon[idx]
    );

    // Distance quality
    float q_dist = distance_quality(dist);

    // Geomagnetic penalty
    float q_geo = geomagnetic_penalty(solar_k_index[idx]);

    // Combined quality
    out_quality[idx] = q_dist * q_geo;
}

// =============================================================================
// Host API
// =============================================================================

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
 * @param d_xray      Device pointer: X-ray flux (can be nullptr)
 * @param d_quality   Device pointer: Output quality scores
 * @param d_distance  Device pointer: Output distances (can be nullptr)
 * @param stream      CUDA stream for async execution
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
    float* d_distance,
    cudaStream_t stream
) {
    // Blackwell-optimized: 256 threads per block
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (d_xray != nullptr) {
        compute_path_quality<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            n, d_tx_lat, d_tx_lon, d_rx_lat, d_rx_lon,
            d_kp, d_xray, d_quality, d_distance
        );
    } else {
        compute_path_quality_simple<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            n, d_tx_lat, d_tx_lon, d_rx_lat, d_rx_lon,
            d_kp, d_quality
        );
    }
}

} // namespace signature
} // namespace wspr
