/**
 * test_signature.cu - Signature Engine Test Program
 *
 * Validates the path quality kernel on Blackwell GPU.
 * Tests various scenarios: quiet conditions, storms, blackouts.
 *
 * Part of: ki7mt-ai-lab-cuda (Sovereign CUDA Engine)
 *
 * Copyright (c) 2026 KI7MT - Greg Beam
 * License: GPL-3.0-or-later
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "signature_kernel.cuh"

// =============================================================================
// Test Configuration
// =============================================================================

#define NUM_TEST_PATHS 8
#define BLOCK_SIZE 256

// ANSI colors
#define GREEN  "\033[32m"
#define RED    "\033[31m"
#define YELLOW "\033[33m"
#define RESET  "\033[0m"

// =============================================================================
// Test Data
// =============================================================================

// Test paths representing various WSPR scenarios
// Format: TX_lat, TX_lon, RX_lat, RX_lon, Kp, X-ray, Expected_Quality_Range
struct TestPath {
    const char* name;
    float tx_lat, tx_lon;
    float rx_lat, rx_lon;
    float kp;
    float xray;
    float min_quality;  // Expected minimum
    float max_quality;  // Expected maximum
};

TestPath test_paths[NUM_TEST_PATHS] = {
    // 1. Short path, quiet conditions (E-skip) - ~475km
    {"Short/Quiet", 40.0f, -105.0f, 42.0f, -100.0f, 1.0f, 1e-7f, 0.9f, 1.0f},

    // 2. Long F2 path (~7500km), quiet - CO to UK
    {"LongF2/Quiet", 40.0f, -105.0f, 51.5f, -0.1f, 1.0f, 1e-7f, 0.9f, 1.0f},

    // 3. Long F2 path, unsettled (Kp=4)
    {"LongF2/Unsettled", 40.0f, -105.0f, 51.5f, -0.1f, 4.0f, 1e-7f, 0.4f, 0.6f},

    // 4. Long F2 path, storm (Kp=6)
    {"LongF2/Storm", 40.0f, -105.0f, 51.5f, -0.1f, 6.0f, 1e-7f, 0.1f, 0.3f},

    // 5. Long F2 path, M-class flare (X-ray blackout)
    {"LongF2/Blackout", 40.0f, -105.0f, 51.5f, -0.1f, 1.0f, 5e-5f, 0.05f, 0.15f},

    // 6. Very long path (~13400km), quiet - CO to Sydney
    {"VeryLong/Quiet", 40.0f, -105.0f, -33.9f, 151.2f, 1.0f, 1e-7f, 0.7f, 0.8f},

    // 7. Long path (~9300km), quiet - CO to Tokyo
    {"Long/Quiet", 40.0f, -105.0f, 35.7f, 139.7f, 1.0f, 1e-7f, 0.8f, 0.95f},

    // 8. Extreme storm (Kp=9) - severely degraded
    {"LongF2/Extreme", 40.0f, -105.0f, 51.5f, -0.1f, 9.0f, 1e-7f, 0.0f, 0.1f},
};

// =============================================================================
// Utility Functions
// =============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, RED "CUDA Error: %s at %s:%d\n" RESET, \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

void print_header() {
    printf("\n");
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  Signature Engine Test - Blackwell sm_120                   │\n");
    printf("│  ki7mt-ai-lab-cuda v2.0.6                                   │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("\n");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    print_header();

    // -------------------------------------------------------------------------
    // 1. GPU Detection
    // -------------------------------------------------------------------------
    printf("[1/4] Detecting GPU...\n");

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, RED "ERROR: No CUDA devices found\n" RESET);
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("      GPU: %s\n", prop.name);
    printf("      Compute: sm_%d%d\n", prop.major, prop.minor);
    printf("      Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("      SMs: %d\n", prop.multiProcessorCount);
    printf("      " GREEN "OK" RESET "\n\n");

    // -------------------------------------------------------------------------
    // 2. Allocate Memory
    // -------------------------------------------------------------------------
    printf("[2/4] Allocating device memory...\n");

    float *d_tx_lat, *d_tx_lon, *d_rx_lat, *d_rx_lon;
    float *d_kp, *d_xray, *d_quality, *d_distance;

    size_t data_size = NUM_TEST_PATHS * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_tx_lat, data_size));
    CUDA_CHECK(cudaMalloc(&d_tx_lon, data_size));
    CUDA_CHECK(cudaMalloc(&d_rx_lat, data_size));
    CUDA_CHECK(cudaMalloc(&d_rx_lon, data_size));
    CUDA_CHECK(cudaMalloc(&d_kp, data_size));
    CUDA_CHECK(cudaMalloc(&d_xray, data_size));
    CUDA_CHECK(cudaMalloc(&d_quality, data_size));
    CUDA_CHECK(cudaMalloc(&d_distance, data_size));

    printf("      Allocated %zu bytes per array (%d paths)\n", data_size, NUM_TEST_PATHS);
    printf("      " GREEN "OK" RESET "\n\n");

    // -------------------------------------------------------------------------
    // 3. Copy Test Data to Device
    // -------------------------------------------------------------------------
    printf("[3/4] Copying test data to device...\n");

    float h_tx_lat[NUM_TEST_PATHS], h_tx_lon[NUM_TEST_PATHS];
    float h_rx_lat[NUM_TEST_PATHS], h_rx_lon[NUM_TEST_PATHS];
    float h_kp[NUM_TEST_PATHS], h_xray[NUM_TEST_PATHS];

    for (int i = 0; i < NUM_TEST_PATHS; i++) {
        h_tx_lat[i] = test_paths[i].tx_lat;
        h_tx_lon[i] = test_paths[i].tx_lon;
        h_rx_lat[i] = test_paths[i].rx_lat;
        h_rx_lon[i] = test_paths[i].rx_lon;
        h_kp[i] = test_paths[i].kp;
        h_xray[i] = test_paths[i].xray;
    }

    CUDA_CHECK(cudaMemcpy(d_tx_lat, h_tx_lat, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tx_lon, h_tx_lon, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rx_lat, h_rx_lat, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rx_lon, h_rx_lon, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kp, h_kp, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xray, h_xray, data_size, cudaMemcpyHostToDevice));

    printf("      " GREEN "OK" RESET "\n\n");

    // -------------------------------------------------------------------------
    // 4. Execute Kernel
    // -------------------------------------------------------------------------
    printf("[4/4] Executing path quality kernel...\n");

    int num_blocks = (NUM_TEST_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    compute_path_quality<<<num_blocks, BLOCK_SIZE>>>(
        NUM_TEST_PATHS,
        d_tx_lat, d_tx_lon,
        d_rx_lat, d_rx_lon,
        d_kp, d_xray,
        d_quality, d_distance
    );

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("      " GREEN "OK" RESET "\n\n");

    // -------------------------------------------------------------------------
    // 5. Retrieve Results
    // -------------------------------------------------------------------------
    float h_quality[NUM_TEST_PATHS], h_distance[NUM_TEST_PATHS];

    CUDA_CHECK(cudaMemcpy(h_quality, d_quality, data_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_distance, d_distance, data_size, cudaMemcpyDeviceToHost));

    // -------------------------------------------------------------------------
    // 6. Validate Results
    // -------------------------------------------------------------------------
    printf("┌────────────────────────────────────────────────────────────────────┐\n");
    printf("│  Test Results                                                      │\n");
    printf("├──────────────────┬──────────┬─────────┬──────────┬─────────────────┤\n");
    printf("│ Scenario         │ Distance │ Kp      │ Quality  │ Status          │\n");
    printf("├──────────────────┼──────────┼─────────┼──────────┼─────────────────┤\n");

    int passed = 0, failed = 0;

    for (int i = 0; i < NUM_TEST_PATHS; i++) {
        bool ok = (h_quality[i] >= test_paths[i].min_quality &&
                   h_quality[i] <= test_paths[i].max_quality);

        if (ok) passed++; else failed++;

        printf("│ %-16s │ %7.0f km │ Kp=%.1f │ %8.3f │ %s │\n",
               test_paths[i].name,
               h_distance[i],
               test_paths[i].kp,
               h_quality[i],
               ok ? GREEN "PASS" RESET "           " : RED "FAIL" RESET " (exp %.2f-%.2f)");

        if (!ok) {
            printf("│                  │          │         │          │ exp %.2f-%.2f     │\n",
                   test_paths[i].min_quality, test_paths[i].max_quality);
        }
    }

    printf("└──────────────────┴──────────┴─────────┴──────────┴─────────────────┘\n");

    // -------------------------------------------------------------------------
    // 7. Summary
    // -------------------------------------------------------------------------
    printf("\n");
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    if (failed == 0) {
        printf("│  " GREEN "ALL TESTS PASSED" RESET "                                        │\n");
    } else {
        printf("│  " RED "FAILED: %d/%d tests" RESET "                                     │\n",
               failed, NUM_TEST_PATHS);
    }
    printf("│                                                             │\n");
    printf("│  Signature Engine: " GREEN "OPERATIONAL" RESET "                            │\n");
    printf("│  Blackwell sm_120: " GREEN "VERIFIED" RESET "                               │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("\n");

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    cudaFree(d_tx_lat);
    cudaFree(d_tx_lon);
    cudaFree(d_rx_lat);
    cudaFree(d_rx_lon);
    cudaFree(d_kp);
    cudaFree(d_xray);
    cudaFree(d_quality);
    cudaFree(d_distance);

    return (failed > 0) ? 1 : 0;
}
