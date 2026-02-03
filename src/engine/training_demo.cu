/**
 * training_demo.cu - Signature Engine Training Demo
 *
 * Fetches WSPR spots from ClickHouse, processes on Blackwell GPU,
 * and computes path quality signatures.
 *
 * Part of: ki7mt-ai-lab-cuda (Sovereign CUDA Engine)
 *
 * Copyright (c) 2026 KI7MT - Greg Beam
 * License: GPL-3.0-or-later
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "signature_kernel.cuh"
#include "clickhouse_loader.hpp"

// =============================================================================
// Configuration
// =============================================================================

#define DEFAULT_BATCH_SIZE 100000
#define BLOCK_SIZE 256

// ANSI colors
#define GREEN  "\033[32m"
#define RED    "\033[31m"
#define YELLOW "\033[33m"
#define CYAN   "\033[36m"
#define RESET  "\033[0m"

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
    printf("│  Signature Engine Training Demo - Blackwell sm_120          │\n");
    printf("│  ki7mt-ai-lab-cuda v2.0.6                                   │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("\n");
}

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\n");
    printf("Options:\n");
    printf("  -n, --batch-size N    Number of rows to fetch (default: %d)\n", DEFAULT_BATCH_SIZE);
    printf("  -s, --start DATE      Start date (YYYY-MM-DD)\n");
    printf("  -e, --end DATE        End date (YYYY-MM-DD)\n");
    printf("  -b, --band N          Band filter (ADIF band ID, 0 = all)\n");
    printf("  -h, --help            Show this help\n");
    printf("\n");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    print_header();

    // Parse arguments
    size_t batch_size = DEFAULT_BATCH_SIZE;
    std::string start_date = "";
    std::string end_date = "";
    int band = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--batch-size") && i + 1 < argc) {
            batch_size = std::stoul(argv[++i]);
        } else if ((arg == "-s" || arg == "--start") && i + 1 < argc) {
            start_date = argv[++i];
        } else if ((arg == "-e" || arg == "--end") && i + 1 < argc) {
            end_date = argv[++i];
        } else if ((arg == "-b" || arg == "--band") && i + 1 < argc) {
            band = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // -------------------------------------------------------------------------
    // 1. GPU Detection
    // -------------------------------------------------------------------------
    printf("[1/5] Detecting GPU...\n");

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
    printf("      " GREEN "OK" RESET "\n\n");

    // -------------------------------------------------------------------------
    // 2. Connect to ClickHouse
    // -------------------------------------------------------------------------
    printf("[2/5] Connecting to ClickHouse...\n");

    wspr::io::ConnectionConfig config;
    config.host = "localhost";
    config.port = 9000;
    config.database = "wspr";

    wspr::io::ClickHouseLoader loader(config);

    if (!loader.connect()) {
        fprintf(stderr, RED "ERROR: %s\n" RESET, loader.last_error().c_str());
        return 1;
    }

    printf("      Connected to %s:%d\n", config.host.c_str(), config.port);
    printf("      " GREEN "OK" RESET "\n\n");

    // -------------------------------------------------------------------------
    // 3. Fetch Training Batch
    // -------------------------------------------------------------------------
    printf("[3/5] Fetching training batch...\n");
    printf("      Batch size: %zu\n", batch_size);
    if (!start_date.empty()) printf("      Start date: %s\n", start_date.c_str());
    if (!end_date.empty()) printf("      End date: %s\n", end_date.c_str());
    if (band > 0) printf("      Band filter: %d\n", band);

    auto fetch_start = std::chrono::high_resolution_clock::now();

    wspr::io::TrainingBatch batch = loader.fetch_batch(batch_size, start_date, end_date, band);

    auto fetch_end = std::chrono::high_resolution_clock::now();
    double fetch_time = std::chrono::duration<double>(fetch_end - fetch_start).count();

    if (batch.empty()) {
        fprintf(stderr, RED "ERROR: No data fetched - %s\n" RESET, loader.last_error().c_str());
        return 1;
    }

    printf("      Fetched: %zu rows\n", batch.size());
    printf("      Time: %.3f seconds (%.2f Krps)\n", fetch_time, batch.size() / fetch_time / 1000.0);
    printf("      " GREEN "OK" RESET "\n\n");

    // -------------------------------------------------------------------------
    // 4. Copy to GPU
    // -------------------------------------------------------------------------
    printf("[4/5] Copying data to GPU...\n");

    size_t n = batch.size();
    size_t data_size = n * sizeof(float);

    float *d_tx_lat, *d_tx_lon, *d_rx_lat, *d_rx_lon;
    float *d_kp, *d_xray, *d_quality, *d_distance;

    CUDA_CHECK(cudaMalloc(&d_tx_lat, data_size));
    CUDA_CHECK(cudaMalloc(&d_tx_lon, data_size));
    CUDA_CHECK(cudaMalloc(&d_rx_lat, data_size));
    CUDA_CHECK(cudaMalloc(&d_rx_lon, data_size));
    CUDA_CHECK(cudaMalloc(&d_kp, data_size));
    CUDA_CHECK(cudaMalloc(&d_xray, data_size));
    CUDA_CHECK(cudaMalloc(&d_quality, data_size));
    CUDA_CHECK(cudaMalloc(&d_distance, data_size));

    CUDA_CHECK(cudaMemcpy(d_tx_lat, batch.tx_lat.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tx_lon, batch.tx_lon.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rx_lat, batch.rx_lat.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rx_lon, batch.rx_lon.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kp, batch.kp_index.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xray, batch.xray_flux.data(), data_size, cudaMemcpyHostToDevice));

    printf("      Allocated %.2f MB device memory\n", (8 * data_size) / (1024.0 * 1024.0));
    printf("      " GREEN "OK" RESET "\n\n");

    // -------------------------------------------------------------------------
    // 5. Execute Kernel
    // -------------------------------------------------------------------------
    printf("[5/5] Executing signature kernel...\n");

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto kernel_start = std::chrono::high_resolution_clock::now();

    compute_path_quality<<<num_blocks, BLOCK_SIZE>>>(
        n,
        d_tx_lat, d_tx_lon,
        d_rx_lat, d_rx_lon,
        d_kp, d_xray,
        d_quality, d_distance
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    auto kernel_end = std::chrono::high_resolution_clock::now();
    double kernel_time = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();

    printf("      Kernel time: %.3f ms\n", kernel_time);
    printf("      Throughput: %.2f M paths/sec\n", n / kernel_time / 1000.0);
    printf("      " GREEN "OK" RESET "\n\n");

    // -------------------------------------------------------------------------
    // Retrieve and Analyze Results
    // -------------------------------------------------------------------------
    std::vector<float> h_quality(n);
    std::vector<float> h_distance(n);

    CUDA_CHECK(cudaMemcpy(h_quality.data(), d_quality, data_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_distance.data(), d_distance, data_size, cudaMemcpyDeviceToHost));

    // Compute statistics
    float sum_quality = 0, sum_distance = 0;
    float min_quality = 1.0f, max_quality = 0.0f;
    float min_distance = 1e9f, max_distance = 0.0f;

    for (size_t i = 0; i < n; i++) {
        sum_quality += h_quality[i];
        sum_distance += h_distance[i];
        if (h_quality[i] < min_quality) min_quality = h_quality[i];
        if (h_quality[i] > max_quality) max_quality = h_quality[i];
        if (h_distance[i] < min_distance) min_distance = h_distance[i];
        if (h_distance[i] > max_distance) max_distance = h_distance[i];
    }

    float avg_quality = sum_quality / n;
    float avg_distance = sum_distance / n;

    // Print statistics
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  Results Summary                                            │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Paths processed:  %-40zu │\n", n);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Distance (km):    min=%-8.0f  avg=%-8.0f  max=%-8.0f  │\n",
           min_distance, avg_distance, max_distance);
    printf("│  Quality (0-1):    min=%-8.4f  avg=%-8.4f  max=%-8.4f  │\n",
           min_quality, avg_quality, max_quality);
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("\n");

    // Quality distribution
    int q_buckets[10] = {0};
    for (size_t i = 0; i < n; i++) {
        int bucket = (int)(h_quality[i] * 10);
        if (bucket >= 10) bucket = 9;
        q_buckets[bucket]++;
    }

    printf("Quality Distribution:\n");
    for (int b = 0; b < 10; b++) {
        float pct = 100.0f * q_buckets[b] / n;
        printf("  %.1f-%.1f: ", b * 0.1f, (b + 1) * 0.1f);
        int bar_len = (int)(pct / 2);
        for (int j = 0; j < bar_len; j++) printf("█");
        printf(" %.1f%% (%d)\n", pct, q_buckets[b]);
    }
    printf("\n");

    // Sample output
    printf("Sample paths (first 5):\n");
    printf("  %-12s %-12s %-10s %-10s %-8s\n", "TX Lat", "TX Lon", "Distance", "Kp", "Quality");
    for (size_t i = 0; i < 5 && i < n; i++) {
        printf("  %-12.4f %-12.4f %-10.0f %-10.2f %-8.4f\n",
               batch.tx_lat[i], batch.tx_lon[i],
               h_distance[i], batch.kp_index[i], h_quality[i]);
    }
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

    printf(GREEN "Training demo complete.\n" RESET);
    printf("\n");

    return 0;
}
