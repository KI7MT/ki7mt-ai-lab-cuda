/**
 * training_demo.cu - Signature Engine Training Demo
 *
 * Fetches WSPR spots from ClickHouse, computes embeddings on Blackwell GPU,
 * and writes results back to wspr.model_features table.
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
    printf("│  Phase 7: Vector Vault (Embedding Write-Back)               │\n");
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
    printf("  -w, --write           Write embeddings to wspr.model_features\n");
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
    bool write_back = false;

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
        } else if (arg == "-w" || arg == "--write") {
            write_back = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // -------------------------------------------------------------------------
    // 1. GPU Detection
    // -------------------------------------------------------------------------
    printf("[1/6] Detecting GPU...\n");

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
    printf("[2/6] Connecting to ClickHouse...\n");

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
    printf("[3/6] Fetching training batch...\n");
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
    printf("[4/6] Copying data to GPU...\n");

    size_t n = batch.size();
    size_t data_size = n * sizeof(float);
    size_t embedding_size = n * sizeof(float4);

    float *d_tx_lat, *d_tx_lon, *d_rx_lat, *d_rx_lon;
    float *d_kp, *d_xray, *d_distance;
    float4 *d_embedding;

    CUDA_CHECK(cudaMalloc(&d_tx_lat, data_size));
    CUDA_CHECK(cudaMalloc(&d_tx_lon, data_size));
    CUDA_CHECK(cudaMalloc(&d_rx_lat, data_size));
    CUDA_CHECK(cudaMalloc(&d_rx_lon, data_size));
    CUDA_CHECK(cudaMalloc(&d_kp, data_size));
    CUDA_CHECK(cudaMalloc(&d_xray, data_size));
    CUDA_CHECK(cudaMalloc(&d_distance, data_size));
    CUDA_CHECK(cudaMalloc(&d_embedding, embedding_size));

    CUDA_CHECK(cudaMemcpy(d_tx_lat, batch.tx_lat.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tx_lon, batch.tx_lon.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rx_lat, batch.rx_lat.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rx_lon, batch.rx_lon.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kp, batch.kp_index.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xray, batch.xray_flux.data(), data_size, cudaMemcpyHostToDevice));

    printf("      Allocated %.2f MB device memory\n",
           (6 * data_size + data_size + embedding_size) / (1024.0 * 1024.0));
    printf("      " GREEN "OK" RESET "\n\n");

    // -------------------------------------------------------------------------
    // 5. Execute Embedding Kernel
    // -------------------------------------------------------------------------
    printf("[5/6] Computing signature embeddings...\n");

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto kernel_start = std::chrono::high_resolution_clock::now();

    compute_signature_embedding<<<num_blocks, BLOCK_SIZE>>>(
        n,
        d_tx_lat, d_tx_lon,
        d_rx_lat, d_rx_lon,
        d_kp, d_xray,
        d_embedding, d_distance
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    auto kernel_end = std::chrono::high_resolution_clock::now();
    double kernel_time = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();

    printf("      Kernel time: %.3f ms\n", kernel_time);
    printf("      Throughput: %.2f M embeddings/sec\n", n / kernel_time / 1000.0);
    printf("      " GREEN "OK" RESET "\n\n");

    // -------------------------------------------------------------------------
    // Retrieve Results
    // -------------------------------------------------------------------------
    std::vector<float4> h_embedding(n);
    std::vector<float> h_distance(n);

    CUDA_CHECK(cudaMemcpy(h_embedding.data(), d_embedding, embedding_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_distance.data(), d_distance, data_size, cudaMemcpyDeviceToHost));

    // Convert to EmbeddingResult struct
    wspr::io::EmbeddingResult embeddings;
    embeddings.resize(n);
    for (size_t i = 0; i < n; i++) {
        embeddings.norm_distance[i] = h_embedding[i].x;
        embeddings.solar_penalty[i] = h_embedding[i].y;
        embeddings.geo_penalty[i] = h_embedding[i].z;
        embeddings.quality[i] = h_embedding[i].w;
        embeddings.distance_km[i] = h_distance[i];
    }

    // -------------------------------------------------------------------------
    // 6. Write Back to ClickHouse (optional)
    // -------------------------------------------------------------------------
    if (write_back) {
        printf("[6/6] Writing embeddings to wspr.model_features...\n");

        auto write_start = std::chrono::high_resolution_clock::now();

        size_t inserted = loader.insert_batch(batch, embeddings);

        auto write_end = std::chrono::high_resolution_clock::now();
        double write_time = std::chrono::duration<double>(write_end - write_start).count();

        if (inserted == 0) {
            fprintf(stderr, RED "ERROR: %s\n" RESET, loader.last_error().c_str());
            printf("      " YELLOW "SKIPPED" RESET " (table may not exist - run sql/01-model_features.sql)\n\n");
        } else {
            printf("      Inserted: %zu rows\n", inserted);
            printf("      Time: %.3f seconds (%.2f Krps)\n", write_time, inserted / write_time / 1000.0);
            printf("      " GREEN "OK" RESET "\n\n");
        }
    } else {
        printf("[6/6] " YELLOW "SKIPPED" RESET " (use -w to write embeddings)\n\n");
    }

    // -------------------------------------------------------------------------
    // Compute Statistics
    // -------------------------------------------------------------------------
    float sum_quality = 0, sum_distance = 0;
    float min_quality = 1.0f, max_quality = 0.0f;
    float sum_solar = 0, sum_geo = 0;

    for (size_t i = 0; i < n; i++) {
        sum_quality += embeddings.quality[i];
        sum_distance += embeddings.distance_km[i];
        sum_solar += embeddings.solar_penalty[i];
        sum_geo += embeddings.geo_penalty[i];
        if (embeddings.quality[i] < min_quality) min_quality = embeddings.quality[i];
        if (embeddings.quality[i] > max_quality) max_quality = embeddings.quality[i];
    }

    float avg_quality = sum_quality / n;
    float avg_distance = sum_distance / n;
    float avg_solar = sum_solar / n;
    float avg_geo = sum_geo / n;

    // Print statistics
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  Embedding Statistics                                       │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Paths processed:     %-37zu │\n", n);
    printf("│  Avg distance:        %-37.0f km │\n", avg_distance);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Embedding Vector Components (avg):                         │\n");
    printf("│    [0] Norm Distance: %-38.4f │\n", avg_distance / 20000.0f);
    printf("│    [1] Solar Penalty: %-38.4f │\n", avg_solar);
    printf("│    [2] Geo Penalty:   %-38.4f │\n", avg_geo);
    printf("│    [3] Quality:       %-38.4f │\n", avg_quality);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Quality range:       min=%.4f  max=%.4f                  │\n", min_quality, max_quality);
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("\n");

    // Quality distribution
    int q_buckets[10] = {0};
    for (size_t i = 0; i < n; i++) {
        int bucket = (int)(embeddings.quality[i] * 10);
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

    // Sample embeddings
    printf("Sample embeddings (first 5):\n");
    printf("  %-10s %-10s %-10s %-10s %-10s\n",
           "NormDist", "Solar", "Geo", "Quality", "DistKm");
    for (size_t i = 0; i < 5 && i < n; i++) {
        printf("  %-10.4f %-10.4f %-10.4f %-10.4f %-10.0f\n",
               embeddings.norm_distance[i],
               embeddings.solar_penalty[i],
               embeddings.geo_penalty[i],
               embeddings.quality[i],
               embeddings.distance_km[i]);
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
    cudaFree(d_distance);
    cudaFree(d_embedding);

    printf(GREEN "Training demo complete.\n" RESET);
    if (!write_back) {
        printf(YELLOW "Hint: Use -w flag to write embeddings to ClickHouse\n" RESET);
    }
    printf("\n");

    return 0;
}
