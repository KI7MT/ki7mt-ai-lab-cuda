/**
 * bulk_processor.cu - Bulk Embedding Processor for Full Dataset
 *
 * Processes WSPR spots in time-based chunks, computing embeddings on
 * Blackwell GPU and writing to wspr.model_features for ML training.
 *
 * Features:
 *   - Time-based iteration (hourly or daily)
 *   - Dynamic chunking for large batches (>1M rows)
 *   - Resilient: continues on batch failures
 *   - Progress reporting with throughput stats
 *
 * Part of: ki7mt-ai-lab-cuda (Sovereign CUDA Engine)
 *
 * Copyright (c) 2026 KI7MT - Greg Beam
 * License: GPL-3.0-or-later
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include "signature_kernel.cuh"
#include "clickhouse_loader.hpp"

// =============================================================================
// Configuration
// =============================================================================

#define MAX_BATCH_SIZE 1000000   // 1M rows per GPU batch (memory safety)
#define MIN_BATCH_SIZE 10000     // Don't process tiny batches
#define BLOCK_SIZE 256

// ANSI colors
#define GREEN   "\033[32m"
#define RED     "\033[31m"
#define YELLOW  "\033[33m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"
#define RESET   "\033[0m"

// =============================================================================
// Utility Functions
// =============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, RED "CUDA Error: %s at %s:%d\n" RESET, \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

void print_header() {
    printf("\n");
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  " BOLD "Bulk Embedding Processor" RESET " - Blackwell sm_120              │\n");
    printf("│  ki7mt-ai-lab-cuda v2.0.8                                   │\n");
    printf("│  Phase 8: The Big Crunch                                    │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("\n");
}

void print_usage(const char* prog) {
    printf("Usage: %s --start YYYY-MM-DD --end YYYY-MM-DD [options]\n", prog);
    printf("\n");
    printf("Required:\n");
    printf("  --start DATE      Start date (inclusive)\n");
    printf("  --end DATE        End date (inclusive)\n");
    printf("\n");
    printf("Options:\n");
    printf("  --hourly          Process hour-by-hour (default: daily)\n");
    printf("  --band N          Filter by ADIF band ID (0 = all bands)\n");
    printf("  --dry-run         Count rows only, don't process\n");
    printf("  --batch-size N    Max rows per GPU batch (default: %d)\n", MAX_BATCH_SIZE);
    printf("  -h, --help        Show this help\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s --start 2025-01-01 --end 2025-12-31 --hourly\n", prog);
    printf("\n");
}

// Parse date string to time_t
time_t parse_date(const std::string& date_str) {
    struct tm tm = {};
    if (sscanf(date_str.c_str(), "%d-%d-%d", &tm.tm_year, &tm.tm_mon, &tm.tm_mday) != 3) {
        return -1;
    }
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    tm.tm_hour = 0;
    tm.tm_min = 0;
    tm.tm_sec = 0;
    return mktime(&tm);
}

// Format time_t to string
std::string format_datetime(time_t t, bool include_hour = false) {
    struct tm* tm = localtime(&t);
    char buf[32];
    if (include_hour) {
        strftime(buf, sizeof(buf), "%Y-%m-%d %H:00", tm);
    } else {
        strftime(buf, sizeof(buf), "%Y-%m-%d", tm);
    }
    return std::string(buf);
}

// Format duration
std::string format_duration(double seconds) {
    if (seconds < 60) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.1fs", seconds);
        return buf;
    } else if (seconds < 3600) {
        int mins = (int)(seconds / 60);
        int secs = (int)seconds % 60;
        char buf[32];
        snprintf(buf, sizeof(buf), "%dm%ds", mins, secs);
        return buf;
    } else {
        int hours = (int)(seconds / 3600);
        int mins = ((int)seconds % 3600) / 60;
        char buf[32];
        snprintf(buf, sizeof(buf), "%dh%dm", hours, mins);
        return buf;
    }
}

// =============================================================================
// GPU Processing
// =============================================================================

struct GPUContext {
    float *d_tx_lat, *d_tx_lon, *d_rx_lat, *d_rx_lon;
    float *d_kp, *d_xray, *d_distance;
    float4 *d_embedding;
    size_t allocated_size;
    bool initialized;

    GPUContext() : allocated_size(0), initialized(false) {}

    bool allocate(size_t n) {
        if (n <= allocated_size && initialized) {
            return true;  // Already have enough memory
        }

        // Free existing if any
        free();

        size_t data_size = n * sizeof(float);
        size_t embedding_size = n * sizeof(float4);

        CUDA_CHECK(cudaMalloc(&d_tx_lat, data_size));
        CUDA_CHECK(cudaMalloc(&d_tx_lon, data_size));
        CUDA_CHECK(cudaMalloc(&d_rx_lat, data_size));
        CUDA_CHECK(cudaMalloc(&d_rx_lon, data_size));
        CUDA_CHECK(cudaMalloc(&d_kp, data_size));
        CUDA_CHECK(cudaMalloc(&d_xray, data_size));
        CUDA_CHECK(cudaMalloc(&d_distance, data_size));
        CUDA_CHECK(cudaMalloc(&d_embedding, embedding_size));

        allocated_size = n;
        initialized = true;
        return true;
    }

    void free() {
        if (initialized) {
            cudaFree(d_tx_lat);
            cudaFree(d_tx_lon);
            cudaFree(d_rx_lat);
            cudaFree(d_rx_lon);
            cudaFree(d_kp);
            cudaFree(d_xray);
            cudaFree(d_distance);
            cudaFree(d_embedding);
            initialized = false;
            allocated_size = 0;
        }
    }

    ~GPUContext() { free(); }
};

bool process_batch(
    GPUContext& gpu,
    wspr::io::ClickHouseLoader& loader,
    const wspr::io::TrainingBatch& batch,
    size_t& inserted_count,
    double& kernel_time_ms
) {
    size_t n = batch.size();
    if (n == 0) return true;

    // Ensure GPU memory is allocated
    if (!gpu.allocate(n)) {
        return false;
    }

    size_t data_size = n * sizeof(float);
    size_t embedding_size = n * sizeof(float4);

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(gpu.d_tx_lat, batch.tx_lat.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_tx_lon, batch.tx_lon.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_rx_lat, batch.rx_lat.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_rx_lon, batch.rx_lon.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_kp, batch.kp_index.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_xray, batch.xray_flux.data(), data_size, cudaMemcpyHostToDevice));

    // Execute kernel
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto kernel_start = std::chrono::high_resolution_clock::now();

    compute_signature_embedding<<<num_blocks, BLOCK_SIZE>>>(
        n,
        gpu.d_tx_lat, gpu.d_tx_lon,
        gpu.d_rx_lat, gpu.d_rx_lon,
        gpu.d_kp, gpu.d_xray,
        gpu.d_embedding, gpu.d_distance
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    auto kernel_end = std::chrono::high_resolution_clock::now();
    kernel_time_ms = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();

    // Copy results back
    std::vector<float4> h_embedding(n);
    std::vector<float> h_distance(n);

    CUDA_CHECK(cudaMemcpy(h_embedding.data(), gpu.d_embedding, embedding_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_distance.data(), gpu.d_distance, data_size, cudaMemcpyDeviceToHost));

    // Convert to EmbeddingResult
    wspr::io::EmbeddingResult embeddings;
    embeddings.resize(n);
    for (size_t i = 0; i < n; i++) {
        embeddings.norm_distance[i] = h_embedding[i].x;
        embeddings.solar_penalty[i] = h_embedding[i].y;
        embeddings.geo_penalty[i] = h_embedding[i].z;
        embeddings.quality[i] = h_embedding[i].w;
        embeddings.distance_km[i] = h_distance[i];
    }

    // Insert to ClickHouse
    inserted_count = loader.insert_batch(batch, embeddings);

    return inserted_count > 0;
}

// =============================================================================
// Main Processing Loop
// =============================================================================

int main(int argc, char* argv[]) {
    print_header();

    // Parse arguments
    std::string start_date, end_date;
    bool hourly = false;
    bool dry_run = false;
    int band = 0;
    size_t max_batch = MAX_BATCH_SIZE;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--start" && i + 1 < argc) {
            start_date = argv[++i];
        } else if (arg == "--end" && i + 1 < argc) {
            end_date = argv[++i];
        } else if (arg == "--hourly") {
            hourly = true;
        } else if (arg == "--band" && i + 1 < argc) {
            band = std::stoi(argv[++i]);
        } else if (arg == "--dry-run") {
            dry_run = true;
        } else if (arg == "--batch-size" && i + 1 < argc) {
            max_batch = std::stoul(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (start_date.empty() || end_date.empty()) {
        fprintf(stderr, RED "ERROR: --start and --end are required\n" RESET);
        print_usage(argv[0]);
        return 1;
    }

    time_t start_time = parse_date(start_date);
    time_t end_time = parse_date(end_date);

    if (start_time < 0 || end_time < 0) {
        fprintf(stderr, RED "ERROR: Invalid date format. Use YYYY-MM-DD\n" RESET);
        return 1;
    }

    // Add 23:59:59 to end date to include full day
    end_time += 86400 - 1;

    // -------------------------------------------------------------------------
    // Initialize
    // -------------------------------------------------------------------------
    printf("Configuration:\n");
    printf("  Start:      %s\n", start_date.c_str());
    printf("  End:        %s\n", end_date.c_str());
    printf("  Mode:       %s\n", hourly ? "Hourly" : "Daily");
    printf("  Band:       %s\n", band > 0 ? std::to_string(band).c_str() : "All");
    printf("  Max batch:  %zu rows\n", max_batch);
    printf("  Dry run:    %s\n", dry_run ? "Yes" : "No");
    printf("\n");

    // GPU check
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, RED "ERROR: No CUDA devices found\n" RESET);
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (sm_%d%d, %.0f GB)\n\n", prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Connect to ClickHouse
    wspr::io::ConnectionConfig config;
    config.host = "localhost";
    config.port = 9000;
    config.database = "wspr";

    wspr::io::ClickHouseLoader loader(config);

    if (!loader.connect()) {
        fprintf(stderr, RED "ERROR: %s\n" RESET, loader.last_error().c_str());
        return 1;
    }

    printf("Connected to ClickHouse\n\n");

    // -------------------------------------------------------------------------
    // Processing Loop
    // -------------------------------------------------------------------------
    GPUContext gpu;
    auto global_start = std::chrono::high_resolution_clock::now();

    size_t total_fetched = 0;
    size_t total_inserted = 0;
    size_t total_errors = 0;
    size_t windows_processed = 0;
    double total_kernel_time = 0;

    // Time increment: 1 hour or 1 day
    int increment = hourly ? 3600 : 86400;

    printf("Processing:\n");
    printf("─────────────────────────────────────────────────────────────────────\n");

    for (time_t current = start_time; current <= end_time; current += increment) {
        std::string window_start = format_datetime(current, hourly);
        std::string window_end = format_datetime(current + increment - 1, hourly);

        // Build date strings for query
        std::string query_start = format_datetime(current, false);
        std::string query_end = format_datetime(current + increment - 1, false);

        try {
            // Fetch batch for this window
            auto fetch_start = std::chrono::high_resolution_clock::now();

            wspr::io::TrainingBatch batch = loader.fetch_batch(
                max_batch * 2,  // Fetch up to 2x max to check if we need to split
                query_start,
                query_end,
                band
            );

            auto fetch_end = std::chrono::high_resolution_clock::now();
            double fetch_time = std::chrono::duration<double>(fetch_end - fetch_start).count();

            size_t fetched = batch.size();
            total_fetched += fetched;

            if (fetched == 0) {
                printf("[%s] " YELLOW "No data" RESET "\n", window_start.c_str());
                continue;
            }

            if (dry_run) {
                printf("[%s] Would process %zu rows\n", window_start.c_str(), fetched);
                windows_processed++;
                continue;
            }

            // Process in chunks if needed
            size_t chunk_start = 0;
            size_t window_inserted = 0;
            double window_kernel_time = 0;

            while (chunk_start < fetched) {
                size_t chunk_size = std::min(max_batch, fetched - chunk_start);

                // Create sub-batch
                wspr::io::TrainingBatch chunk;
                chunk.reserve(chunk_size);

                for (size_t i = chunk_start; i < chunk_start + chunk_size; i++) {
                    chunk.tx_lat.push_back(batch.tx_lat[i]);
                    chunk.tx_lon.push_back(batch.tx_lon[i]);
                    chunk.rx_lat.push_back(batch.rx_lat[i]);
                    chunk.rx_lon.push_back(batch.rx_lon[i]);
                    chunk.kp_index.push_back(batch.kp_index[i]);
                    chunk.xray_flux.push_back(batch.xray_flux[i]);
                    chunk.sfi.push_back(batch.sfi[i]);
                    chunk.timestamps.push_back(batch.timestamps[i]);
                    chunk.tx_grids.push_back(batch.tx_grids[i]);
                    chunk.rx_grids.push_back(batch.rx_grids[i]);
                    chunk.frequencies.push_back(batch.frequencies[i]);
                    chunk.bands.push_back(batch.bands[i]);
                }

                size_t inserted = 0;
                double kernel_ms = 0;

                if (process_batch(gpu, loader, chunk, inserted, kernel_ms)) {
                    window_inserted += inserted;
                    window_kernel_time += kernel_ms;
                } else {
                    fprintf(stderr, RED "[%s] Batch failed: %s\n" RESET,
                            window_start.c_str(), loader.last_error().c_str());
                    total_errors++;
                }

                chunk_start += chunk_size;
            }

            total_inserted += window_inserted;
            total_kernel_time += window_kernel_time;
            windows_processed++;

            // Progress output
            double rate = window_inserted / (fetch_time + window_kernel_time / 1000.0);
            printf("[%s] Fetched %6zu | GPU %6.2fms | Inserted %6zu | %s%.0f rows/s%s\n",
                   window_start.c_str(),
                   fetched,
                   window_kernel_time,
                   window_inserted,
                   GREEN, rate, RESET);

        } catch (const std::exception& e) {
            fprintf(stderr, RED "[%s] Exception: %s\n" RESET, window_start.c_str(), e.what());
            total_errors++;
            // Continue to next window
        }
    }

    printf("─────────────────────────────────────────────────────────────────────\n");

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    auto global_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(global_end - global_start).count();

    printf("\n");
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  " BOLD "Processing Complete" RESET "                                      │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Windows processed: %-39zu │\n", windows_processed);
    printf("│  Total fetched:     %-39zu │\n", total_fetched);
    printf("│  Total inserted:    %-39zu │\n", total_inserted);
    printf("│  Errors:            %-39zu │\n", total_errors);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│  Total time:        %-39s │\n", format_duration(total_time).c_str());
    printf("│  GPU kernel time:   %-39s │\n", format_duration(total_kernel_time / 1000.0).c_str());
    printf("│  Avg throughput:    %-39.0f rows/sec │\n", total_inserted / total_time);
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("\n");

    if (total_errors > 0) {
        printf(YELLOW "Warning: %zu windows had errors. Check logs above.\n" RESET, total_errors);
    }

    if (dry_run) {
        printf(CYAN "Dry run complete. Use without --dry-run to process.\n" RESET);
    }

    return (total_errors > 0 && total_inserted == 0) ? 1 : 0;
}
