/**
 * clickhouse_loader.hpp - ClickHouse Data Loader for Signature Engine
 *
 * Fetches WSPR spots joined with solar indices for GPU processing.
 * Converts Maidenhead grids to lat/lon coordinates.
 *
 * Part of: ki7mt-ai-lab-cuda (Sovereign CUDA Engine)
 *
 * Copyright (c) 2026 KI7MT - Greg Beam
 * License: GPL-3.0-or-later
 */

#ifndef WSPR_CLICKHOUSE_LOADER_HPP
#define WSPR_CLICKHOUSE_LOADER_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace wspr {
namespace io {

/**
 * Training batch for signature engine
 *
 * Columnar layout optimized for cudaMemcpy to device.
 * All vectors have the same length (batch_size).
 */
struct TrainingBatch {
    // Path geometry (from grid conversion)
    std::vector<float> tx_lat;
    std::vector<float> tx_lon;
    std::vector<float> rx_lat;
    std::vector<float> rx_lon;

    // Solar context (from solar.indices_raw join)
    std::vector<float> kp_index;
    std::vector<float> xray_flux;  // xray_long (0.1-0.8nm band)
    std::vector<float> sfi;        // Solar flux index

    // Timestamps for temporal features
    std::vector<uint32_t> timestamps;

    // Original grid strings (for write-back)
    std::vector<std::string> tx_grids;
    std::vector<std::string> rx_grids;

    // Frequency and band (for write-back)
    std::vector<uint64_t> frequencies;
    std::vector<int32_t> bands;

    // Metadata
    size_t size() const { return tx_lat.size(); }
    bool empty() const { return tx_lat.empty(); }

    void reserve(size_t n) {
        tx_lat.reserve(n);
        tx_lon.reserve(n);
        rx_lat.reserve(n);
        rx_lon.reserve(n);
        kp_index.reserve(n);
        xray_flux.reserve(n);
        sfi.reserve(n);
        timestamps.reserve(n);
        tx_grids.reserve(n);
        rx_grids.reserve(n);
        frequencies.reserve(n);
        bands.reserve(n);
    }

    void clear() {
        tx_lat.clear();
        tx_lon.clear();
        rx_lat.clear();
        rx_lon.clear();
        kp_index.clear();
        xray_flux.clear();
        sfi.clear();
        timestamps.clear();
        tx_grids.clear();
        rx_grids.clear();
        frequencies.clear();
        bands.clear();
    }
};

/**
 * Embedding result from GPU computation
 *
 * Matches float4 output from compute_signature_embedding kernel.
 */
struct EmbeddingResult {
    std::vector<float> norm_distance;   // Normalized distance (0-1)
    std::vector<float> solar_penalty;   // X-ray impact factor
    std::vector<float> geo_penalty;     // Kp impact factor
    std::vector<float> quality;         // Combined quality
    std::vector<float> distance_km;     // Raw distance in km

    size_t size() const { return quality.size(); }

    void reserve(size_t n) {
        norm_distance.reserve(n);
        solar_penalty.reserve(n);
        geo_penalty.reserve(n);
        quality.reserve(n);
        distance_km.reserve(n);
    }

    void resize(size_t n) {
        norm_distance.resize(n);
        solar_penalty.resize(n);
        geo_penalty.resize(n);
        quality.resize(n);
        distance_km.resize(n);
    }
};

/**
 * ClickHouse connection configuration
 */
struct ConnectionConfig {
    std::string host = "localhost";
    uint16_t port = 9000;
    std::string user = "default";
    std::string password = "";
    std::string database = "wspr";
};

/**
 * ClickHouse data loader for signature engine training
 */
class ClickHouseLoader {
public:
    explicit ClickHouseLoader(const ConnectionConfig& config = ConnectionConfig());
    ~ClickHouseLoader();

    // Non-copyable
    ClickHouseLoader(const ClickHouseLoader&) = delete;
    ClickHouseLoader& operator=(const ClickHouseLoader&) = delete;

    /**
     * Connect to ClickHouse server
     * @return true if connection successful
     */
    bool connect();

    /**
     * Check if connected
     */
    bool is_connected() const;

    /**
     * Fetch a batch of training data
     *
     * @param batch_size   Maximum rows to fetch
     * @param start_date   Start date (YYYY-MM-DD format)
     * @param end_date     End date (YYYY-MM-DD format)
     * @param band         Optional band filter (0 = all bands)
     * @return TrainingBatch with columnar data ready for GPU
     */
    TrainingBatch fetch_batch(
        size_t batch_size,
        const std::string& start_date = "",
        const std::string& end_date = "",
        int band = 0
    );

    /**
     * Get total row count for date range
     */
    uint64_t get_row_count(
        const std::string& start_date = "",
        const std::string& end_date = ""
    );

    /**
     * Insert computed embeddings into wspr.model_features
     *
     * @param batch      Original training batch (for metadata)
     * @param embeddings Computed embeddings from GPU
     * @return Number of rows inserted, or 0 on error
     */
    size_t insert_batch(
        const TrainingBatch& batch,
        const EmbeddingResult& embeddings
    );

    /**
     * Get last error message
     */
    const std::string& last_error() const { return last_error_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ConnectionConfig config_;
    std::string last_error_;
    bool connected_ = false;

    // Grid conversion utilities
    static bool maidenhead_to_latlon(const char* grid, float& lat, float& lon);
};

/**
 * Convert Maidenhead grid square to lat/lon
 *
 * Supports 4-character (e.g., "FN31") and 6-character (e.g., "FN31pr") grids.
 * Returns center of grid square.
 *
 * @param grid  Maidenhead grid string (4 or 6 characters)
 * @param lat   Output latitude (-90 to +90)
 * @param lon   Output longitude (-180 to +180)
 * @return true if conversion successful
 */
bool grid_to_latlon(const std::string& grid, float& lat, float& lon);

} // namespace io
} // namespace wspr

#endif // WSPR_CLICKHOUSE_LOADER_HPP
