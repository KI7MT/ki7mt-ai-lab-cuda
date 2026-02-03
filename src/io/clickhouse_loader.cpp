/**
 * clickhouse_loader.cpp - ClickHouse Data Loader Implementation
 *
 * Fetches WSPR spots joined with solar indices for GPU processing.
 *
 * Part of: ki7mt-ai-lab-cuda (Sovereign CUDA Engine)
 *
 * Copyright (c) 2026 KI7MT - Greg Beam
 * License: GPL-3.0-or-later
 */

#include "clickhouse_loader.hpp"
#include <clickhouse/client.h>
#include <cmath>
#include <cctype>
#include <cstring>
#include <sstream>
#include <iostream>

namespace wspr {
namespace io {

// =============================================================================
// Grid Conversion
// =============================================================================

bool grid_to_latlon(const std::string& grid, float& lat, float& lon) {
    if (grid.length() < 4) {
        return false;
    }

    // Field (first 2 chars): A-R for both lon and lat
    char field_lon = std::toupper(grid[0]);
    char field_lat = std::toupper(grid[1]);

    if (field_lon < 'A' || field_lon > 'R' ||
        field_lat < 'A' || field_lat > 'R') {
        return false;
    }

    // Square (next 2 chars): 0-9
    if (!std::isdigit(grid[2]) || !std::isdigit(grid[3])) {
        return false;
    }

    int square_lon = grid[2] - '0';
    int square_lat = grid[3] - '0';

    // Calculate base coordinates
    // Longitude: -180 to +180, 20 degrees per field, 2 degrees per square
    // Latitude: -90 to +90, 10 degrees per field, 1 degree per square
    lon = (field_lon - 'A') * 20.0f - 180.0f + square_lon * 2.0f;
    lat = (field_lat - 'A') * 10.0f - 90.0f + square_lat * 1.0f;

    // Subsquare (optional 5th and 6th chars): a-x
    if (grid.length() >= 6) {
        char sub_lon = std::tolower(grid[4]);
        char sub_lat = std::tolower(grid[5]);

        if (sub_lon >= 'a' && sub_lon <= 'x' &&
            sub_lat >= 'a' && sub_lat <= 'x') {
            // 5 minutes per subsquare (1/24 of square)
            lon += (sub_lon - 'a') * (2.0f / 24.0f);
            lat += (sub_lat - 'a') * (1.0f / 24.0f);

            // Center of subsquare
            lon += 1.0f / 24.0f;
            lat += 0.5f / 24.0f;
        }
    } else {
        // Center of square (no subsquare)
        lon += 1.0f;  // Half of 2 degrees
        lat += 0.5f;  // Half of 1 degree
    }

    return true;
}

bool ClickHouseLoader::maidenhead_to_latlon(const char* grid, float& lat, float& lon) {
    if (!grid || grid[0] == '\0') {
        return false;
    }
    return grid_to_latlon(std::string(grid), lat, lon);
}

// =============================================================================
// Implementation Details
// =============================================================================

struct ClickHouseLoader::Impl {
    std::unique_ptr<clickhouse::Client> client;
};

// =============================================================================
// Constructor / Destructor
// =============================================================================

ClickHouseLoader::ClickHouseLoader(const ConnectionConfig& config)
    : impl_(std::make_unique<Impl>())
    , config_(config)
{
}

ClickHouseLoader::~ClickHouseLoader() = default;

// =============================================================================
// Connection Management
// =============================================================================

bool ClickHouseLoader::connect() {
    try {
        clickhouse::ClientOptions options;
        options.SetHost(config_.host);
        options.SetPort(config_.port);
        options.SetUser(config_.user);
        options.SetPassword(config_.password);
        options.SetDefaultDatabase(config_.database);

        impl_->client = std::make_unique<clickhouse::Client>(options);
        connected_ = true;
        last_error_.clear();
        return true;
    } catch (const std::exception& e) {
        last_error_ = std::string("Connection failed: ") + e.what();
        connected_ = false;
        return false;
    }
}

bool ClickHouseLoader::is_connected() const {
    return connected_;
}

// =============================================================================
// Data Fetching
// =============================================================================

TrainingBatch ClickHouseLoader::fetch_batch(
    size_t batch_size,
    const std::string& start_date,
    const std::string& end_date,
    int band
) {
    TrainingBatch batch;

    if (!connected_) {
        last_error_ = "Not connected to ClickHouse";
        return batch;
    }

    try {
        // Build query with LEFT JOIN to solar indices
        // Match on hour (solar data is typically hourly/3-hourly)
        std::ostringstream sql;
        sql << "SELECT "
            << "    w.grid AS tx_grid, "
            << "    w.reporter_grid AS rx_grid, "
            << "    toUnixTimestamp(w.timestamp) AS ts, "
            << "    coalesce(s.kp_index, 0) AS kp, "
            << "    coalesce(s.xray_long, 0) AS xray, "
            << "    coalesce(s.observed_flux, 0) AS sfi "
            << "FROM wspr.spots_raw w "
            << "LEFT JOIN solar.indices_raw s ON "
            << "    toDate(w.timestamp) = s.date AND "
            << "    toHour(w.timestamp) = toHour(s.time) "
            << "WHERE 1=1 ";

        if (!start_date.empty()) {
            sql << "AND w.timestamp >= toDateTime('" << start_date << " 00:00:00') ";
        }
        if (!end_date.empty()) {
            sql << "AND w.timestamp <= toDateTime('" << end_date << " 23:59:59') ";
        }
        if (band > 0) {
            sql << "AND w.band = " << band << " ";
        }

        // Filter for valid grids (at least 4 characters)
        sql << "AND length(w.grid) >= 4 "
            << "AND length(w.reporter_grid) >= 4 ";

        sql << "ORDER BY rand() "  // Random sampling for training
            << "LIMIT " << batch_size;

        batch.reserve(batch_size);

        impl_->client->Select(sql.str(), [&batch](const clickhouse::Block& block) {
            for (size_t row = 0; row < block.GetRowCount(); ++row) {
                // Extract grid strings
                auto tx_grid_col = block[0]->As<clickhouse::ColumnFixedString>();
                auto rx_grid_col = block[1]->As<clickhouse::ColumnFixedString>();
                auto ts_col = block[2]->As<clickhouse::ColumnUInt32>();
                auto kp_col = block[3]->As<clickhouse::ColumnFloat32>();
                auto xray_col = block[4]->As<clickhouse::ColumnFloat32>();
                auto sfi_col = block[5]->As<clickhouse::ColumnFloat32>();

                std::string tx_grid(tx_grid_col->At(row));
                std::string rx_grid(rx_grid_col->At(row));

                // Convert grids to lat/lon
                float tx_lat, tx_lon, rx_lat, rx_lon;
                if (!grid_to_latlon(tx_grid, tx_lat, tx_lon) ||
                    !grid_to_latlon(rx_grid, rx_lat, rx_lon)) {
                    continue;  // Skip invalid grids
                }

                // Add to batch
                batch.tx_lat.push_back(tx_lat);
                batch.tx_lon.push_back(tx_lon);
                batch.rx_lat.push_back(rx_lat);
                batch.rx_lon.push_back(rx_lon);
                batch.kp_index.push_back(kp_col->At(row));
                batch.xray_flux.push_back(xray_col->At(row));
                batch.sfi.push_back(sfi_col->At(row));
                batch.timestamps.push_back(ts_col->At(row));
            }
        });

        last_error_.clear();
    } catch (const std::exception& e) {
        last_error_ = std::string("Query failed: ") + e.what();
        batch.clear();
    }

    return batch;
}

uint64_t ClickHouseLoader::get_row_count(
    const std::string& start_date,
    const std::string& end_date
) {
    if (!connected_) {
        last_error_ = "Not connected to ClickHouse";
        return 0;
    }

    uint64_t count = 0;

    try {
        std::ostringstream sql;
        sql << "SELECT count() FROM wspr.spots_raw WHERE 1=1 ";

        if (!start_date.empty()) {
            sql << "AND timestamp >= toDateTime('" << start_date << " 00:00:00') ";
        }
        if (!end_date.empty()) {
            sql << "AND timestamp <= toDateTime('" << end_date << " 23:59:59') ";
        }

        impl_->client->Select(sql.str(), [&count](const clickhouse::Block& block) {
            if (block.GetRowCount() > 0) {
                count = block[0]->As<clickhouse::ColumnUInt64>()->At(0);
            }
        });

        last_error_.clear();
    } catch (const std::exception& e) {
        last_error_ = std::string("Count query failed: ") + e.what();
        count = 0;
    }

    return count;
}

} // namespace io
} // namespace wspr
