-- =============================================================================
-- wspr.model_features - ML Feature Storage for Propagation Signatures
-- =============================================================================
--
-- Stores GPU-computed embeddings for k-NN search and model training.
-- Optimized for vector similarity queries and M3 Ultra training exports.
--
-- Embedding vector components (Array(Float32)):
--   [0] norm_distance  - Normalized great circle distance (0-1)
--   [1] solar_penalty  - X-ray/D-layer absorption factor (0-1)
--   [2] geo_penalty    - Geomagnetic storm factor (0-1)
--   [3] quality        - Combined path quality score (0-1)
--
-- Part of: ki7mt-ai-lab-cuda (Sovereign CUDA Engine)
-- Copyright (c) 2026 KI7MT - Greg Beam
-- License: GPL-3.0-or-later
-- =============================================================================

CREATE TABLE IF NOT EXISTS wspr.model_features
(
    -- Temporal key
    timestamp DateTime CODEC(Delta, ZSTD(1)),

    -- Path identification
    tx_grid FixedString(8) CODEC(ZSTD(1)),
    rx_grid FixedString(8) CODEC(ZSTD(1)),

    -- Frequency (Hz) for band-specific analysis
    frequency UInt64 CODEC(Delta, ZSTD(1)),

    -- Band (ADIF) for partitioning
    band Int32 CODEC(ZSTD(1)),

    -- Raw distance (km) for filtering
    distance UInt32 CODEC(Delta, ZSTD(1)),

    -- Solar context at time of observation
    kp_index Float32 CODEC(ZSTD(1)),
    xray_flux Float32 CODEC(ZSTD(1)),

    -- GPU-computed embedding vector
    -- [norm_distance, solar_penalty, geo_penalty, quality]
    embedding Array(Float32) CODEC(ZSTD(1)),

    -- Processing metadata
    computed_at DateTime DEFAULT now() CODEC(Delta, ZSTD(1))
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (band, timestamp, tx_grid, rx_grid)
SETTINGS index_granularity = 8192;

-- Materialized view for quality distribution analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS wspr.v_quality_distribution
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (band, quality_bucket, timestamp)
AS SELECT
    toStartOfHour(timestamp) AS timestamp,
    band,
    floor(embedding[4] * 10) / 10 AS quality_bucket,
    count() AS count,
    avg(distance) AS avg_distance,
    avg(kp_index) AS avg_kp
FROM wspr.model_features
GROUP BY timestamp, band, quality_bucket;

-- Index for efficient embedding similarity queries
-- Note: ClickHouse doesn't have native vector indexes, but we can use
-- bloom filters for pre-filtering before exact computation
ALTER TABLE wspr.model_features ADD INDEX idx_quality (embedding[4]) TYPE minmax GRANULARITY 4;
ALTER TABLE wspr.model_features ADD INDEX idx_kp (kp_index) TYPE minmax GRANULARITY 4;
