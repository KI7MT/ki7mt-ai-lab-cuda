/**
 * @file wspr_structs.h
 * @brief WSPR Data Structures - Synchronized with ClickHouse Schema v2
 * @version 2.0.0
 *
 * CRITICAL: This struct layout MUST match:
 *   1. ClickHouse schema: wspr.spots_raw (01-wspr_schema_v2.sql)
 *   2. Go bridge: internal/parser/cuda_bridge.go
 *
 * MEMORY LAYOUT DESIGN PRINCIPLES:
 *   - 128-byte total size (8 x uint4 for RTX 5090 Blackwell vectorization)
 *   - 16-byte alignment for coalesced GPU memory access
 *   - Column order matches ClickHouse schema exactly
 *   - FixedString(N) in ClickHouse maps to char[N] in C
 *   - Explicit padding prevents compiler-dependent alignment issues
 *
 * SCHEMA VERSION: 2.0 (17 columns including mode and column_count)
 */

#ifndef WSPR_STRUCTS_H
#define WSPR_STRUCTS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* ============================================================================
 * Compile-Time Configuration
 * ============================================================================ */

/** Expected struct size (must be multiple of 16 for uint4 vectorization) */
#define WSPR_SPOT_SIZE_BYTES      128

/** Expected struct alignment (cache line friendly) */
#define WSPR_SPOT_ALIGNMENT       16

/** Schema version for runtime compatibility checks */
#define WSPR_SCHEMA_VERSION       2

/** Number of data columns (excluding padding) */
#define WSPR_COLUMN_COUNT         17

/* ============================================================================
 * WSPRSpot Structure (128 bytes, 16-byte aligned)
 * ============================================================================
 *
 * Memory Layout (matches ClickHouse wspr.spots_raw column order):
 *
 *   Offset   0: id             (uint64_t,  8 bytes)  UInt64
 *   Offset   8: timestamp      (uint32_t,  4 bytes)  DateTime (Unix seconds)
 *   Offset  12: _pad1          (uint8_t[4], 4 bytes) Alignment padding
 *   Offset  16: reporter       (char[16], 16 bytes)  FixedString(16)
 *   Offset  32: reporter_grid  (char[8],   8 bytes)  FixedString(8)
 *   Offset  40: snr            (int8_t,    1 byte)   Int8
 *   Offset  41: _pad2          (uint8_t[7], 7 bytes) Alignment padding
 *   Offset  48: frequency      (uint64_t,  8 bytes)  UInt64 (Hz, NOT MHz!)
 *   Offset  56: callsign       (char[16], 16 bytes)  FixedString(16)
 *   Offset  72: grid           (char[8],   8 bytes)  FixedString(8)
 *   Offset  80: power          (int8_t,    1 byte)   Int8
 *   Offset  81: drift          (int8_t,    1 byte)   Int8
 *   Offset  82: _pad3          (uint8_t[2], 2 bytes) Alignment padding
 *   Offset  84: distance       (uint32_t,  4 bytes)  UInt32
 *   Offset  88: azimuth        (uint16_t,  2 bytes)  UInt16
 *   Offset  90: _pad4          (uint8_t[2], 2 bytes) Alignment padding
 *   Offset  92: band           (int32_t,   4 bytes)  Int32
 *   Offset  96: mode           (char[8],   8 bytes)  FixedString(8)
 *   Offset 104: version        (char[8],   8 bytes)  FixedString(8)
 *   Offset 112: code           (uint8_t,   1 byte)   UInt8
 *   Offset 113: column_count   (uint8_t,   1 byte)   UInt8
 *   Offset 114: _pad5          (uint8_t[14], 14 bytes) Final alignment padding
 *   -----------------------------------------------------------------------
 *   Total: 128 bytes (8 x 16-byte uint4 loads on RTX 5090)
 *
 * ============================================================================ */

#pragma pack(push, 1)  /* Disable compiler padding - we handle it explicitly */

typedef struct __attribute__((aligned(WSPR_SPOT_ALIGNMENT))) WSPRSpot {
    /* ---- 64-bit Section (Offset 0-15) ---- */
    uint64_t id;                    /**< Offset 0:   WSPRnet spot ID (UInt64) */
    uint32_t timestamp;             /**< Offset 8:   Unix seconds (DateTime) */
    uint8_t  _pad1[4];              /**< Offset 12:  Align to 16-byte boundary */

    /* ---- Reporter Section (Offset 16-47) ---- */
    char     reporter[16];          /**< Offset 16:  Receiving callsign (FixedString(16)) */
    char     reporter_grid[8];      /**< Offset 32:  Receiver grid (FixedString(8)) */
    int8_t   snr;                   /**< Offset 40:  Signal-to-noise ratio dB (Int8) */
    uint8_t  _pad2[7];              /**< Offset 41:  Align frequency to 8-byte boundary */

    /* ---- Frequency Section (Offset 48-55) ---- */
    uint64_t frequency;             /**< Offset 48:  Frequency in Hz (UInt64, NOT MHz!) */

    /* ---- Transmitter Section (Offset 56-79) ---- */
    char     callsign[16];          /**< Offset 56:  Transmitting callsign (FixedString(16)) */
    char     grid[8];               /**< Offset 72:  Transmitter grid (FixedString(8)) */

    /* ---- Signal Parameters (Offset 80-95) ---- */
    int8_t   power;                 /**< Offset 80:  TX power dBm (Int8) */
    int8_t   drift;                 /**< Offset 81:  Frequency drift Hz/min (Int8) */
    uint8_t  _pad3[2];              /**< Offset 82:  Align distance to 4-byte boundary */
    uint32_t distance;              /**< Offset 84:  Great circle distance km (UInt32) */
    uint16_t azimuth;               /**< Offset 88:  Bearing degrees 0-359 (UInt16) */
    uint8_t  _pad4[2];              /**< Offset 90:  Align band to 4-byte boundary */
    int32_t  band;                  /**< Offset 92:  ADIF band ID (Int32) */

    /* ---- Metadata Section (Offset 96-113) ---- */
    char     mode[8];               /**< Offset 96:  WSPR mode e.g. "WSPR" (FixedString(8)) */
    char     version[8];            /**< Offset 104: Software version (FixedString(8)) */
    uint8_t  code;                  /**< Offset 112: Status/decode code (UInt8) */
    uint8_t  column_count;          /**< Offset 113: CSV column count for validation (UInt8) */

    /* ---- Final Padding (Offset 114-127) ---- */
    uint8_t  _pad5[14];             /**< Offset 114: Pad to 128 bytes (16-byte aligned) */

} WSPRSpot;

#pragma pack(pop)

/* ============================================================================
 * Compile-Time Static Assertions
 * ============================================================================
 * These assertions MUST pass or the build will fail. They ensure memory
 * layout compatibility between C, CUDA, and ClickHouse.
 * ============================================================================ */

/* Verify total struct size */
_Static_assert(
    sizeof(WSPRSpot) == WSPR_SPOT_SIZE_BYTES,
    "WSPRSpot size mismatch: expected 128 bytes for RTX 5090 uint4 vectorization"
);

/* Verify 16-byte alignment for coalesced GPU memory access */
_Static_assert(
    sizeof(WSPRSpot) % WSPR_SPOT_ALIGNMENT == 0,
    "WSPRSpot must be 16-byte aligned for Blackwell architecture"
);

/* Verify field offsets match ClickHouse column layout */
_Static_assert(offsetof(WSPRSpot, id) == 0,
    "id offset mismatch");
_Static_assert(offsetof(WSPRSpot, timestamp) == 8,
    "timestamp offset mismatch");
_Static_assert(offsetof(WSPRSpot, reporter) == 16,
    "reporter offset mismatch");
_Static_assert(offsetof(WSPRSpot, reporter_grid) == 32,
    "reporter_grid offset mismatch");
_Static_assert(offsetof(WSPRSpot, snr) == 40,
    "snr offset mismatch");
_Static_assert(offsetof(WSPRSpot, frequency) == 48,
    "frequency offset mismatch");
_Static_assert(offsetof(WSPRSpot, callsign) == 56,
    "callsign offset mismatch");
_Static_assert(offsetof(WSPRSpot, grid) == 72,
    "grid offset mismatch");
_Static_assert(offsetof(WSPRSpot, power) == 80,
    "power offset mismatch");
_Static_assert(offsetof(WSPRSpot, drift) == 81,
    "drift offset mismatch");
_Static_assert(offsetof(WSPRSpot, distance) == 84,
    "distance offset mismatch");
_Static_assert(offsetof(WSPRSpot, azimuth) == 88,
    "azimuth offset mismatch");
_Static_assert(offsetof(WSPRSpot, band) == 92,
    "band offset mismatch");
_Static_assert(offsetof(WSPRSpot, mode) == 96,
    "mode offset mismatch");
_Static_assert(offsetof(WSPRSpot, version) == 104,
    "version offset mismatch");
_Static_assert(offsetof(WSPRSpot, code) == 112,
    "code offset mismatch");
_Static_assert(offsetof(WSPRSpot, column_count) == 113,
    "column_count offset mismatch");

/* ============================================================================
 * Helper Macros for GPU Kernels
 * ============================================================================ */

/** Number of WSPRSpot structs per 2KB shared memory tile */
#define WSPR_SPOTS_PER_TILE       (2048 / WSPR_SPOT_SIZE_BYTES)  /* 16 spots */

/** Number of uint4 loads required per WSPRSpot */
#define WSPR_UINT4_PER_SPOT       (WSPR_SPOT_SIZE_BYTES / 16)    /* 8 loads */

/** Warp-aligned batch size for coalesced access */
#define WSPR_WARP_BATCH_SIZE      32

/** ClickHouse RowBinary size (no padding) */
#define WSPR_CLICKHOUSE_ROW_SIZE  99

/* ============================================================================
 * WSPRSpotCH - ClickHouse RowBinary Format (99 bytes, no padding)
 * ============================================================================
 *
 * This struct matches ClickHouse's RowBinary format EXACTLY.
 * Use this for direct I/O with ClickHouse via FORMAT RowBinary.
 *
 * The GPU uses WSPRSpot (128 bytes), then we strip padding to WSPRSpotCH
 * (99 bytes) before sending to ClickHouse.
 *
 * ============================================================================ */

#pragma pack(push, 1)

typedef struct WSPRSpotCH {
    uint64_t id;                /**< 8 bytes  - UInt64 */
    uint32_t timestamp;         /**< 4 bytes  - DateTime */
    char     reporter[16];      /**< 16 bytes - FixedString(16) */
    char     reporter_grid[8];  /**< 8 bytes  - FixedString(8) */
    int8_t   snr;               /**< 1 byte   - Int8 */
    uint64_t frequency;         /**< 8 bytes  - UInt64 */
    char     callsign[16];      /**< 16 bytes - FixedString(16) */
    char     grid[8];           /**< 8 bytes  - FixedString(8) */
    int8_t   power;             /**< 1 byte   - Int8 */
    int8_t   drift;             /**< 1 byte   - Int8 */
    uint32_t distance;          /**< 4 bytes  - UInt32 */
    uint16_t azimuth;           /**< 2 bytes  - UInt16 */
    int32_t  band;              /**< 4 bytes  - Int32 */
    char     mode[8];           /**< 8 bytes  - FixedString(8) */
    char     version[8];        /**< 8 bytes  - FixedString(8) */
    uint8_t  code;              /**< 1 byte   - UInt8 */
    uint8_t  column_count;      /**< 1 byte   - UInt8 */
    /* Total: 99 bytes */
} WSPRSpotCH;

#pragma pack(pop)

/* Verify ClickHouse struct size */
_Static_assert(
    sizeof(WSPRSpotCH) == WSPR_CLICKHOUSE_ROW_SIZE,
    "WSPRSpotCH size mismatch: expected 99 bytes for ClickHouse RowBinary"
);

/* ============================================================================
 * Padding Strip Macros - GPU (128 bytes) → ClickHouse (99 bytes)
 * ============================================================================
 *
 * These macros provide efficient conversion from the GPU-optimized WSPRSpot
 * struct to the ClickHouse-compatible WSPRSpotCH format.
 *
 * Strategy: Direct field copy (compiler optimizes to minimal memcpy calls)
 *
 * ============================================================================ */

/**
 * @brief Convert WSPRSpot (128 bytes) to WSPRSpotCH (99 bytes)
 * @param dst Pointer to WSPRSpotCH destination
 * @param src Pointer to WSPRSpot source
 *
 * This strips the 29 bytes of padding (_pad1 through _pad5).
 * Optimized: Compiler will generate efficient field-by-field copy.
 */
#define WSPR_STRIP_PADDING(dst, src) do { \
    (dst)->id           = (src)->id; \
    (dst)->timestamp    = (src)->timestamp; \
    __builtin_memcpy((dst)->reporter, (src)->reporter, 16); \
    __builtin_memcpy((dst)->reporter_grid, (src)->reporter_grid, 8); \
    (dst)->snr          = (src)->snr; \
    (dst)->frequency    = (src)->frequency; \
    __builtin_memcpy((dst)->callsign, (src)->callsign, 16); \
    __builtin_memcpy((dst)->grid, (src)->grid, 8); \
    (dst)->power        = (src)->power; \
    (dst)->drift        = (src)->drift; \
    (dst)->distance     = (src)->distance; \
    (dst)->azimuth      = (src)->azimuth; \
    (dst)->band         = (src)->band; \
    __builtin_memcpy((dst)->mode, (src)->mode, 8); \
    __builtin_memcpy((dst)->version, (src)->version, 8); \
    (dst)->code         = (src)->code; \
    (dst)->column_count = (src)->column_count; \
} while(0)

/**
 * @brief Convert WSPRSpotCH (99 bytes) to WSPRSpot (128 bytes)
 * @param dst Pointer to WSPRSpot destination (will be zero-initialized)
 * @param src Pointer to WSPRSpotCH source
 *
 * This adds the 29 bytes of padding for GPU processing.
 */
#define WSPR_ADD_PADDING(dst, src) do { \
    __builtin_memset((dst), 0, sizeof(WSPRSpot)); \
    (dst)->id           = (src)->id; \
    (dst)->timestamp    = (src)->timestamp; \
    __builtin_memcpy((dst)->reporter, (src)->reporter, 16); \
    __builtin_memcpy((dst)->reporter_grid, (src)->reporter_grid, 8); \
    (dst)->snr          = (src)->snr; \
    (dst)->frequency    = (src)->frequency; \
    __builtin_memcpy((dst)->callsign, (src)->callsign, 16); \
    __builtin_memcpy((dst)->grid, (src)->grid, 8); \
    (dst)->power        = (src)->power; \
    (dst)->drift        = (src)->drift; \
    (dst)->distance     = (src)->distance; \
    (dst)->azimuth      = (src)->azimuth; \
    (dst)->band         = (src)->band; \
    __builtin_memcpy((dst)->mode, (src)->mode, 8); \
    __builtin_memcpy((dst)->version, (src)->version, 8); \
    (dst)->code         = (src)->code; \
    (dst)->column_count = (src)->column_count; \
} while(0)

/* ============================================================================
 * Batch Conversion Functions (for bulk GPU → ClickHouse transfer)
 * ============================================================================ */

/**
 * @brief Batch convert GPU spots to ClickHouse format
 * @param dst Output buffer (must hold count * 99 bytes)
 * @param src Input buffer of WSPRSpot structs
 * @param count Number of spots to convert
 * @return Number of bytes written (count * 99)
 *
 * This function is optimized for streaming to ClickHouse.
 * For maximum throughput, use with pinned memory buffers.
 */
static inline size_t wspr_batch_strip_padding(
    WSPRSpotCH* dst,
    const WSPRSpot* src,
    size_t count
) {
    for (size_t i = 0; i < count; i++) {
        WSPR_STRIP_PADDING(&dst[i], &src[i]);
    }
    return count * sizeof(WSPRSpotCH);
}

/**
 * @brief Batch convert ClickHouse rows to GPU format
 * @param dst Output buffer (must hold count * 128 bytes, 16-byte aligned)
 * @param src Input buffer of WSPRSpotCH structs
 * @param count Number of spots to convert
 * @return Number of bytes written (count * 128)
 */
static inline size_t wspr_batch_add_padding(
    WSPRSpot* dst,
    const WSPRSpotCH* src,
    size_t count
) {
    for (size_t i = 0; i < count; i++) {
        WSPR_ADD_PADDING(&dst[i], &src[i]);
    }
    return count * sizeof(WSPRSpot);
}

/* ============================================================================
 * Direct Memory Offset Table (for advanced zero-copy scenarios)
 * ============================================================================
 *
 * Use these offsets when you need to extract individual fields from a
 * WSPRSpot without converting to WSPRSpotCH. Useful for CUDA kernels
 * that need to write directly to a ClickHouse-compatible buffer.
 *
 * ============================================================================ */

/* GPU struct offsets (128-byte WSPRSpot) */
#define WSPR_GPU_OFF_ID              0
#define WSPR_GPU_OFF_TIMESTAMP       8
#define WSPR_GPU_OFF_REPORTER       16
#define WSPR_GPU_OFF_REPORTER_GRID  32
#define WSPR_GPU_OFF_SNR            40
#define WSPR_GPU_OFF_FREQUENCY      48
#define WSPR_GPU_OFF_CALLSIGN       56
#define WSPR_GPU_OFF_GRID           72
#define WSPR_GPU_OFF_POWER          80
#define WSPR_GPU_OFF_DRIFT          81
#define WSPR_GPU_OFF_DISTANCE       84
#define WSPR_GPU_OFF_AZIMUTH        88
#define WSPR_GPU_OFF_BAND           92
#define WSPR_GPU_OFF_MODE           96
#define WSPR_GPU_OFF_VERSION       104
#define WSPR_GPU_OFF_CODE          112
#define WSPR_GPU_OFF_COLUMN_COUNT  113

/* ClickHouse struct offsets (99-byte WSPRSpotCH) */
#define WSPR_CH_OFF_ID              0
#define WSPR_CH_OFF_TIMESTAMP       8
#define WSPR_CH_OFF_REPORTER       12
#define WSPR_CH_OFF_REPORTER_GRID  28
#define WSPR_CH_OFF_SNR            36
#define WSPR_CH_OFF_FREQUENCY      37
#define WSPR_CH_OFF_CALLSIGN       45
#define WSPR_CH_OFF_GRID           61
#define WSPR_CH_OFF_POWER          69
#define WSPR_CH_OFF_DRIFT          70
#define WSPR_CH_OFF_DISTANCE       71
#define WSPR_CH_OFF_AZIMUTH        75
#define WSPR_CH_OFF_BAND           77
#define WSPR_CH_OFF_MODE           81
#define WSPR_CH_OFF_VERSION        89
#define WSPR_CH_OFF_CODE           97
#define WSPR_CH_OFF_COLUMN_COUNT   98

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief Initialize a WSPRSpot with zero values
 * @param spot Pointer to WSPRSpot structure
 */
static inline void wspr_spot_init(WSPRSpot* spot) {
    __builtin_memset(spot, 0, sizeof(WSPRSpot));
    spot->column_count = WSPR_COLUMN_COUNT;
}

/**
 * @brief Validate WSPRSpot has minimum required data
 * @param spot Pointer to WSPRSpot structure
 * @return 1 if valid, 0 if invalid
 */
static inline int wspr_spot_is_valid(const WSPRSpot* spot) {
    return (spot->timestamp > 0) &&
           (spot->frequency > 0) &&
           (spot->callsign[0] != '\0') &&
           (spot->reporter[0] != '\0');
}

/**
 * @brief Convert frequency from Hz to MHz (for display)
 * @param hz Frequency in Hz
 * @return Frequency in MHz
 */
static inline double wspr_hz_to_mhz(uint64_t hz) {
    return (double)hz / 1000000.0;
}

/**
 * @brief Convert frequency from MHz to Hz (for storage)
 * @param mhz Frequency in MHz
 * @return Frequency in Hz
 */
static inline uint64_t wspr_mhz_to_hz(double mhz) {
    return (uint64_t)(mhz * 1000000.0 + 0.5);  /* Round to nearest Hz */
}

#ifdef __cplusplus
}
#endif

#endif /* WSPR_STRUCTS_H */
