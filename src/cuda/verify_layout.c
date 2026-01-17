/**
 * @file verify_layout.c
 * @brief Memory Layout Verification and Test Binary Generator
 * @version 2.0.0
 *
 * This utility:
 *   1. Prints byte-level memory layout of both WSPRSpot and WSPRSpotCH
 *   2. Generates test_row.bin with known values for ClickHouse verification
 *   3. Validates padding strip conversion is lossless
 *
 * Compile:
 *   gcc -std=c11 -Wall -Wextra -O2 -o verify_layout verify_layout.c
 *
 * Usage:
 *   ./verify_layout              # Print layout and generate test_row.bin
 *   ./verify_layout --hex        # Also print hex dump of binary
 *   ./verify_layout --help       # Show help
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <inttypes.h>

#include "wspr_structs.h"

/* ANSI color codes for terminal output */
#define C_RESET   "\033[0m"
#define C_RED     "\033[31m"
#define C_GREEN   "\033[32m"
#define C_YELLOW  "\033[33m"
#define C_CYAN    "\033[36m"
#define C_BOLD    "\033[1m"

/* Test data values (easily recognizable for verification) */
#define TEST_ID             12345ULL
#define TEST_TIMESTAMP      1704067200U  /* 2024-01-01 00:00:00 UTC */
#define TEST_REPORTER       "W1AW"
#define TEST_REPORTER_GRID  "FN31pr"
#define TEST_SNR            -15
#define TEST_FREQUENCY      14097100ULL  /* 14.0971 MHz in Hz */
#define TEST_CALLSIGN       "KI7MT"
#define TEST_GRID           "DN40aq"
#define TEST_POWER          37
#define TEST_DRIFT          -1
#define TEST_DISTANCE       4567U
#define TEST_AZIMUTH        270U
#define TEST_BAND           20
#define TEST_MODE           "WSPR"
#define TEST_VERSION        "2.6.1"
#define TEST_CODE           0
#define TEST_COLUMN_COUNT   17

/* Output filenames */
#define GPU_BIN_FILE        "test_row_gpu.bin"    /* 128-byte GPU format */
#define CH_BIN_FILE         "test_row.bin"        /* 99-byte ClickHouse format */

/**
 * @brief Print section header
 */
static void print_header(const char* title) {
    printf("\n%s%s==================================================================%s\n",
           C_BOLD, C_CYAN, C_RESET);
    printf("%s%s %s %s\n", C_BOLD, C_CYAN, title, C_RESET);
    printf("%s%s==================================================================%s\n\n",
           C_BOLD, C_CYAN, C_RESET);
}

/**
 * @brief Print a field's layout information
 */
static void print_field(const char* name, size_t offset, size_t size,
                        const char* ch_type, int is_padding) {
    if (is_padding) {
        printf("  %s%-18s%s  %3zu    %3zu    %3zu    %s\n",
               C_YELLOW, name, C_RESET, offset, size, offset + size, ch_type);
    } else {
        printf("  %-18s  %3zu    %3zu    %3zu    %s\n",
               name, offset, size, offset + size, ch_type);
    }
}

/**
 * @brief Print GPU struct (WSPRSpot) layout
 */
static void print_gpu_layout(void) {
    print_header("GPU Struct Layout: WSPRSpot (128 bytes)");

    printf("  %-18s  %3s    %4s    %3s    %s\n",
           "Field", "Off", "Size", "End", "ClickHouse Type");
    printf("  %-18s  %3s    %4s    %3s    %s\n",
           "------------------", "---", "----", "---", "--------------------");

    #define P(field, chtype, pad) \
        print_field(#field, offsetof(WSPRSpot, field), \
                    sizeof(((WSPRSpot*)0)->field), chtype, pad)

    P(id,            "UInt64",           0);
    P(timestamp,     "DateTime",         0);
    P(_pad1,         "[padding]",        1);
    P(reporter,      "FixedString(16)",  0);
    P(reporter_grid, "FixedString(8)",   0);
    P(snr,           "Int8",             0);
    P(_pad2,         "[padding]",        1);
    P(frequency,     "UInt64",           0);
    P(callsign,      "FixedString(16)",  0);
    P(grid,          "FixedString(8)",   0);
    P(power,         "Int8",             0);
    P(drift,         "Int8",             0);
    P(_pad3,         "[padding]",        1);
    P(distance,      "UInt32",           0);
    P(azimuth,       "UInt16",           0);
    P(_pad4,         "[padding]",        1);
    P(band,          "Int32",            0);
    P(mode,          "FixedString(8)",   0);
    P(version,       "FixedString(8)",   0);
    P(code,          "UInt8",            0);
    P(column_count,  "UInt8",            0);
    P(_pad5,         "[padding]",        1);

    #undef P

    printf("  %-18s  %3s    %4zu    %s\n", "------------------", "---",
           sizeof(WSPRSpot), "---");
    printf("  %-18s         %4zu bytes\n", "TOTAL", sizeof(WSPRSpot));
}

/**
 * @brief Print ClickHouse struct (WSPRSpotCH) layout
 */
static void print_ch_layout(void) {
    print_header("ClickHouse Struct Layout: WSPRSpotCH (99 bytes)");

    printf("  %-18s  %3s    %4s    %3s    %s\n",
           "Field", "Off", "Size", "End", "ClickHouse Type");
    printf("  %-18s  %3s    %4s    %3s    %s\n",
           "------------------", "---", "----", "---", "--------------------");

    #define P(field, chtype) \
        printf("  %-18s  %3zu    %3zu    %3zu    %s\n", \
               #field, offsetof(WSPRSpotCH, field), \
               sizeof(((WSPRSpotCH*)0)->field), \
               offsetof(WSPRSpotCH, field) + sizeof(((WSPRSpotCH*)0)->field), \
               chtype)

    P(id,            "UInt64");
    P(timestamp,     "DateTime");
    P(reporter,      "FixedString(16)");
    P(reporter_grid, "FixedString(8)");
    P(snr,           "Int8");
    P(frequency,     "UInt64");
    P(callsign,      "FixedString(16)");
    P(grid,          "FixedString(8)");
    P(power,         "Int8");
    P(drift,         "Int8");
    P(distance,      "UInt32");
    P(azimuth,       "UInt16");
    P(band,          "Int32");
    P(mode,          "FixedString(8)");
    P(version,       "FixedString(8)");
    P(code,          "UInt8");
    P(column_count,  "UInt8");

    #undef P

    printf("  %-18s  %3s    %4zu    %s\n", "------------------", "---",
           sizeof(WSPRSpotCH), "---");
    printf("  %-18s         %4zu bytes\n", "TOTAL", sizeof(WSPRSpotCH));
}

/**
 * @brief Initialize test WSPRSpot with known values
 */
static void init_test_spot(WSPRSpot* spot) {
    wspr_spot_init(spot);

    spot->id = TEST_ID;
    spot->timestamp = TEST_TIMESTAMP;

    /* Copy strings, ensuring null-padding */
    strncpy(spot->reporter, TEST_REPORTER, sizeof(spot->reporter));
    strncpy(spot->reporter_grid, TEST_REPORTER_GRID, sizeof(spot->reporter_grid));
    strncpy(spot->callsign, TEST_CALLSIGN, sizeof(spot->callsign));
    strncpy(spot->grid, TEST_GRID, sizeof(spot->grid));
    strncpy(spot->mode, TEST_MODE, sizeof(spot->mode));
    strncpy(spot->version, TEST_VERSION, sizeof(spot->version));

    spot->snr = TEST_SNR;
    spot->frequency = TEST_FREQUENCY;
    spot->power = TEST_POWER;
    spot->drift = TEST_DRIFT;
    spot->distance = TEST_DISTANCE;
    spot->azimuth = TEST_AZIMUTH;
    spot->band = TEST_BAND;
    spot->code = TEST_CODE;
    spot->column_count = TEST_COLUMN_COUNT;
}

/**
 * @brief Print test values that will be written
 */
static void print_test_values(void) {
    print_header("Test Values (for verification)");

    printf("  %-18s  %s\n", "Field", "Value");
    printf("  %-18s  %s\n", "------------------", "--------------------");
    printf("  %-18s  %" PRIu64 "\n", "id", (uint64_t)TEST_ID);
    printf("  %-18s  %u (2024-01-01 00:00:00 UTC)\n", "timestamp", TEST_TIMESTAMP);
    printf("  %-18s  \"%s\"\n", "reporter", TEST_REPORTER);
    printf("  %-18s  \"%s\"\n", "reporter_grid", TEST_REPORTER_GRID);
    printf("  %-18s  %d dB\n", "snr", TEST_SNR);
    printf("  %-18s  %" PRIu64 " Hz (%.6f MHz)\n", "frequency",
           (uint64_t)TEST_FREQUENCY, TEST_FREQUENCY / 1000000.0);
    printf("  %-18s  \"%s\"\n", "callsign", TEST_CALLSIGN);
    printf("  %-18s  \"%s\"\n", "grid", TEST_GRID);
    printf("  %-18s  %d dBm\n", "power", TEST_POWER);
    printf("  %-18s  %d Hz/min\n", "drift", TEST_DRIFT);
    printf("  %-18s  %u km\n", "distance", TEST_DISTANCE);
    printf("  %-18s  %u degrees\n", "azimuth", TEST_AZIMUTH);
    printf("  %-18s  %d (20m)\n", "band", TEST_BAND);
    printf("  %-18s  \"%s\"\n", "mode", TEST_MODE);
    printf("  %-18s  \"%s\"\n", "version", TEST_VERSION);
    printf("  %-18s  %u\n", "code", TEST_CODE);
    printf("  %-18s  %u\n", "column_count", TEST_COLUMN_COUNT);
}

/**
 * @brief Print hex dump of binary data
 */
static void print_hex_dump(const void* data, size_t size, const char* label) {
    const uint8_t* bytes = (const uint8_t*)data;

    printf("\n  %s%sHex dump: %s (%zu bytes)%s\n", C_BOLD, C_CYAN, label, size, C_RESET);
    printf("  ");

    for (size_t i = 0; i < size; i++) {
        if (i > 0 && i % 16 == 0) {
            printf("\n  ");
        } else if (i > 0 && i % 8 == 0) {
            printf(" ");
        }
        printf("%02x ", bytes[i]);
    }
    printf("\n");
}

/**
 * @brief Verify padding strip is lossless
 */
static int verify_conversion(const WSPRSpot* gpu, const WSPRSpotCH* ch) {
    int errors = 0;

    #define CHECK(field) \
        if (gpu->field != ch->field) { \
            printf("  %sFAIL%s: %s mismatch (gpu=%lld, ch=%lld)\n", \
                   C_RED, C_RESET, #field, \
                   (long long)gpu->field, (long long)ch->field); \
            errors++; \
        }

    #define CHECK_STR(field) \
        if (memcmp(gpu->field, ch->field, sizeof(gpu->field)) != 0) { \
            printf("  %sFAIL%s: %s mismatch\n", C_RED, C_RESET, #field); \
            errors++; \
        }

    CHECK(id);
    CHECK(timestamp);
    CHECK_STR(reporter);
    CHECK_STR(reporter_grid);
    CHECK(snr);
    CHECK(frequency);
    CHECK_STR(callsign);
    CHECK_STR(grid);
    CHECK(power);
    CHECK(drift);
    CHECK(distance);
    CHECK(azimuth);
    CHECK(band);
    CHECK_STR(mode);
    CHECK_STR(version);
    CHECK(code);
    CHECK(column_count);

    #undef CHECK
    #undef CHECK_STR

    if (errors == 0) {
        printf("  %sPASS%s: All 17 fields match after padding strip\n",
               C_GREEN, C_RESET);
    }

    return errors;
}

/**
 * @brief Write binary file
 */
static int write_binary_file(const char* filename, const void* data, size_t size) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("  %sERROR%s: Cannot open %s for writing\n",
               C_RED, C_RESET, filename);
        return -1;
    }

    size_t written = fwrite(data, 1, size, fp);
    fclose(fp);

    if (written != size) {
        printf("  %sERROR%s: Wrote %zu/%zu bytes to %s\n",
               C_RED, C_RESET, written, size, filename);
        return -1;
    }

    printf("  %sWROTE%s: %s (%zu bytes)\n", C_GREEN, C_RESET, filename, size);
    return 0;
}

/**
 * @brief Print ClickHouse verification SQL
 */
static void print_verification_sql(void) {
    print_header("ClickHouse Verification SQL");

    printf("  -- After running verify_ingestion.sh, run this query:\n\n");

    printf("  SELECT\n");
    printf("      id = %" PRIu64 " AS id_ok,\n", (uint64_t)TEST_ID);
    printf("      toUnixTimestamp(timestamp) = %u AS timestamp_ok,\n", TEST_TIMESTAMP);
    printf("      trimRight(reporter, '\\0') = '%s' AS reporter_ok,\n", TEST_REPORTER);
    printf("      trimRight(reporter_grid, '\\0') = '%s' AS reporter_grid_ok,\n", TEST_REPORTER_GRID);
    printf("      snr = %d AS snr_ok,\n", TEST_SNR);
    printf("      frequency = %" PRIu64 " AS frequency_ok,\n", (uint64_t)TEST_FREQUENCY);
    printf("      trimRight(callsign, '\\0') = '%s' AS callsign_ok,\n", TEST_CALLSIGN);
    printf("      trimRight(grid, '\\0') = '%s' AS grid_ok,\n", TEST_GRID);
    printf("      power = %d AS power_ok,\n", TEST_POWER);
    printf("      drift = %d AS drift_ok,\n", TEST_DRIFT);
    printf("      distance = %u AS distance_ok,\n", TEST_DISTANCE);
    printf("      azimuth = %u AS azimuth_ok,\n", TEST_AZIMUTH);
    printf("      band = %d AS band_ok,\n", TEST_BAND);
    printf("      trimRight(mode, '\\0') = '%s' AS mode_ok,\n", TEST_MODE);
    printf("      trimRight(version, '\\0') = '%s' AS version_ok,\n", TEST_VERSION);
    printf("      code = %u AS code_ok,\n", TEST_CODE);
    printf("      column_count = %u AS column_count_ok\n", TEST_COLUMN_COUNT);
    printf("  FROM wspr.spots_raw\n");
    printf("  WHERE id = %" PRIu64 ";\n", (uint64_t)TEST_ID);
}

/**
 * @brief Print usage
 */
static void print_usage(const char* prog) {
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("Options:\n");
    printf("  --hex     Print hex dump of generated binary files\n");
    printf("  --help    Show this help message\n");
}

int main(int argc, char* argv[]) {
    int show_hex = 0;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--hex") == 0) {
            show_hex = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    printf("\n%s%s╔══════════════════════════════════════════════════════════════════╗%s\n",
           C_BOLD, C_CYAN, C_RESET);
    printf("%s%s║         WSPR Memory Layout Verification Tool v2.0                ║%s\n",
           C_BOLD, C_CYAN, C_RESET);
    printf("%s%s║         RTX 5090 (128 bytes) ↔ ClickHouse (99 bytes)             ║%s\n",
           C_BOLD, C_CYAN, C_RESET);
    printf("%s%s╚══════════════════════════════════════════════════════════════════╝%s\n",
           C_BOLD, C_CYAN, C_RESET);

    /* Print struct sizes */
    print_header("Struct Size Summary");
    printf("  WSPRSpot (GPU):       %3zu bytes (128 expected)\n", sizeof(WSPRSpot));
    printf("  WSPRSpotCH (CH):      %3zu bytes (99 expected)\n", sizeof(WSPRSpotCH));
    printf("  Padding overhead:     %3zu bytes (29 expected)\n",
           sizeof(WSPRSpot) - sizeof(WSPRSpotCH));

    /* Print layouts */
    print_gpu_layout();
    print_ch_layout();

    /* Print test values */
    print_test_values();

    /* Create and convert test data */
    print_header("Generating Test Binary Files");

    WSPRSpot gpu_spot;
    WSPRSpotCH ch_spot;

    init_test_spot(&gpu_spot);
    WSPR_STRIP_PADDING(&ch_spot, &gpu_spot);

    /* Verify conversion */
    printf("\n  Verifying padding strip conversion...\n");
    int errors = verify_conversion(&gpu_spot, &ch_spot);

    /* Write binary files */
    printf("\n");
    write_binary_file(GPU_BIN_FILE, &gpu_spot, sizeof(gpu_spot));
    write_binary_file(CH_BIN_FILE, &ch_spot, sizeof(ch_spot));

    /* Optional hex dump */
    if (show_hex) {
        print_hex_dump(&gpu_spot, sizeof(gpu_spot), "WSPRSpot (GPU)");
        print_hex_dump(&ch_spot, sizeof(ch_spot), "WSPRSpotCH (ClickHouse)");
    }

    /* Print verification SQL */
    print_verification_sql();

    /* Summary */
    print_header("Summary");
    if (errors == 0) {
        printf("  %s✓ All static assertions passed%s\n", C_GREEN, C_RESET);
        printf("  %s✓ Padding strip conversion verified%s\n", C_GREEN, C_RESET);
        printf("  %s✓ Binary files generated%s\n", C_GREEN, C_RESET);
        printf("\n  Next step: Run ./verify_ingestion.sh to test ClickHouse ingestion\n");
        return 0;
    } else {
        printf("  %s✗ %d errors detected%s\n", C_RED, errors, C_RESET);
        return 1;
    }
}
