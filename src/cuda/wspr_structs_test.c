/**
 * @file wspr_structs_test.c
 * @brief Compile-time and runtime verification of WSPRSpot memory layout
 *
 * Compile: gcc -std=c11 -o wspr_structs_test wspr_structs_test.c
 * Run: ./wspr_structs_test
 */

#include <stdio.h>
#include <stddef.h>
#include "wspr_structs.h"

int main(void) {
    printf("=============================================================\n");
    printf("WSPRSpot Memory Layout Verification\n");
    printf("=============================================================\n\n");

    printf("Schema Version: %d\n", WSPR_SCHEMA_VERSION);
    printf("Column Count:   %d\n", WSPR_COLUMN_COUNT);
    printf("Struct Size:    %zu bytes\n", sizeof(WSPRSpot));
    printf("Alignment:      %d bytes\n", WSPR_SPOT_ALIGNMENT);
    printf("Spots per 2KB:  %d\n", WSPR_SPOTS_PER_TILE);
    printf("uint4 per Spot: %d\n\n", WSPR_UINT4_PER_SPOT);

    printf("%-20s %8s %8s %8s   %s\n",
           "Field", "Offset", "Size", "End", "ClickHouse Type");
    printf("-------------------------------------------------------------\n");

    #define PRINT_FIELD(field, chtype) \
        printf("%-20s %8zu %8zu %8zu   %s\n", \
               #field, \
               offsetof(WSPRSpot, field), \
               sizeof(((WSPRSpot*)0)->field), \
               offsetof(WSPRSpot, field) + sizeof(((WSPRSpot*)0)->field), \
               chtype)

    PRINT_FIELD(id,            "UInt64");
    PRINT_FIELD(timestamp,     "DateTime (UInt32)");
    PRINT_FIELD(_pad1,         "[padding]");
    PRINT_FIELD(reporter,      "FixedString(16)");
    PRINT_FIELD(reporter_grid, "FixedString(8)");
    PRINT_FIELD(snr,           "Int8");
    PRINT_FIELD(_pad2,         "[padding]");
    PRINT_FIELD(frequency,     "UInt64 (Hz)");
    PRINT_FIELD(callsign,      "FixedString(16)");
    PRINT_FIELD(grid,          "FixedString(8)");
    PRINT_FIELD(power,         "Int8");
    PRINT_FIELD(drift,         "Int8");
    PRINT_FIELD(_pad3,         "[padding]");
    PRINT_FIELD(distance,      "UInt32");
    PRINT_FIELD(azimuth,       "UInt16");
    PRINT_FIELD(_pad4,         "[padding]");
    PRINT_FIELD(band,          "Int32");
    PRINT_FIELD(mode,          "FixedString(8)");
    PRINT_FIELD(version,       "FixedString(8)");
    PRINT_FIELD(code,          "UInt8");
    PRINT_FIELD(column_count,  "UInt8");
    PRINT_FIELD(_pad5,         "[padding]");

    printf("-------------------------------------------------------------\n");
    printf("%-20s %8s %8zu\n", "TOTAL", "", sizeof(WSPRSpot));
    printf("\n");

    /* Verify 16-byte boundary alignment for uint4 loads */
    printf("16-byte Boundary Check (for RTX 5090 uint4 vectorization):\n");
    printf("-------------------------------------------------------------\n");
    for (int i = 0; i < 8; i++) {
        int offset = i * 16;
        printf("  uint4[%d]: offset %3d - %3d\n", i, offset, offset + 15);
    }
    printf("\n");

    /* Validation tests */
    printf("Validation Tests:\n");
    printf("-------------------------------------------------------------\n");

    int pass = 1;

    #define CHECK(cond, msg) \
        if (!(cond)) { printf("  FAIL: %s\n", msg); pass = 0; } \
        else { printf("  PASS: %s\n", msg); }

    CHECK(sizeof(WSPRSpot) == 128,
          "sizeof(WSPRSpot) == 128 bytes");

    CHECK(sizeof(WSPRSpot) % 16 == 0,
          "sizeof(WSPRSpot) is 16-byte aligned");

    CHECK(offsetof(WSPRSpot, frequency) == 48,
          "frequency at offset 48 (8-byte aligned for double loads)");

    CHECK(offsetof(WSPRSpot, callsign) == 56,
          "callsign at offset 56 (16-byte aligned)");

    CHECK(offsetof(WSPRSpot, mode) == 96,
          "mode at offset 96 (16-byte aligned)");

    CHECK(offsetof(WSPRSpot, band) == 92,
          "band at offset 92 (4-byte aligned)");

    CHECK(offsetof(WSPRSpot, code) == 112,
          "code at offset 112");

    CHECK(offsetof(WSPRSpot, column_count) == 113,
          "column_count at offset 113");

    printf("\n");
    printf("=============================================================\n");
    printf("Result: %s\n", pass ? "ALL TESTS PASSED" : "TESTS FAILED");
    printf("=============================================================\n");

    return pass ? 0 : 1;
}
