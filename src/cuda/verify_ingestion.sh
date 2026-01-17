#!/bin/bash
# ==============================================================================
# Name..........: verify_ingestion.sh
# Version.......: 2.0.0
# Description...: End-to-end verification of WSPR struct ↔ ClickHouse alignment
# ==============================================================================
#
# This script:
#   1. Compiles and runs verify_layout to generate test_row.bin
#   2. Creates test table in ClickHouse (wspr.spots_raw_test)
#   3. Ingests test_row.bin using RowBinary format
#   4. Queries the table to verify all 17 fields match expected values
#   5. Cleans up test data
#
# Prerequisites:
#   - GCC compiler
#   - ClickHouse server running locally (or set CLICKHOUSE_HOST)
#   - wspr database exists (run 01-wspr_schema_v2.sql first)
#
# Usage:
#   ./verify_ingestion.sh              # Run full verification
#   ./verify_ingestion.sh --keep       # Keep test data after verification
#   ./verify_ingestion.sh --help       # Show help
#
# ==============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLICKHOUSE_HOST="${CLICKHOUSE_HOST:-localhost}"
CLICKHOUSE_PORT="${CLICKHOUSE_PORT:-9000}"
CLICKHOUSE_CLIENT="clickhouse-client --host=${CLICKHOUSE_HOST} --port=${CLICKHOUSE_PORT}"

# Test data (must match verify_layout.c TEST_* values)
TEST_ID=12345
TEST_TIMESTAMP=1704067200
TEST_REPORTER="W1AW"
TEST_REPORTER_GRID="FN31pr"
TEST_SNR=-15
TEST_FREQUENCY=14097100
TEST_CALLSIGN="KI7MT"
TEST_GRID="DN40aq"
TEST_POWER=37
TEST_DRIFT=-1
TEST_DISTANCE=4567
TEST_AZIMUTH=270
TEST_BAND=20
TEST_MODE="WSPR"
TEST_VERSION="2.6.1"
TEST_CODE=0
TEST_COLUMN_COUNT=17

# Colors
C_RESET="\033[0m"
C_RED="\033[31m"
C_GREEN="\033[32m"
C_YELLOW="\033[33m"
C_CYAN="\033[36m"
C_BOLD="\033[1m"

# Parse arguments
KEEP_DATA=0
for arg in "$@"; do
    case "$arg" in
        --keep)
            KEEP_DATA=1
            ;;
        --help)
            printf "Usage: %s [OPTIONS]\n\n" "$0"
            printf "Options:\n"
            printf "  --keep    Keep test data in ClickHouse after verification\n"
            printf "  --help    Show this help message\n"
            exit 0
            ;;
    esac
done

# Utility functions
print_header() {
    printf "\n${C_BOLD}${C_CYAN}══════════════════════════════════════════════════════════════════${C_RESET}\n"
    printf "${C_BOLD}${C_CYAN} %s${C_RESET}\n" "$1"
    printf "${C_BOLD}${C_CYAN}══════════════════════════════════════════════════════════════════${C_RESET}\n\n"
}

print_ok() {
    printf "  ${C_GREEN}✓${C_RESET} %s\n" "$1"
}

print_fail() {
    printf "  ${C_RED}✗${C_RESET} %s\n" "$1"
}

print_info() {
    printf "  ${C_YELLOW}→${C_RESET} %s\n" "$1"
}

# Check prerequisites
check_prerequisites() {
    print_header "Step 1: Checking Prerequisites"

    # Check GCC
    if command -v gcc &>/dev/null; then
        print_ok "GCC found: $(gcc --version | head -1)"
    else
        print_fail "GCC not found"
        exit 1
    fi

    # Check ClickHouse client
    if command -v clickhouse-client &>/dev/null; then
        print_ok "ClickHouse client found"
    else
        print_fail "ClickHouse client not found"
        exit 1
    fi

    # Check ClickHouse server
    if ${CLICKHOUSE_CLIENT} -q "SELECT 1" &>/dev/null; then
        print_ok "ClickHouse server responding at ${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}"
    else
        print_fail "Cannot connect to ClickHouse at ${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}"
        exit 1
    fi

    # Check wspr database
    if ${CLICKHOUSE_CLIENT} -q "EXISTS DATABASE wspr" | grep -q 1; then
        print_ok "Database 'wspr' exists"
    else
        print_info "Creating database 'wspr'..."
        ${CLICKHOUSE_CLIENT} -q "CREATE DATABASE IF NOT EXISTS wspr"
        print_ok "Database 'wspr' created"
    fi
}

# Compile and run verify_layout
compile_and_generate() {
    print_header "Step 2: Compiling and Generating Test Binary"

    cd "${SCRIPT_DIR}"

    # Compile
    print_info "Compiling verify_layout.c..."
    gcc -std=c11 -Wall -Wextra -O2 -o verify_layout verify_layout.c

    if [ -f verify_layout ]; then
        print_ok "Compiled verify_layout"
    else
        print_fail "Compilation failed"
        exit 1
    fi

    # Run
    print_info "Running verify_layout..."
    ./verify_layout --hex

    if [ -f test_row.bin ]; then
        print_ok "Generated test_row.bin ($(wc -c < test_row.bin) bytes)"
    else
        print_fail "test_row.bin not generated"
        exit 1
    fi
}

# Create ClickHouse test table
create_test_table() {
    print_header "Step 3: Creating ClickHouse Test Table"

    print_info "Dropping existing test table (if any)..."
    ${CLICKHOUSE_CLIENT} -q "DROP TABLE IF EXISTS wspr.spots_raw_test"

    print_info "Creating wspr.spots_raw_test with FixedString columns..."
    ${CLICKHOUSE_CLIENT} -q "
        CREATE TABLE wspr.spots_raw_test (
            id UInt64,
            timestamp DateTime,
            reporter FixedString(16),
            reporter_grid FixedString(8),
            snr Int8,
            frequency UInt64,
            callsign FixedString(16),
            grid FixedString(8),
            power Int8,
            drift Int8,
            distance UInt32,
            azimuth UInt16,
            band Int32,
            mode FixedString(8),
            version FixedString(8),
            code UInt8,
            column_count UInt8
        ) ENGINE = MergeTree()
        ORDER BY (timestamp, id)
        COMMENT 'Test table for verify_ingestion.sh'
    "

    if ${CLICKHOUSE_CLIENT} -q "EXISTS TABLE wspr.spots_raw_test" | grep -q 1; then
        print_ok "Created wspr.spots_raw_test"
    else
        print_fail "Failed to create test table"
        exit 1
    fi
}

# Ingest binary data
ingest_binary() {
    print_header "Step 4: Ingesting Binary Data via RowBinary"

    cd "${SCRIPT_DIR}"

    print_info "Inserting test_row.bin into wspr.spots_raw_test..."

    # Use RowBinary format - ClickHouse reads columns in order
    cat test_row.bin | ${CLICKHOUSE_CLIENT} -q \
        "INSERT INTO wspr.spots_raw_test FORMAT RowBinary"

    # Verify row count
    ROW_COUNT=$(${CLICKHOUSE_CLIENT} -q "SELECT count() FROM wspr.spots_raw_test")

    if [ "$ROW_COUNT" = "1" ]; then
        print_ok "Inserted 1 row"
    else
        print_fail "Expected 1 row, got ${ROW_COUNT}"
        exit 1
    fi
}

# Verify data integrity
verify_data() {
    print_header "Step 5: Verifying Data Integrity (17 fields)"

    local ERRORS=0

    # Helper function to check a field
    check_field() {
        local FIELD_NAME="$1"
        local EXPECTED="$2"
        local QUERY="$3"

        local ACTUAL
        ACTUAL=$(${CLICKHOUSE_CLIENT} -q "$QUERY")

        if [ "$ACTUAL" = "$EXPECTED" ]; then
            printf "  ${C_GREEN}✓${C_RESET} %-18s = %s\n" "$FIELD_NAME" "$ACTUAL"
        else
            printf "  ${C_RED}✗${C_RESET} %-18s = %s (expected: %s)\n" \
                   "$FIELD_NAME" "$ACTUAL" "$EXPECTED"
            ((ERRORS++)) || true
        fi
    }

    printf "  %-18s   %s\n" "Field" "Value"
    printf "  %-18s   %s\n" "------------------" "--------------------"

    # Check each field
    check_field "id" "${TEST_ID}" \
        "SELECT id FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "timestamp" "${TEST_TIMESTAMP}" \
        "SELECT toUnixTimestamp(timestamp) FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "reporter" "${TEST_REPORTER}" \
        "SELECT trimRight(reporter, char(0)) FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "reporter_grid" "${TEST_REPORTER_GRID}" \
        "SELECT trimRight(reporter_grid, char(0)) FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "snr" "${TEST_SNR}" \
        "SELECT snr FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "frequency" "${TEST_FREQUENCY}" \
        "SELECT frequency FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "callsign" "${TEST_CALLSIGN}" \
        "SELECT trimRight(callsign, char(0)) FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "grid" "${TEST_GRID}" \
        "SELECT trimRight(grid, char(0)) FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "power" "${TEST_POWER}" \
        "SELECT power FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "drift" "${TEST_DRIFT}" \
        "SELECT drift FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "distance" "${TEST_DISTANCE}" \
        "SELECT distance FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "azimuth" "${TEST_AZIMUTH}" \
        "SELECT azimuth FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "band" "${TEST_BAND}" \
        "SELECT band FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "mode" "${TEST_MODE}" \
        "SELECT trimRight(mode, char(0)) FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "version" "${TEST_VERSION}" \
        "SELECT trimRight(version, char(0)) FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "code" "${TEST_CODE}" \
        "SELECT code FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    check_field "column_count" "${TEST_COLUMN_COUNT}" \
        "SELECT column_count FROM wspr.spots_raw_test WHERE id = ${TEST_ID}"

    printf "\n"

    if [ $ERRORS -eq 0 ]; then
        print_ok "All 17 fields verified successfully!"
        return 0
    else
        print_fail "${ERRORS} field(s) failed verification"
        return 1
    fi
}

# Cleanup
cleanup() {
    print_header "Step 6: Cleanup"

    cd "${SCRIPT_DIR}"

    if [ $KEEP_DATA -eq 0 ]; then
        print_info "Dropping test table..."
        ${CLICKHOUSE_CLIENT} -q "DROP TABLE IF EXISTS wspr.spots_raw_test"
        print_ok "Dropped wspr.spots_raw_test"

        print_info "Removing temporary files..."
        rm -f verify_layout test_row.bin test_row_gpu.bin
        print_ok "Removed temporary files"
    else
        print_info "Keeping test data (--keep specified)"
        print_info "Test table: wspr.spots_raw_test"
        print_info "Binary files: test_row.bin, test_row_gpu.bin"
    fi
}

# Main
main() {
    printf "\n${C_BOLD}${C_CYAN}╔══════════════════════════════════════════════════════════════════╗${C_RESET}\n"
    printf "${C_BOLD}${C_CYAN}║      WSPR RTX 5090 ↔ ClickHouse Ingestion Verification          ║${C_RESET}\n"
    printf "${C_BOLD}${C_CYAN}║      Testing 128-byte GPU struct → 99-byte RowBinary            ║${C_RESET}\n"
    printf "${C_BOLD}${C_CYAN}╚══════════════════════════════════════════════════════════════════╝${C_RESET}\n"

    check_prerequisites
    compile_and_generate
    create_test_table
    ingest_binary

    VERIFY_RESULT=0
    verify_data || VERIFY_RESULT=$?

    cleanup

    # Final summary
    print_header "Final Result"

    if [ $VERIFY_RESULT -eq 0 ]; then
        printf "  ${C_GREEN}${C_BOLD}══════════════════════════════════════════════════════════${C_RESET}\n"
        printf "  ${C_GREEN}${C_BOLD}  SUCCESS: Memory layout is perfectly aligned!            ${C_RESET}\n"
        printf "  ${C_GREEN}${C_BOLD}══════════════════════════════════════════════════════════${C_RESET}\n"
        printf "\n"
        printf "  The WSPRSpot C struct (128 bytes) correctly maps to\n"
        printf "  ClickHouse RowBinary format (99 bytes) via WSPR_STRIP_PADDING.\n"
        printf "\n"
        printf "  Your RTX 5090 GPU pipeline is ready for production.\n"
        printf "\n"
        return 0
    else
        printf "  ${C_RED}${C_BOLD}══════════════════════════════════════════════════════════${C_RESET}\n"
        printf "  ${C_RED}${C_BOLD}  FAILURE: Memory layout mismatch detected!               ${C_RESET}\n"
        printf "  ${C_RED}${C_BOLD}══════════════════════════════════════════════════════════${C_RESET}\n"
        printf "\n"
        printf "  Check the field verification output above for details.\n"
        printf "  Common causes:\n"
        printf "    - ClickHouse schema doesn't match wspr_structs.h\n"
        printf "    - Endianness mismatch (unlikely on x86_64)\n"
        printf "    - Padding strip macro has a bug\n"
        printf "\n"
        return 1
    fi
}

main "$@"
