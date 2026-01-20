# Makefile for ki7mt-ai-lab-cuda
#
# Sovereign Hardware Abstraction Layer (HAL) for WSPR CUDA processing
# Target: RTX 5090 (sm_100) with legacy support (sm_80/86/89)
#
# Usage:
#   make              # Show help
#   make all          # Build static lib, shared lib, and check utility
#   make install      # Install to system (requires sudo)
#   make test         # Run verification tests
#   make clean        # Remove build artifacts

SHELL := /bin/bash
.PHONY: help all build install uninstall test clean check-cuda

# =============================================================================
# Package Metadata (from VERSION file)
# =============================================================================
NAME        := ki7mt-ai-lab-cuda
VERSION     := $(shell cat VERSION 2>/dev/null || echo "0.0.0")
SOVERSION   := $(word 1,$(subst ., ,$(VERSION)))

# =============================================================================
# Installation Paths (FHS-compliant)
# =============================================================================
PREFIX      := /usr
LIBDIR      := $(PREFIX)/lib64
INCDIR      := $(PREFIX)/include/ki7mt
BINDIR      := $(PREFIX)/bin
DATADIR     := $(PREFIX)/share/$(NAME)

# =============================================================================
# CUDA Toolkit Configuration (NVIDIA Upstream)
# =============================================================================
NVCC        := /usr/local/cuda/bin/nvcc
CUDA_PATH   := /usr/local/cuda
CUDA_INC    := $(CUDA_PATH)/include
CUDA_LIB    := $(CUDA_PATH)/lib64

# =============================================================================
# Compiler Flags
# =============================================================================
# Fat Binary: sm_80 (Ampere A100), sm_86 (RTX 30xx), sm_89 (RTX 40xx), sm_100 (RTX 50xx), sm_120 (Blackwell refresh)
GENCODE_FLAGS := \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_100,code=sm_100 \
    -gencode arch=compute_120,code=sm_120 \
    -gencode arch=compute_120,code=compute_120

# Optimization flags
# EXTRA_NVCCFLAGS can be set externally (e.g., -allow-unsupported-compiler for COPR)
EXTRA_NVCCFLAGS ?=
NVCC_FLAGS  := -O3 --use_fast_math -Xcompiler -fPIC $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS)
NVCC_SHARED := -shared -Xcompiler -fPIC

# Host compiler flags
CFLAGS      := -O3 -Wall -fPIC -I$(CUDA_INC)
LDFLAGS     := -L$(CUDA_LIB) -lcudart

# =============================================================================
# Build Directories
# =============================================================================
BUILDDIR    := build
OBJDIR      := $(BUILDDIR)/obj
LIBDIR_BUILD := $(BUILDDIR)/lib
BINDIR_BUILD := $(BUILDDIR)/bin

# =============================================================================
# Source Files
# =============================================================================
CUDA_SRCS   := src/cuda/bridge.cu src/cuda/bulk_kernels.cu
CUDA_HDRS   := src/cuda/bridge.h src/cuda/bulk_kernels.h src/cuda/wspr_structs.h
CHECK_SRC   := src/wspr-cuda-check.c

# =============================================================================
# Output Files
# =============================================================================
STATIC_LIB  := $(LIBDIR_BUILD)/lib$(NAME).a
SHARED_LIB  := $(LIBDIR_BUILD)/lib$(NAME).so.$(VERSION)
SONAME_LINK := $(LIBDIR_BUILD)/lib$(NAME).so.$(SOVERSION)
SHARED_LINK := $(LIBDIR_BUILD)/lib$(NAME).so
CHECK_BIN   := $(BINDIR_BUILD)/wspr-cuda-check
CUDA_OBJS   := $(OBJDIR)/bridge.o $(OBJDIR)/bulk_kernels.o

# =============================================================================
# Default Target
# =============================================================================
.DEFAULT_GOAL := help

help:
	@printf "\n"
	@printf "┌─────────────────────────────────────────────────────────────────┐\n"
	@printf "│  ki7mt-ai-lab-cuda v$(VERSION)                                      │\n"
	@printf "│  Sovereign CUDA HAL for WSPR Processing                         │\n"
	@printf "└─────────────────────────────────────────────────────────────────┘\n"
	@printf "\n"
	@printf "Fat Binary Targets: sm_80, sm_86, sm_89, sm_100, sm_120 + PTX\n"
	@printf "\n"
	@printf "Usage: make [target]\n"
	@printf "\n"
	@printf "Targets:\n"
	@printf "  help        Show this help message\n"
	@printf "  all         Build all outputs (static lib, shared lib, check utility)\n"
	@printf "  static      Build static library only (lib$(NAME).a)\n"
	@printf "  shared      Build shared library only (lib$(NAME).so)\n"
	@printf "  check-util  Build wspr-cuda-check utility only\n"
	@printf "  install     Install to system (PREFIX=$(PREFIX), requires sudo)\n"
	@printf "  uninstall   Remove installed files (requires sudo)\n"
	@printf "  test        Run verification tests\n"
	@printf "  clean       Remove all build artifacts\n"
	@printf "  check-cuda  Verify CUDA toolkit installation\n"
	@printf "\n"
	@printf "Variables:\n"
	@printf "  PREFIX           Installation prefix (default: /usr)\n"
	@printf "  DESTDIR          Staging directory for packaging\n"
	@printf "  CUDA_PATH        CUDA toolkit path (default: /usr/local/cuda)\n"
	@printf "  EXTRA_NVCCFLAGS  Additional nvcc flags (e.g., -allow-unsupported-compiler)\n"
	@printf "\n"
	@printf "Examples:\n"
	@printf "  make all                      # Build everything\n"
	@printf "  sudo make install             # Install to /usr\n"
	@printf "  DESTDIR=/tmp/stage make install  # Stage for RPM packaging\n"
	@printf "\n"
	@printf "Verify fat binary after build:\n"
	@printf "  cuobjdump --list-elf $(SHARED_LIB)\n"
	@printf "\n"

# =============================================================================
# CUDA Toolkit Verification
# =============================================================================
check-cuda:
	@printf "Checking CUDA toolkit installation...\n"
	@printf "\n"
	@if [ ! -x "$(NVCC)" ]; then \
		printf "ERROR: nvcc not found at $(NVCC)\n"; \
		printf "Install CUDA toolkit or set CUDA_PATH\n"; \
		exit 1; \
	fi
	@printf "  CUDA Path:  $(CUDA_PATH)\n"
	@printf "  nvcc:       $$($(NVCC) --version | grep release)\n"
	@printf "  Targets:    sm_80, sm_86, sm_89, sm_100, sm_120, compute_120 (PTX)\n"
	@printf "\n"
	@printf "CUDA toolkit check passed.\n"

# =============================================================================
# Build Targets
# =============================================================================
all: $(STATIC_LIB) $(SHARED_LIB) $(CHECK_BIN)
	@printf "\n"
	@printf "Build complete:\n"
	@printf "  Static:  $(STATIC_LIB)\n"
	@printf "  Shared:  $(SHARED_LIB)\n"
	@printf "  Utility: $(CHECK_BIN)\n"

static: $(STATIC_LIB)

shared: $(SHARED_LIB)

check-util: $(CHECK_BIN)

# Create build directories
$(OBJDIR) $(LIBDIR_BUILD) $(BINDIR_BUILD):
	@mkdir -p $@

# Compile CUDA sources to object files (fat binary)
$(OBJDIR)/bridge.o: src/cuda/bridge.cu $(CUDA_HDRS) | $(OBJDIR)
	@printf "Compiling bridge.cu (fat binary)...\n"
	@printf "  Architectures: sm_80, sm_86, sm_89, sm_100, sm_120 + PTX\n"
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<
	@printf "  Output: $@\n"

$(OBJDIR)/bulk_kernels.o: src/cuda/bulk_kernels.cu src/cuda/bulk_kernels.h | $(OBJDIR)
	@printf "Compiling bulk_kernels.cu (fat binary)...\n"
	@printf "  Architectures: sm_80, sm_86, sm_89, sm_100, sm_120 + PTX\n"
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<
	@printf "  Output: $@\n"

# Build static library
$(STATIC_LIB): $(CUDA_OBJS) | $(LIBDIR_BUILD)
	@printf "Creating static library...\n"
	ar rcs $@ $(CUDA_OBJS)
	@printf "  Output: $@\n"

# Build shared library with proper SONAME
$(SHARED_LIB): $(CUDA_OBJS) | $(LIBDIR_BUILD)
	@printf "Creating shared library...\n"
	$(NVCC) $(NVCC_FLAGS) $(NVCC_SHARED) \
		-Xlinker -soname,lib$(NAME).so.$(SOVERSION) \
		-o $@ $(CUDA_OBJS) -L$(CUDA_LIB) -lcudart
	ln -sf lib$(NAME).so.$(VERSION) $(LIBDIR_BUILD)/lib$(NAME).so.$(SOVERSION)
	ln -sf lib$(NAME).so.$(SOVERSION) $(SHARED_LINK)
	@printf "  Output: $@\n"
	@printf "  Soname: lib$(NAME).so.$(SOVERSION)\n"
	@printf "  Devlink: $(SHARED_LINK)\n"

# Build check utility
# NOTE: No RPATH - relies on ldconfig for installed library path
$(CHECK_BIN): $(CHECK_SRC) $(SHARED_LIB) | $(BINDIR_BUILD)
	@printf "Building wspr-cuda-check utility...\n"
	gcc $(CFLAGS) -I src/cuda -o $@ $(CHECK_SRC) -L$(LIBDIR_BUILD) -l$(NAME) $(LDFLAGS)
	@printf "  Output: $@\n"

# =============================================================================
# Installation
# =============================================================================
install: all
	@printf "Installing $(NAME) v$(VERSION) to $(DESTDIR)$(PREFIX)...\n"
	@printf "\n"
	# Create directories
	install -d $(DESTDIR)$(LIBDIR)
	install -d $(DESTDIR)$(INCDIR)
	install -d $(DESTDIR)$(BINDIR)
	install -d $(DESTDIR)$(DATADIR)/src
	# Install libraries
	install -m 644 $(STATIC_LIB) $(DESTDIR)$(LIBDIR)/
	install -m 755 $(SHARED_LIB) $(DESTDIR)$(LIBDIR)/
	ln -sf lib$(NAME).so.$(VERSION) $(DESTDIR)$(LIBDIR)/lib$(NAME).so.$(SOVERSION)
	ln -sf lib$(NAME).so.$(SOVERSION) $(DESTDIR)$(LIBDIR)/lib$(NAME).so
	# Install header
	install -m 644 $(CUDA_HDRS) $(DESTDIR)$(INCDIR)/
	# Install utility
	install -m 755 $(CHECK_BIN) $(DESTDIR)$(BINDIR)/
	# Install source (for reference/rebuilding)
	install -m 644 src/cuda/*.cu $(DESTDIR)$(DATADIR)/src/
	install -m 644 src/cuda/*.h $(DESTDIR)$(DATADIR)/src/
	install -m 644 src/cuda/*.c $(DESTDIR)$(DATADIR)/src/
	install -m 755 src/cuda/*.sh $(DESTDIR)$(DATADIR)/src/
	# Install wspr_structs.h to include directory (for development)
	install -m 644 src/cuda/wspr_structs.h $(DESTDIR)$(INCDIR)/
	@printf "\n"
	@printf "Installed:\n"
	@printf "  Header:   $(DESTDIR)$(INCDIR)/bridge.h\n"
	@printf "  Static:   $(DESTDIR)$(LIBDIR)/lib$(NAME).a\n"
	@printf "  Shared:   $(DESTDIR)$(LIBDIR)/lib$(NAME).so.$(VERSION)\n"
	@printf "  Utility:  $(DESTDIR)$(BINDIR)/wspr-cuda-check\n"
	@printf "  Source:   $(DESTDIR)$(DATADIR)/src/\n"
	@printf "\n"
	@printf "Run 'ldconfig' to update library cache.\n"

uninstall:
	@printf "Uninstalling $(NAME) from $(DESTDIR)$(PREFIX)...\n"
	rm -f $(DESTDIR)$(LIBDIR)/lib$(NAME).a
	rm -f $(DESTDIR)$(LIBDIR)/lib$(NAME).so*
	rm -f $(DESTDIR)$(INCDIR)/bridge.h
	rm -f $(DESTDIR)$(BINDIR)/wspr-cuda-check
	rm -rf $(DESTDIR)$(DATADIR)
	@printf "Uninstall complete.\n"

# =============================================================================
# Testing
# =============================================================================
test: all
	@printf "Running tests for $(NAME) v$(VERSION)...\n"
	@printf "\n"
	@# Test 1: Static library exists and is valid
	@printf "[TEST] Static library exists... "
	@test -f $(STATIC_LIB) && printf "PASS\n" || { printf "FAIL\n"; exit 1; }
	@# Test 2: Shared library exists
	@printf "[TEST] Shared library exists... "
	@test -f $(SHARED_LIB) && printf "PASS\n" || { printf "FAIL\n"; exit 1; }
	@# Test 3: Check utility exists
	@printf "[TEST] Check utility exists... "
	@test -x $(CHECK_BIN) && printf "PASS\n" || { printf "FAIL\n"; exit 1; }
	@# Test 4: Verify fat binary contains sm_100
	@printf "[TEST] Fat binary contains sm_100... "
	@$(CUDA_PATH)/bin/cuobjdump --list-elf $(SHARED_LIB) 2>/dev/null | grep -q "sm_100" && \
		printf "PASS\n" || { printf "FAIL (cuobjdump not available or sm_100 missing)\n"; }
	@# Test 5: Verify fat binary contains sm_80
	@printf "[TEST] Fat binary contains sm_80... "
	@$(CUDA_PATH)/bin/cuobjdump --list-elf $(SHARED_LIB) 2>/dev/null | grep -q "sm_80" && \
		printf "PASS\n" || { printf "FAIL (cuobjdump not available or sm_80 missing)\n"; }
	@# Test 5b: Verify fat binary contains sm_120 (Blackwell refresh)
	@printf "[TEST] Fat binary contains sm_120... "
	@$(CUDA_PATH)/bin/cuobjdump --list-elf $(SHARED_LIB) 2>/dev/null | grep -q "sm_120" && \
		printf "PASS\n" || { printf "FAIL (cuobjdump not available or sm_120 missing)\n"; }
	@# Test 6: Verify PTX embedded
	@printf "[TEST] Fat binary contains PTX... "
	@$(CUDA_PATH)/bin/cuobjdump --list-ptx $(SHARED_LIB) 2>/dev/null | grep -q "ptx" && \
		printf "PASS\n" || { printf "FAIL (cuobjdump not available or PTX missing)\n"; }
	@# Test 7: Header is C-pure (has extern "C")
	@printf "[TEST] Header is C-pure for CGO... "
	@grep -q 'extern "C"' src/cuda/bridge.h && printf "PASS\n" || { printf "FAIL\n"; exit 1; }
	@# Test 8: Run check utility (if GPU available)
	@printf "[TEST] Running wspr-cuda-check... "
	@LD_LIBRARY_PATH=$(LIBDIR_BUILD):$(CUDA_LIB) $(CHECK_BIN) >/dev/null 2>&1 && \
		printf "PASS\n" || printf "SKIP (no GPU)\n"
	@printf "\nAll tests completed.\n"

# =============================================================================
# Cleanup
# =============================================================================
clean:
	@printf "Cleaning build artifacts...\n"
	rm -rf $(BUILDDIR)
	@printf "Clean complete.\n"
