# Disable debug package generation (CUDA binaries don't have standard debug info)
%global debug_package %{nil}

Name:           ki7mt-ai-lab-cuda
Version:        2.1.0
Release:        1%{?dist}
Summary:        Sovereign CUDA HAL for KI7MT AI Lab WSPR processing

License:        GPL-3.0-or-later
URL:            https://github.com/KI7MT/ki7mt-ai-lab-cuda
# Hardcoded Source avoids rpkg naming conflicts
Source0:        https://github.com/KI7MT/ki7mt-ai-lab-cuda/archive/v%{version}.tar.gz

# Architecture-specific (fat binary: sm_80, sm_86, sm_89, sm_100)
ExclusiveArch:  x86_64

# Build requirements (NVIDIA CUDA Toolkit 13.1 upstream)
BuildRequires:  cuda-nvcc-13-1
BuildRequires:  cuda-cudart-devel-13-1
BuildRequires:  gcc
BuildRequires:  make

# Runtime requirements (RTX 5090 Blackwell requires driver >= 590.48.01)
Requires:       nvidia-driver-cuda >= 590.48.01
Requires:       cuda-cudart-13-1 >= 13.1.0

%description
Sovereign Hardware Abstraction Layer (HAL) providing high-performance CUDA
kernels for GPU-accelerated WSPR (Weak Signal Propagation Reporter) data
processing. Fat binary supports RTX 5090 (sm_100), RTX 40xx (sm_89),
RTX 30xx (sm_86), and Ampere A100 (sm_80).

Features:
- Zero-copy pinned memory allocation
- Async host-to-device and device-to-host transfers
- CUDA stream management for pipelined processing
- WSPR spot validation kernel
- Vectorized processing kernel (normalize + validate + convert)
- GPU-based deduplication kernel
- Callsign sanitization kernel

Optimized for 10+ billion row WSPR datasets with RTX 5090 (32GB VRAM).

%package devel
Summary:        Development files for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}
Requires:       cuda-cudart-devel-13-1

%description devel
Header files and CUDA source code for the KI7MT AI Lab CUDA HAL.
Required for building applications with CGO that use the CUDA kernels.

%prep
%autosetup -n %{name}-%{version}

%build
# Set CUDA paths for NVIDIA upstream toolkit
export CUDA_PATH=/usr/local/cuda-13.1
export PATH=$CUDA_PATH/bin:$PATH

# Build fat binary (sm_80, sm_86, sm_89, sm_100 + PTX)
# EXTRA_NVCCFLAGS allows COPR builders to handle GCC version mismatches
make all CUDA_PATH=$CUDA_PATH EXTRA_NVCCFLAGS="-allow-unsupported-compiler"

%install
# Set CUDA paths
export CUDA_PATH=/usr/local/cuda-13.1

# Use Makefile install with DESTDIR
make install DESTDIR=%{buildroot} CUDA_PATH=$CUDA_PATH

# Modern EL9 standard for ldconfig scriptlets
%ldconfig_scriptlets

%files
%license COPYING
%doc README.md
# Shared library with soname versioning
%{_libdir}/lib%{name}.so.*
# Check utility
%{_bindir}/wspr-cuda-check

%files devel
%doc src/cuda/README.md
# Header files (in /usr/include/ki7mt/)
%dir %{_includedir}/ki7mt
%{_includedir}/ki7mt/bridge.h
%{_includedir}/ki7mt/bulk_kernels.h
%{_includedir}/ki7mt/wspr_structs.h
# Static library
%{_libdir}/lib%{name}.a
# Development symlink
%{_libdir}/lib%{name}.so
# Source files for reference/rebuilding
%dir %{_datadir}/%{name}
%dir %{_datadir}/%{name}/src
%{_datadir}/%{name}/src/*.cu
%{_datadir}/%{name}/src/*.h
%{_datadir}/%{name}/src/*.c
# Verification utilities
%attr(755,root,root) %{_datadir}/%{name}/src/*.sh

%changelog
* Mon Feb 03 2026 Greg Beam <ki7mt@yahoo.com> - 2.1.0-1
- Align version across all lab packages at 2.1.0

* Mon Feb 03 2026 Greg Beam <ki7mt@yahoo.com> - 2.0.8-1
- Fix Physics Gap: 3-hour bucket JOIN for solar data matching
- ClickHouse loader CTE aggregates Kp/X-ray/SFI into 3-hour windows
- Solar penalty now varies 0.1-1.0 (was stuck at 1.0)
- Geo penalty now varies 0.26-1.0 (was stuck at 1.0)
- Add wspr.training_set_v1 Gold Standard dataset (6M rows, Jan 27-Feb 1)

* Sun Feb 02 2026 Greg Beam <ki7mt@yahoo.com> - 2.0.7-1
- Phase 7: Vector Vault - float4 embeddings with write-back to ClickHouse
- Phase 8: Bulk Processor for overnight dataset processing
- Add wspr.model_features table DDL for ML training
- ClickHouse loader with insert_batch() for embedding storage
- Tested: 48M embeddings in 69 seconds (686K rows/sec)

* Sun Feb 02 2026 Greg Beam <ki7mt@yahoo.com> - 2.0.6-1
- Add CUDA Signature Engine for Blackwell sm_120
- Add CMakeLists.txt with CMake 3.28+ CUDA language support
- Add signature_kernel.cu with path quality computation (Haversine, Kp, X-ray)
- Add test_signature.cu with 8 test scenarios
- Verified on RTX PRO 6000 Blackwell (95GB, 188 SMs)

* Mon Jan 20 2025 Greg Beam <ki7mt@yahoo.com> - 2.0.4-1
- Fix spec: README -> README.md
- Fix changelog day-of-week errors

* Mon Jan 20 2025 Greg Beam <ki7mt@yahoo.com> - 2.0.3-1
- Fix maintainer email in changelog

* Mon Jan 20 2025 Greg Beam <ki7mt@yahoo.com> - 2.0.2-1
- Fix RPATH issue: remove build-time path from wspr-cuda-check binary
- Fix spec: add missing bulk_kernels.h to devel package

* Mon Jan 20 2025 Greg Beam <ki7mt@yahoo.com> - 2.0.1-1
- Add sm_120 (Blackwell refresh) to fat binary targets
- Add bulk_kernels.cu: SoA-based bulk processing kernels for CGO
- Sync version with ki7mt-ai-lab-core v2.0.1

* Fri Jan 17 2025 Greg Beam <ki7mt@yahoo.com> - 1.1.7-1
- Add wspr_structs.h: RTX 5090-optimized 128-byte struct synchronized with ClickHouse
- Add WSPRSpotCH: 99-byte ClickHouse RowBinary format struct
- Add WSPR_STRIP_PADDING macro for GPU to ClickHouse conversion
- Add verify_layout.c: Memory layout verification utility
- Add verify_ingestion.sh: End-to-end ClickHouse ingestion test
- Add wspr_structs_test.c: Compile-time static assertion tests
- Install wspr_structs.h to include directory for CGO development

* Fri Jan 17 2025 Greg Beam <ki7mt@yahoo.com> - 1.1.6-1
- Add spec changelog for v1.1.5 and v1.1.6

* Fri Jan 17 2025 Greg Beam <ki7mt@yahoo.com> - 1.1.5-1
- Add --help and --version flags to wspr-cuda-check
- Update bump-version to track source file VERSION define

* Thu Jan 16 2025 Greg Beam <ki7mt@yahoo.com> - 1.1.4-1
- Hardcode Source0 URL to avoid rpkg naming conflicts

* Thu Jan 16 2025 Greg Beam <ki7mt@yahoo.com> - 1.1.3-1
- COPR compatibility for headless GPU-less builds
- Update nvidia-driver-cuda requirement to >= 590.48.01 for RTX 5090 Blackwell
- Add EXTRA_NVCCFLAGS for -allow-unsupported-compiler flag
- Embed proper SONAME in shared library ELF header
- Replace manual post/postun with ldconfig_scriptlets (EL9 standard)
- Switch to GitHub archive Source0 for COPR builds
