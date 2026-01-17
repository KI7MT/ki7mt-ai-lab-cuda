# Disable debug package generation (CUDA binaries don't have standard debug info)
%global debug_package %{nil}

Name:           ki7mt-ai-lab-cuda
Version:        1.1.6
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
%doc README
# Shared library with soname versioning
%{_libdir}/lib%{name}.so.*
# Check utility
%{_bindir}/wspr-cuda-check

%files devel
%doc src/cuda/README.md
# Header file (in /usr/include/ki7mt/)
%dir %{_includedir}/ki7mt
%{_includedir}/ki7mt/bridge.h
# Static library
%{_libdir}/lib%{name}.a
# Development symlink
%{_libdir}/lib%{name}.so
# Source files for reference/rebuilding
%dir %{_datadir}/%{name}
%dir %{_datadir}/%{name}/src
%{_datadir}/%{name}/src/*.cu
%{_datadir}/%{name}/src/*.h

%changelog
* Sat Jan 17 2026 Greg Beam <ki7mt@outlook.com> - 1.1.6-1
- Add spec changelog for v1.1.5 and v1.1.6

* Sat Jan 17 2026 Greg Beam <ki7mt@outlook.com> - 1.1.5-1
- Add --help and --version flags to wspr-cuda-check
- Update bump-version to track source file VERSION define

* Fri Jan 16 2026 Greg Beam <ki7mt@outlook.com> - 1.1.4-1
- Hardcode Source0 URL to avoid rpkg naming conflicts

* Fri Jan 16 2026 Greg Beam <ki7mt@outlook.com> - 1.1.3-1
- COPR compatibility for headless GPU-less builds
- Update nvidia-driver-cuda requirement to >= 590.48.01 for RTX 5090 Blackwell
- Add EXTRA_NVCCFLAGS for -allow-unsupported-compiler flag
- Embed proper SONAME in shared library ELF header
- Replace manual post/postun with ldconfig_scriptlets (EL9 standard)
- Switch to GitHub archive Source0 for COPR builds
