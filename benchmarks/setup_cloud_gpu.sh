#!/bin/bash
# rust-cuda Benchmark Environment Setup for Cloud GPU Instances
# Works on: RunPod, Lambda Labs, Vast.ai, AWS, GCP, Azure, etc.
#
# Usage: bash setup_cloud_gpu.sh [INSTALL_DIR]
#
# Requirements:
# - Ubuntu 22.04 or 24.04 (other distros may work with modifications)
# - CUDA 12.0+ pre-installed
# - NVIDIA GPU with drivers
# - Internet connection

set -e  # Exit on error

# Configuration
INSTALL_DIR="${1:-$HOME/rust-cuda-benchmarks}"
LLVM_DIR="/opt/llvm-7"
RUST_CUDA_REPO="https://github.com/cong-or/rust-cuda.git"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
}

echo "=========================================="
echo "rust-cuda Benchmark Environment Setup"
echo "=========================================="
echo ""
echo "Install directory: $INSTALL_DIR"
echo ""

# Detect if running as root
IS_ROOT=false
if [ "$EUID" -eq 0 ]; then
    IS_ROOT=true
    log_info "Running as root"
else
    log_info "Running as non-root user"
fi

# Check for sudo if not root
SUDO=""
if [ "$IS_ROOT" = false ]; then
    if command -v sudo &> /dev/null; then
        SUDO="sudo"
        log_info "Will use sudo for privileged operations"
    else
        log_error "Not running as root and sudo not available"
        exit 1
    fi
fi

# Step 1: Verify prerequisites
log_step "Step 1: Verifying Prerequisites"

if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. NVIDIA drivers not installed?"
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    log_error "nvcc not found. CUDA toolkit not installed?"
    log_error "Install CUDA from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)

log_info "GPU: $GPU_NAME"
log_info "CUDA Version: $CUDA_VERSION"
log_info "Driver Version: $DRIVER_VERSION"

# Check CUDA version is >= 12.0
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
if [ "$CUDA_MAJOR" -lt 12 ]; then
    log_error "CUDA 12.0+ required, found $CUDA_VERSION"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_NAME=$NAME
    OS_VERSION=$VERSION_ID
    log_info "OS: $OS_NAME $OS_VERSION"
else
    log_warn "Cannot detect OS version"
fi

# Step 2: Install system dependencies
log_step "Step 2: Installing System Dependencies"

if [ "$OS_NAME" = "Ubuntu" ] || [ "$OS_NAME" = "Debian GNU/Linux" ]; then
    log_info "Installing dependencies via apt..."
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq build-essential cmake ninja-build curl git \
        pkg-config libssl-dev wget ca-certificates > /dev/null 2>&1
    log_info "✓ System dependencies installed"
else
    log_warn "Non-Ubuntu/Debian OS detected. You may need to install dependencies manually:"
    log_warn "  - build-essential (gcc, g++, make)"
    log_warn "  - cmake, ninja-build"
    log_warn "  - curl, git, wget"
    log_warn "  - pkg-config, libssl-dev"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 3: Install Rust
log_step "Step 3: Installing Rust"

if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    log_warn "Rust already installed: $RUST_VERSION"
else
    log_info "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --profile minimal --default-toolchain none > /dev/null 2>&1

    # Load cargo environment
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi

    RUST_VERSION=$(rustc --version)
    log_info "✓ Rust installed: $RUST_VERSION"
fi

# Step 4: Build LLVM 7.1.0
log_step "Step 4: Building LLVM 7.1.0"

if [ -f "$LLVM_DIR/bin/llvm-config" ]; then
    LLVM_VERSION=$($LLVM_DIR/bin/llvm-config --version)
    log_warn "LLVM already installed at $LLVM_DIR (version $LLVM_VERSION)"
else
    log_info "Building LLVM 7.1.0 (this takes ~20-40 minutes)..."
    log_info "You can safely background this process if needed"

    # Create temporary build directory
    BUILD_DIR=$(mktemp -d)
    cd $BUILD_DIR

    log_info "Downloading LLVM 7.1.0..."
    wget -q --show-progress https://github.com/llvm/llvm-project/releases/download/llvmorg-7.1.0/llvm-7.1.0.src.tar.xz

    log_info "Extracting..."
    tar -xf llvm-7.1.0.src.tar.xz
    cd llvm-7.1.0.src
    mkdir -p build && cd build

    # Detect architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        TARGETS="X86;NVPTX"
    elif [ "$ARCH" = "aarch64" ]; then
        TARGETS="AArch64;NVPTX"
    else
        log_warn "Unknown architecture $ARCH, using X86;NVPTX"
        TARGETS="X86;NVPTX"
    fi

    log_info "Configuring LLVM build for $ARCH..."
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_TARGETS_TO_BUILD="$TARGETS" \
        -DLLVM_BUILD_LLVM_DYLIB=ON \
        -DLLVM_LINK_LLVM_DYLIB=ON \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DLLVM_ENABLE_BINDINGS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_ENABLE_ZLIB=ON \
        -DLLVM_ENABLE_TERMINFO=ON \
        -DCMAKE_INSTALL_PREFIX=$LLVM_DIR \
        .. > /dev/null

    log_info "Building LLVM (using $(nproc) cores)..."
    ninja -j$(nproc)

    log_info "Installing LLVM to $LLVM_DIR..."
    $SUDO ninja install > /dev/null

    # Cleanup
    cd /
    rm -rf $BUILD_DIR

    log_info "✓ LLVM 7.1.0 built and installed"
fi

# Create symlink if doesn't exist
if [ ! -e /usr/bin/llvm-config ] && [ ! -L /usr/bin/llvm-config ]; then
    $SUDO ln -s $LLVM_DIR/bin/llvm-config /usr/bin/llvm-config
    log_info "Created symlink: /usr/bin/llvm-config -> $LLVM_DIR/bin/llvm-config"
fi

# Step 5: Set environment variables
log_step "Step 5: Configuring Environment"

# Determine which shell config to use
if [ -n "$BASH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
elif [ -n "$ZSH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
else
    SHELL_CONFIG="$HOME/.profile"
fi

log_info "Using shell config: $SHELL_CONFIG"

# Add environment variables if not already present
grep -qxF 'export LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"' $SHELL_CONFIG 2>/dev/null || \
    echo 'export LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"' >> $SHELL_CONFIG

grep -qxF 'export LLVM_LINK_STATIC=1' $SHELL_CONFIG 2>/dev/null || \
    echo 'export LLVM_LINK_STATIC=1' >> $SHELL_CONFIG

grep -qxF "export PATH=\"$LLVM_DIR/bin:\$HOME/.cargo/bin:\$PATH\"" $SHELL_CONFIG 2>/dev/null || \
    echo "export PATH=\"$LLVM_DIR/bin:\$HOME/.cargo/bin:\$PATH\"" >> $SHELL_CONFIG

# Load environment for current session
export LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
export LLVM_LINK_STATIC=1
export PATH="$LLVM_DIR/bin:$HOME/.cargo/bin:$PATH"

log_info "✓ Environment configured"

# Step 6: Clone rust-cuda
log_step "Step 6: Cloning rust-cuda"

mkdir -p $(dirname "$INSTALL_DIR")
cd $(dirname "$INSTALL_DIR")

if [ -d "$INSTALL_DIR" ]; then
    log_warn "Directory $INSTALL_DIR already exists"
    cd $INSTALL_DIR
    if [ -d ".git" ]; then
        log_info "Pulling latest changes..."
        git pull origin main > /dev/null 2>&1 || true
    fi
else
    log_info "Cloning from $RUST_CUDA_REPO..."
    git clone $RUST_CUDA_REPO $(basename "$INSTALL_DIR") > /dev/null 2>&1
    cd $INSTALL_DIR
fi

log_info "✓ rust-cuda ready at $INSTALL_DIR"

# Step 7: Install Rust toolchain
log_step "Step 7: Installing Rust Toolchain"

log_info "Installing toolchain from rust-toolchain.toml..."
rustup show

# Step 8: Build rust-cuda
log_step "Step 8: Building rust-cuda"

log_info "Building core library (cust)..."
cargo build --release -p cust 2>&1 | grep -E "(Compiling|Finished)" || true
log_info "✓ rust-cuda built successfully"

# Step 9: Test with vecadd
log_step "Step 9: Testing Installation"

cd examples/vecadd
log_info "Running vecadd test..."
if cargo run --release 2>&1 | grep -q "using"; then
    log_info "✓ vecadd test PASSED"
else
    log_error "vecadd test FAILED"
    exit 1
fi

# Step 10: Build benchmarks
log_step "Step 10: Building Benchmarks"

cd "$INSTALL_DIR/benchmarks"
log_info "Building rust-cuda benchmarks..."
cargo build --release 2>&1 | grep -E "(Compiling|Finished)" || true
log_info "✓ rust-cuda benchmarks built"

# Step 11: Build native CUDA benchmarks
log_info "Building native CUDA benchmarks..."
cd native
make > /dev/null 2>&1
log_info "✓ Native CUDA benchmarks built"

# Final summary
echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "System Information:"
echo "  GPU:   $GPU_NAME"
echo "  CUDA:  $CUDA_VERSION"
echo "  Rust:  $RUST_VERSION"
echo "  LLVM:  $(llvm-config --version)"
echo ""
echo "Installation Directory:"
echo "  $INSTALL_DIR"
echo ""
echo "Next Steps:"
echo ""
echo "1. Load environment (if in new shell):"
echo "   source $SHELL_CONFIG"
echo ""
echo "2. Run rust-cuda benchmarks:"
echo "   cd $INSTALL_DIR/benchmarks"
echo "   cargo run --release"
echo ""
echo "3. Run native CUDA benchmarks:"
echo "   cd $INSTALL_DIR/benchmarks/native"
echo "   ./saxpy 10000000"
echo "   ./gemm 1024 1024 1024"
echo "   ./reduction 10000000"
echo ""
echo "4. Run automated comparison:"
echo "   cd $INSTALL_DIR/benchmarks"
echo "   ./run_comparison.sh | tee results.txt"
echo ""
echo "=========================================="
