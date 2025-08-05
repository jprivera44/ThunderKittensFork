#!/bin/bash
# setup_cuda12.6_tk.sh - Complete setup script for CUDA 12.6 and ThunderKittens

set -e  # Exit on error

echo "=== Setting up CUDA 12.6 for ThunderKittens Development ==="
echo

# Function to print colored output
print_status() {
    echo -e "\033[1;32m[✓]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[✗]\033[0m $1"
}

# Check if running as root (required for RunPod)
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root"
   exit 1
fi

# Step 1: Download and install CUDA 12.6 toolkit
print_status "Downloading CUDA 12.6 toolkit..."
cd /tmp
wget -q --show-progress https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run

print_status "Installing CUDA 12.6 (toolkit only, no driver)..."
sh cuda_12.6.0_560.28.03_linux.run --toolkit --silent --override
rm -f cuda_12.6.0_560.28.03_linux.run

# Step 2: Set up CUDA environment variables
print_status "Setting up CUDA environment variables..."
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Make permanent
echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
echo 'export PATH=${CUDA_HOME}/bin:${PATH}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc

# Step 3: Clean up package caches to save space
print_status "Cleaning up package caches..."
pip cache purge
apt-get clean
apt-get autoremove -y

# Step 4: Set up pip to use volume for large installs
print_status "Configuring pip to use volume storage..."
mkdir -p /workspace/.pip/cache /workspace/.pip/tmp
export PIP_CACHE_DIR=/workspace/.pip/cache
export TMPDIR=/workspace/.pip/tmp

# Make pip config permanent
echo 'export PIP_CACHE_DIR=/workspace/.pip/cache' >> ~/.bashrc
echo 'export TMPDIR=/workspace/.pip/tmp' >> ~/.bashrc

# Step 5: Install PyTorch 2.6.0 with CUDA 12.6 support
print_status "Installing PyTorch 2.6.0 with CUDA 12.6 support..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Step 6: Install GCC 11 for C++20 support (ThunderKittens requirement)
print_status "Installing GCC 11 for C++20 support..."
apt update -qq
apt install -y gcc-11 g++-11 clang-11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11

# Step 7: Verify installations
print_status "Verifying installations..."
echo
echo "=== Installation Summary ==="
echo -n "CUDA Version: "
nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1
echo -n "GCC Version: "
gcc --version | head -n1
echo -n "Python Version: "
python --version
echo

# Test PyTorch and CUDA
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version in PyTorch: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"


echo
print_status "Setup complete! 🎉"
echo
echo "=== Next Steps ==="
echo "1. Source your environment: source ~/.bashrc"
echo