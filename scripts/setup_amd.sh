#!/bin/bash
# ============================================================================
# Vendor Risk Intel — AMD MI300X Setup Script
# Run this once after cloning the repo on the AMD Developer Cloud
# Usage: bash scripts/setup_amd.sh
# ============================================================================

set -e
echo "============================================"
echo " Vendor Risk Intel — AMD MI300X Setup"
echo "============================================"

# ── 1. Check ROCm ─────────────────────────────────────────────────────────────
echo ""
echo "[1/8] Checking ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname 2>/dev/null | head -5
    echo "✓ ROCm detected"
else
    echo "⚠ rocm-smi not found — ensure ROCm is installed"
    echo "  See: https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html"
fi

# ── 2. Python environment ──────────────────────────────────────────────────────
echo ""
echo "[2/8] Installing base Python requirements..."
pip install -r requirements.txt --quiet
echo "✓ Base requirements installed"

# ── 3. Clean Slate ─────────────────────────────────────────────────────────────
echo ""
echo "[3/8] Cleaning previous PyTorch/vLLM installations to avoid conflicts..."
pip uninstall -y vllm torch torchvision torchaudio pytorch-triton-rocm triton --quiet
echo "✓ Environment slate wiped clean"

# ── 4. vLLM & Dependency Resolution ────────────────────────────────────────────
echo ""
echo "[4/8] Installing vLLM (pinned version) and resolving Numpy conflicts..."
# Pin vLLM under 0.7.0 to match the ROCm 6.1 / PyTorch 2.6.0 era
pip install --no-cache-dir "vllm>=0.4.2,<0.7.0" "numpy<2.0.0" --quiet
# Strip out the CUDA Triton that vLLM quietly pulls in
pip uninstall -y triton --quiet
echo "✓ Pinned vLLM installed"

# ── 5. PyTorch with ROCm (Forced Overwrite) ──────────────────────────────────
echo ""
echo "[5/8] Forcing PyTorch ROCm installation..."
pip install --no-cache-dir --force-reinstall torch torchvision torchaudio pytorch-triton-rocm \
    --index-url https://download.pytorch.org/whl/rocm6.1 \
    --quiet
echo "✓ PyTorch ROCm forced successfully"

# Verify GPU access
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" || echo "⚠ PyTorch GPU check failed — continuing"

# ── 6. AMD-specific requirements ─────────────────────────────────────────────
echo ""
echo "[6/8] Installing additional AMD requirements..."
pip install -r requirements-amd.txt --quiet
echo "✓ AMD requirements installed"

# ── 7. Download model ────────────────────────────────────────────────────────
echo ""
echo "[7/8] Downloading Qwen2.5-14B-Instruct-GPTQ-Int4 (~9GB)..."
echo "  This will be saved to ./models/ inside your 25GB persistent storage"
echo "  Estimated time: 5-10 minutes depending on network speed"

mkdir -p models

# Use huggingface-cli if available, else fall back to Python
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli download \
        Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \
        --local-dir ./models/Qwen2.5-14B-Instruct-GPTQ-Int4 \
        --local-dir-use-symlinks False
else
    pip install huggingface_hub --quiet
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4',
    local_dir='./models/Qwen2.5-14B-Instruct-GPTQ-Int4',
)
print('Model downloaded successfully')
"
fi
echo "✓ Model downloaded"

# ── 8. Environment file ──────────────────────────────────────────────────────
echo ""
echo "[8/8] Setting up .env for AMD..."
if [ ! -f .env ]; then
    cp .env.example .env
fi

# Update .env values for AMD
sed -i 's/^LLM_BACKEND=.*/LLM_BACKEND=vllm/' .env
sed -i 's|^VLLM_MODEL_NAME=.*|VLLM_MODEL_NAME=./models/Qwen2.5-14B-Instruct-GPTQ-Int4|' .env
echo "✓ .env configured for vLLM"

# ── Print vLLM start command ──────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "Start vLLM server (run in a separate terminal or background):"
echo ""
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "      --model ./models/Qwen2.5-14B-Instruct-GPTQ-Int4 \\"
echo "      --quantization gptq \\"
echo "      --dtype float16 \\"
echo "      --max-model-len 8192 \\"
echo "      --gpu-memory-utilization 0.90 \\"
echo "      --host 0.0.0.0 --port 8000 &"
echo ""
echo "Then run the demo pipeline:"
echo "  python scripts/generate_demo.py --backend vllm"
echo ""
echo "Or the full CLI:"
echo "  python scripts/run_pipeline.py --company 'Apple Inc' --ticker AAPL --backend vllm --open"
echo ""
echo "Storage check:"
du -sh data/ models/ 2>/dev/null || true
df -h . | tail -1