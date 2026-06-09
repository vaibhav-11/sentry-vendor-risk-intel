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
echo "[1/7] Checking ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname 2>/dev/null | head -5
    echo "✓ ROCm detected"
else
    echo "⚠ rocm-smi not found — ensure ROCm is installed"
    echo "  See: https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html"
fi

# ── 2. Python environment ──────────────────────────────────────────────────────
echo ""
echo "[2/7] Installing base Python requirements..."
pip install -r requirements.txt --quiet
echo "✓ Base requirements installed"

# ── 3. PyTorch with ROCm ──────────────────────────────────────────────────────
echo ""
echo "[3/7] Installing PyTorch with ROCm support..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1 \
    --quiet 2>&1 | tail -3
echo "✓ PyTorch ROCm installed"

# Verify GPU access
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" || echo "⚠ PyTorch GPU check failed — continuing"

# ── 4. vLLM ──────────────────────────────────────────────────────────────────
echo ""
echo "[4/7] Installing vLLM..."
pip install vllm --quiet 2>&1 | tail -3
echo "✓ vLLM installed"

# ── 5. AMD-specific requirements ─────────────────────────────────────────────
echo ""
echo "[5/7] Installing AMD requirements..."
pip install -r requirements-amd.txt --quiet 2>&1 | tail -3
echo "✓ AMD requirements installed"

# ── 6. Download model ────────────────────────────────────────────────────────
echo ""
echo "[6/7] Downloading Qwen2.5-14B-Instruct-GPTQ-Int4 (~9GB)..."
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

# ── 7. Environment file ──────────────────────────────────────────────────────
echo ""
echo "[7/7] Setting up .env for AMD..."
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
