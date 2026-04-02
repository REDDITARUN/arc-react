#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
VENV_PYTHON="${VENV_DIR}/bin/python"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required for data preparation but was not found." >&2
  exit 1
fi

echo "Creating virtual environment at ${VENV_DIR}"
if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
  echo "" >&2
  echo "Failed to create virtual environment." >&2
  echo "On Debian/Ubuntu, install venv support with:" >&2
  echo "  sudo apt install python3-venv" >&2
  echo "Then delete '${VENV_DIR}' and rerun this script." >&2
  exit 1
fi

"${VENV_PYTHON}" -m pip install --upgrade pip wheel

if [ -z "${TORCH_INDEX_URL}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VERSION="$(nvidia-smi | awk '/CUDA Version:/ { for (i = 1; i <= NF; i++) { if ($i == "Version:") { version = $(i + 1) } } } END { print version }')"
    case "${CUDA_VERSION}" in
      12.8*)
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
        ;;
      12.6*)
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
        ;;
      11.8*)
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
        ;;
      *)
        echo "Unsupported or unknown CUDA version '${CUDA_VERSION:-unknown}' from nvidia-smi." >&2
        echo "Set TORCH_INDEX_URL explicitly before rerunning setup.sh." >&2
        echo "Example: TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash setup.sh" >&2
        exit 1
        ;;
    esac
  else
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
  fi
fi

echo "Installing PyTorch from ${TORCH_INDEX_URL}"
"${VENV_PYTHON}" -m pip install --upgrade --force-reinstall --index-url "${TORCH_INDEX_URL}" torch
"${VENV_PYTHON}" -m pip install --upgrade matplotlib tqdm

if command -v nvidia-smi >/dev/null 2>&1 && [[ "${TORCH_INDEX_URL}" != *"/cpu" ]]; then
  "${VENV_PYTHON}" - <<'PY'
import sys
import torch

print(f"Installed torch {torch.__version__} (CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA verification failed. Check the selected PyTorch wheel and NVIDIA driver."
    )
print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
PY
fi

if [ ! -d "data_arc_agi_1/canonical_train/tasks" ]; then
  echo "Preparing ARC-AGI-1 data (see scripts/prepare_data.py for options)"
  "${VENV_PYTHON}" scripts/prepare_data.py --workspace .
else
  echo "Prepared data already exists at data_arc_agi_1/, skipping data prep"
fi

echo ""
echo "Setup complete."
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Smoke test (CPU): ${VENV_PYTHON} train.py --smoke-test --device cpu --batch-size 1"
echo "Smoke test (CUDA): ${VENV_PYTHON} train.py --smoke-test --device cuda --batch-size 1"
