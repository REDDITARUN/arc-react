#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch matplotlib

if [ ! -d "data/canonical_train/tasks" ]; then
  echo "Preparing ARC data"
  python scripts/prepare_data.py --workspace .
else
  echo "Prepared data already exists, skipping data prep"
fi

echo ""
echo "Setup complete."
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Smoke test with: python3 train.py --smoke-test --device cpu --batch-size 1"
