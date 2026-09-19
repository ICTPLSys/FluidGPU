#!/usr/bin/env bash
set -euo pipefail

python_bin="${FLUIDGPU_PYTHON:-python3}"
venv_dir="${FLUIDGPU_VENV:-$HOME/fluidgpu-ae}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "setup_env.sh: Python interpreter not found: $python_bin" >&2
  echo "              set FLUIDGPU_PYTHON to a Python 3.12.13 executable" >&2
  exit 1
fi

python_version="$("$python_bin" -c 'import platform; print(platform.python_version())')"
if [[ "$python_version" != "3.12.13" ]]; then
  echo "setup_env.sh: expected Python 3.12.13, found $python_version" >&2
  echo "              set FLUIDGPU_PYTHON to a Python 3.12.13 executable" >&2
  exit 1
fi

if [[ -x "${venv_dir}/bin/python" ]]; then
  echo "setup_env.sh: virtual environment already exists: $venv_dir"
else
  "$python_bin" -m venv "$venv_dir"
fi

source "${venv_dir}/bin/activate"
venv_python_version="$(python -c 'import platform; print(platform.python_version())')"
if [[ "$venv_python_version" != "3.12.13" ]]; then
  echo "setup_env.sh: existing venv uses Python $venv_python_version, expected 3.12.13" >&2
  echo "              choose a new FLUIDGPU_VENV directory" >&2
  exit 1
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e fluidgpu_runtime

echo "setup_env.sh: ready: $venv_dir"
echo "Run: source ${venv_dir}/bin/activate"
