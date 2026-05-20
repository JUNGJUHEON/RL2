#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="/home/jung/anaconda3/envs/aibirds_tf215"

cd "$ROOT_DIR"

# Keep this launcher isolated from any previously activated conda env.
# A stale libtensorflow/libprotobuf path can make TF 2.15 fail at import time.
unset LD_LIBRARY_PATH
unset LD_PRELOAD
unset PYTHONPATH

export CONDA_PREFIX="$ENV_DIR"
export PATH="$ENV_DIR/bin:/home/jung/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export MPLCONFIGDIR="$ROOT_DIR/.runtime/matplotlib"
mkdir -p "$MPLCONFIGDIR"

exec "$ENV_DIR/bin/python" "$@"
