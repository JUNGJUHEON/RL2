#!/usr/bin/env bash

# TensorFlow 2.15 can fail if this env inherits shared-library paths from
# base, another conda env, CUDA experiments, or an IDE terminal session.
# Keep a copy so deactivation can restore the user's shell.
if [ -z "${_AIBIRDS_TF215_HOOK_ACTIVE:-}" ]; then
    export _AIBIRDS_TF215_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
    export _AIBIRDS_TF215_OLD_LD_PRELOAD="${LD_PRELOAD-}"
    export _AIBIRDS_TF215_OLD_PYTHONPATH="${PYTHONPATH-}"
    export _AIBIRDS_TF215_OLD_MPLCONFIGDIR="${MPLCONFIGDIR-}"
    export _AIBIRDS_TF215_HOOK_ACTIVE=1
fi

unset LD_LIBRARY_PATH
unset LD_PRELOAD
unset PYTHONPATH

export MPLCONFIGDIR="/home/jung/RL2/.runtime/matplotlib"
mkdir -p "$MPLCONFIGDIR"
