#!/usr/bin/env bash

if [ -n "${_AIBIRDS_TF215_HOOK_ACTIVE:-}" ]; then
    if [ -n "${_AIBIRDS_TF215_OLD_LD_LIBRARY_PATH:-}" ]; then
        export LD_LIBRARY_PATH="$_AIBIRDS_TF215_OLD_LD_LIBRARY_PATH"
    else
        unset LD_LIBRARY_PATH
    fi

    if [ -n "${_AIBIRDS_TF215_OLD_LD_PRELOAD:-}" ]; then
        export LD_PRELOAD="$_AIBIRDS_TF215_OLD_LD_PRELOAD"
    else
        unset LD_PRELOAD
    fi

    if [ -n "${_AIBIRDS_TF215_OLD_PYTHONPATH:-}" ]; then
        export PYTHONPATH="$_AIBIRDS_TF215_OLD_PYTHONPATH"
    else
        unset PYTHONPATH
    fi

    if [ -n "${_AIBIRDS_TF215_OLD_MPLCONFIGDIR:-}" ]; then
        export MPLCONFIGDIR="$_AIBIRDS_TF215_OLD_MPLCONFIGDIR"
    else
        unset MPLCONFIGDIR
    fi
fi

unset _AIBIRDS_TF215_OLD_LD_LIBRARY_PATH
unset _AIBIRDS_TF215_OLD_LD_PRELOAD
unset _AIBIRDS_TF215_OLD_PYTHONPATH
unset _AIBIRDS_TF215_OLD_MPLCONFIGDIR
unset _AIBIRDS_TF215_HOOK_ACTIVE
