#!/bin/sh
# Run a Python script from this folder under a hard address-space cap so a
# runaway allocation can never take the VM down.
set -e
cd "$(dirname "$0")"
ulimit -v "${EVAL_DEMO_MEM_KB:-6000000}"
exec ../../../.venv/bin/python "$@"
