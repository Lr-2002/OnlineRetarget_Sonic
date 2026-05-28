#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${CONFIG:-configs/sonic_kin_soma_motionlib_proportional_4gpu.json}"

exec "${SCRIPT_DIR}/remote_start_sonic_kin_soma_motionlib_4gpu.sh" "$@"
