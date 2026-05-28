#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${CONFIG:-configs/sonic_kin_only_soma_encoder_proportional.json}"

exec "${SCRIPT_DIR}/remote_start_sonic_native_retarget_4gpu.sh" "$@"
