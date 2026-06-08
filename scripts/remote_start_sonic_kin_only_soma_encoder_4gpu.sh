#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default is the LR-273 temporal-consistency loss-on treatment. The matched
# LR-274 loss-off baseline uses
# CONFIG=configs/sonic_kin_soma_motionlib_proportional_loss_off_baseline_4gpu.json.
export CONFIG="${CONFIG:-configs/sonic_kin_soma_motionlib_proportional_4gpu.json}"

exec "${SCRIPT_DIR}/remote_start_sonic_kin_soma_motionlib_4gpu.sh" "$@"
