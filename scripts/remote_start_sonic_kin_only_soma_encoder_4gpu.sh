#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Compatibility wrapper for older kin-only SOMA encoder launch instructions.
# The active supervised launcher is remote_start_sonic_kin_soma_motionlib_4gpu.sh.
# Default is the LR-273 temporal-consistency loss-on treatment. The matched
# LR-274 loss-off baseline uses
# CONFIG=configs/sonic_kin_soma_motionlib_proportional_loss_off_baseline_4gpu.json.
# LR-280 kin/walk package runs should pass one of the explicit
# sonic_kin_soma_motionlib_kin_walk_data_package_*_4gpu configs.
export CONFIG="${CONFIG:-configs/sonic_kin_soma_motionlib_proportional_4gpu.json}"

exec "${SCRIPT_DIR}/remote_start_sonic_kin_soma_motionlib_4gpu.sh" "$@"
