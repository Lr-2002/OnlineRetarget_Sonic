# Implementation Log

All times use the local workspace timezone unless otherwise stated.

## 2026-05-13

- Initialized execution goal: implement M1-M7 with live logs and traceable artifacts.
- Current baseline commit before this goal: `aa84901 Establish a traceable retargeting baseline scaffold`.
- Scope decision: implement runnable M1-M4 artifacts first; M5-M7 require torch/Isaac/hardware and will receive executable scaffolds plus explicit blockers until environment is available.
- Verification rule: every milestone claim must map to files, commands, or documented blockers.
- Milestone revision: paper survey is now M1, and data split work is gated behind motion curation/quality flags because source motion and retargeted humanoid targets may be noisy or physically invalid.
- Curation research update: added NMR/CEPR, PHUMA, ExBody2, and KungFuAthlete as filtering references; current direction is source-motion curation + humanoid-target curation + physics-refinement labels, with quality scores instead of only binary keep/drop.
- M2 implementation update: added standard-library split/curation/index code in `src/online_retarget/data/curation.py`, CLI command `split-index`, and tests in `tests/test_bones_seed_index.py`.
- Verification: `PYTHONPATH=src python3 -m unittest discover -s tests` passed 10 tests.
- Real data smoke: `PYTHONPATH=src python3 scripts/inspect_bones_seed.py split-index --data-root /home/user/data/motion_data --output-root runs --seed 17 --train-ratio 0.8 --val-ratio 0.1 --policy-name metadata_balanced_v0 --min-duration-frames 60` wrote `runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0`.
- Real data result: 142,220 rows, 522 actors, actor split train/val/test = 417/52/53, row split train/val/test = 112,789/15,760/13,671, curation actions keep/downweight = 71,132/71,088, mirror flag count = 71,088.
- Added G1 target quality scanner in `src/online_retarget/data/g1_quality.py` plus CLI command `scan-g1-quality` and tests in `tests/test_g1_quality.py`.
- Real G1 quality smoke: scanned first 100 indexed G1 CSV targets from read-only `g1.tar` into `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit100`.
- G1 quality smoke result with temporary thresholds (`fps=30`, max joint velocity 20 rad/s, max root speed 8 m/s): 29 keep, 71 quarantine, flags `joint_velocity_jump=71`, `root_discontinuity=28`; metric summary recorded in `g1_quality_report.json`.
- Current M2 limitation: thresholds are not calibrated yet and only G1 target CSV stats are scanned. Source skeleton stats, foot slide/float, penetration, and self-intersection still need scanners before M2 is fully closed.
- M3 implementation update: added dataset schema and observation/output contract dataclasses in `src/online_retarget/data/schema.py`, including `MotionPairRef`, `ObservationSpec`, `RobotStateSpec`, and `OutputSpec`.
- Split index update: metadata index now includes morphology columns needed by `MotionPairRef`; regenerated `runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0`.
- Verification: `PYTHONPATH=src python3 -m unittest discover -s tests` passed 14 tests after M3 schema additions.
- M4 implementation update: added `src/online_retarget/evaluation.py` and CLI command `offline-eval` for JSONL prediction/target evaluation, with summary JSON, per-sample CSV, failure manifest, and actor/category/package/quality-flag aggregation.
- Verification: `PYTHONPATH=src python3 -m unittest discover -s tests` passed 15 tests after M4 eval additions; CLI help smoke passed for `split-index`, `scan-g1-quality`, and `offline-eval`.
- M5 scaffold update: `scripts/train.py --dry-run` now reads config/index/schema, reports git SHA/dirty, DDP rank/world size, observation/output dims, and sample refs without starting optimization.
- M5 dry-run evidence: `PYTHONPATH=src python3 scripts/train.py --config configs/baseline_mlp.yaml --dry-run --limit 1` reported observation dim 1547, output dim 29, train refs 112,789.
- M6 scaffold update: added `scripts/benchmark_latency.py`; dry-run writes benchmark contract to `runs/benchmarks/baseline_mlp_dry_run.json`.
- M7 scaffold update: added `scripts/eval_isaac.py`; dry-run writes `runs/isaac_eval/dry_run/isaac_eval_status.json` with expected simulator metrics.
- Current M5-M7 blockers: tensor window extraction and optimizer loop are not implemented; real latency requires torch/CUDA/4090; real simulator eval requires Isaac Lab and G1 replay/tracking task binding.
- Status audit added at `docs/status/m1_m7_status.md`; M1-M7 are not complete yet, but current artifacts and blockers are explicitly mapped.

## 2026-05-14

- Planning update: added M2Q as an explicit motion-quality curation gate in `docs/milestones.md`. This gate covers source human motion, G1 target motion, pair consistency, physics provenance, calibrated thresholds, diversity-loss reporting, and worst-clip manifests.
- Research update: added `docs/research/motion_quality_curation.md`, synthesizing NMR/CEPR, PHUMA/PhySINK, GMR, OmniTrack, OmniRetarget, KDMR, contact/dynamics, self-contact retargeting, and foot-contact detection references into implementation-ready quality signals.
- Paper tracking update: expanded `docs/research/paper_matrix.md`, `docs/research/literature_review.md`, and `docs/research/pdf_manifest.md` so motion filtering is traceable to papers rather than only local intuition.
- M2 source curation update: added `src/online_retarget/data/bvh_quality.py`, CLI command `scan-source-quality`, and `tests/test_bvh_quality.py`.
- Real source quality smoke: `PYTHONPATH=src python3 scripts/inspect_bones_seed.py scan-source-quality --data-root /home/user/data/motion_data --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv --output-root runs --limit 100 --max-channel-velocity 3000 --max-root-speed 500` wrote `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_limit100`.
- Source quality result: 100 scanned, 59 keep, 40 quarantine, 1 exclude; flags `source_channel_jump=40`, `nonfinite_value=1`, `channel_width_mismatch=1`.
- Current M2 limitation after source scanner: contact, foot slide/float, penetration, self-intersection, and calibrated threshold policy are still pending.
- M2 threshold calibration update: added `src/online_retarget/data/thresholds.py`, CLI command `propose-thresholds`, and `tests/test_thresholds.py`.
- Threshold proposal smoke: generated p95 proposals for G1 stats at `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit100/g1_threshold_proposals_p95.json` and source stats at `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_limit100/source_threshold_proposals_p95.json`.
- Verification: `PYTHONPATH=src python3 -m unittest discover -s tests` passed 19 tests after threshold proposal additions.
- M5 data-path update: added `src/online_retarget/data/supervised_builder.py`, CLI command `build-supervised-jsonl`, and `tests/test_supervised_builder.py`.
- Real supervised debug artifact: `PYTHONPATH=src python3 scripts/inspect_bones_seed.py build-supervised-jsonl --data-root /home/user/data/motion_data --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv --output-root runs --split train --limit 8 --history-frames 8` wrote `runs/supervised/train_h8_limit8`.
- Supervised debug result: 8 samples, 0 skipped, raw-BVH-channel input dim 1933, G1 output dim 29. This is a data-path debug builder, not the final 30-body observation contract.
- M2 merge update: added `src/online_retarget/data/quality_merge.py`, CLI command `merge-quality`, and `tests/test_quality_merge.py`.
- Curated index smoke: merged split index with first-100 source and G1 quality stats into `runs/curated/smoke_source_g1_limit100/curated_index.csv`.
- Curated index result: 142,220 rows; merged source rows 100, merged G1 rows 100; actions keep/downweight/quarantine/exclude = 71,095/71,051/73/1.
- M5 builder update: `build-supervised-jsonl` now supports `--action-column`, allowing debug sample generation from `merged_quality_action` in curated indexes.
- Curated supervised debug artifact: generated `runs/supervised/train_merged-quality-action_h8_limit8` from `runs/curated/smoke_source_g1_limit100/curated_index.csv` with merged actions keep/downweight only.
- M5 training update: `scripts/train.py` now has a real PyTorch optimizer loop for supervised JSONL artifacts (`--samples-jsonl`) and saves `checkpoint.pt` plus `train_report.json` when torch is available.
- M5 dry-run evidence: `PYTHONPATH=src:. python3 scripts/train.py --config configs/baseline_mlp.yaml --samples-jsonl runs/supervised/train_merged-quality-action_h8_limit8/samples.jsonl --dry-run --limit 1` succeeds and reports the samples path.
- M5 environment blocker evidence: non-dry-run exits with `Training requires the conda environment from environment.yml with torch installed.` in the current base Python, which lacks torch/numpy.
- M5/M2Q gate update: `scripts/train.py` now collects a `quality_gate` context and refuses formal non-dry-run training unless a quality policy ID, existing quality report, curated index path, and `merged_quality_action` are present. `--allow-debug-data` exists only for explicitly labeled debug runs.
- Config update: `configs/baseline_mlp.yaml` now points to the smoke curated index, merged-action supervised JSONL, quality policy ID `smoke_source_g1_limit100`, and curated report path.
- M2Q artifact update: `merge-quality` now writes `worst_clips.csv` and a curated-report breakdown by split, package, and category. Regenerated `runs/curated/smoke_source_g1_limit100`, producing 74 worst-clip rows plus the 142,220-row curated index.
