# M1-M7 Status Audit

Date: 2026-05-14.

Objective: start implementing M1-M7 with live logs and traceable artifacts.

## Verification Commands

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 scripts/inspect_bones_seed.py split-index --data-root /home/user/data/motion_data --output-root runs --seed 17 --train-ratio 0.8 --val-ratio 0.1 --policy-name metadata_balanced_v0 --min-duration-frames 60
PYTHONPATH=src python3 scripts/inspect_bones_seed.py scan-g1-quality --data-root /home/user/data/motion_data --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv --output-root runs --limit 100 --fps 30 --max-joint-velocity 20 --max-root-speed 8
PYTHONPATH=src python3 scripts/inspect_bones_seed.py scan-source-quality --data-root /home/user/data/motion_data --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv --output-root runs --limit 100 --max-channel-velocity 3000 --max-root-speed 500
PYTHONPATH=src python3 scripts/inspect_bones_seed.py scan-source-fk-quality --data-root /home/user/data/motion_data --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv --output-root runs --limit 100 --ground-height 0.0 --fps 30 --frame-stride 2 --max-frames 256 --contact-height-threshold 0.04 --max-contact-slide-speed 0.25 --max-mean-foot-clearance 0.10 --max-penetration-depth 0.03 --min-contact-frame-ratio 0.05
PYTHONPATH=src python3 scripts/inspect_bones_seed.py propose-thresholds --stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit100/g1_quality_stats.jsonl --output-json runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit100/g1_threshold_proposals_p95.json --metric max_abs_joint_velocity --metric joint_jump_rate --metric max_root_speed --percentile 0.95 --action quarantine
PYTHONPATH=src python3 scripts/inspect_bones_seed.py merge-quality --split-index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv --source-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_limit100/source_bvh_quality_stats.jsonl --g1-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit100/g1_quality_stats.jsonl --output-root runs --run-name smoke_source_g1_limit100
PYTHONPATH=src python3 scripts/train.py --config configs/baseline_mlp.yaml --dry-run --limit 1
python3 scripts/benchmark_latency.py --dry-run --output-json runs/benchmarks/baseline_mlp_dry_run.json
python3 scripts/eval_isaac.py --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv --output-root runs --run-name dry_run --dry-run
PYTHONPATH=src python3 scripts/inspect_bones_seed.py build-supervised-jsonl --data-root /home/user/data/motion_data --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv --output-root runs --split train --limit 8 --history-frames 8
PYTHONPATH=src python3 scripts/inspect_bones_seed.py build-supervised-jsonl --data-root /home/user/data/motion_data --index-csv runs/curated/smoke_source_g1_limit100/curated_index.csv --output-root runs --split train --curation-action keep --curation-action downweight --action-column merged_quality_action --limit 8 --history-frames 8
PYTHONPATH=src python3 scripts/inspect_bones_seed.py build-windowed-jsonl --data-root /home/user/data/motion_data --index-csv runs/curated/smoke_source_g1_limit100/curated_index.csv --output-root runs --split train --curation-action keep --curation-action downweight --action-column merged_quality_action --limit 4 --history-frames 8
PYTHONPATH=src:. python3 scripts/train.py --config configs/baseline_mlp.yaml --dry-run --limit 1
PYTHONPATH=src:. python3 scripts/train.py --config configs/baseline_mlp.yaml --samples-jsonl runs/supervised/train_merged-quality-action_h8_limit8/samples.jsonl --max-steps 1
```

Latest unit-test evidence: 39 tests passed with `PYTHONPATH=src:. python3 -m unittest discover -s tests`.

## Checklist

| Milestone | Status | Evidence | Missing / blocker |
| --- | --- | --- | --- |
| M1 Paper survey and design matrix | Partial but usable | `docs/research/paper_matrix.md`, `docs/research/literature_review.md`, `docs/research/bibliography.bib`, `docs/research/pdf_manifest.md`, per-paper notes under `docs/research/papers/` | Deeper PDF reading and reference/citation expansion still needed before final model ablations |
| M2 Data inventory, curation, splits | Partial | `src/online_retarget/data/curation.py`, `src/online_retarget/data/g1_quality.py`, `src/online_retarget/data/bvh_quality.py`, `src/online_retarget/data/source_fk_quality.py`, `src/online_retarget/data/thresholds.py`, `src/online_retarget/data/quality_merge.py`, CLI `split-index`, `scan-g1-quality`, `scan-source-quality`, `scan-source-fk-quality`, `propose-thresholds`, `merge-quality`, related tests, real index and quality artifacts under `runs/` | M2Q quality gate is now explicit; G1 FK/contact, self-intersection, larger quality scans, and final calibrated thresholds still pending |
| M2Q Motion quality curation gate | Partial with enforceable train gate | `docs/research/motion_quality_curation.md`, existing source/G1 smoke scanners, source FK/contact smoke scanner, threshold-proposal scaffold, merged curated index smoke artifact, `runs/curated/smoke_source_g1_limit100/worst_clips.csv`, curated report breakdown by split/package/category, and `scripts/train.py` formal-run quality gate | Need G1 FK/contact metrics, full or representative scans beyond first 100, per-category calibrated thresholds, actor/skeleton diversity-loss report, and manual review loop for worst clips |
| M3 Dataset schema and observation contract | Implemented smoke path | `src/online_retarget/data/schema.py`, `src/online_retarget/data/windowed_builder.py`, CLI `build-windowed-jsonl`, `tests/test_schema.py`, `tests/test_windowed_builder.py`, architecture docs list observation dim 1547 and output dim 29, real artifact `runs/supervised/train_merged-quality-action_30b_h8_limit4/samples.jsonl` | Formal-scale extraction, normalization policy, robot-state wiring, and online preprocessing are pending |
| M4 Offline evaluation suite | Implemented scaffold | `src/online_retarget/evaluation.py`, CLI `offline-eval`, `tests/test_evaluation.py`, docs list output summary/per-sample/failure files | Simulator/contact metrics and real model predictions pending |
| M5 Direct supervised baseline | Partial | `scripts/train.py --dry-run` reads config, curated index, M2Q quality gate context, sample-manifest builder, schema, git state, DDP rank/world size, and sample refs; `scripts/train.py --samples-jsonl ...` has a PyTorch optimizer loop for supervised JSONL samples; formal non-dry-run training now refuses missing quality metadata and rejects `raw_bvh_channel_debug` samples unless `--allow-debug-data`; raw-channel and 30-body window JSONL builders both exist | Formal-scale 30-body dataset, WandB run, automatic M4 eval after training, checkpoint policy for real runs, and actual torch-environment training execution pending |
| M6 Model ablations and latency gate | Scaffold only | `scripts/benchmark_latency.py --dry-run` writes benchmark contract with observation/output dimensions | Real torch/CUDA/4090 latency measurement and ablation comparison pending |
| M7 Physics refinement and simulator eval | Scaffold only | `scripts/eval_isaac.py --dry-run` writes `isaac_eval_status.json` with intended metrics | Isaac Lab install, G1 replay/tracking task binding, and physics-refined target generation pending |

## Current Artifacts

- Split index: `runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv`
- Split report: `runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_report.json`
- G1 quality smoke report: `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit100/g1_quality_report.json`
- Source BVH quality smoke report: `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_limit100/source_bvh_quality_report.json`
- Source FK/contact quality smoke report: `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_fk_limit100/source_fk_quality_report.json`
- Threshold proposal smoke artifacts: `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit100/g1_threshold_proposals_p95.json`, `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_limit100/source_threshold_proposals_p95.json`
- Curated index smoke artifact: `runs/curated/smoke_source_g1_limit100/curated_index.csv`
- Curated report smoke artifact: `runs/curated/smoke_source_g1_limit100/curated_report.json`
- Worst-clip manifest smoke artifact: `runs/curated/smoke_source_g1_limit100/worst_clips.csv`
- Supervised debug samples: `runs/supervised/train_h8_limit8/samples.jsonl`
- Curated-index supervised debug samples: `runs/supervised/train_merged-quality-action_h8_limit8/samples.jsonl`
- Curated-index 30-body window smoke samples: `runs/supervised/train_merged-quality-action_30b_h8_limit4/samples.jsonl`
- Latency dry-run: `runs/benchmarks/baseline_mlp_dry_run.json`
- Isaac dry-run status: `runs/isaac_eval/dry_run/isaac_eval_status.json`

## Stop Condition

Do not mark M1-M7 complete until:

- paper matrix has complete curation/model/eval extraction for the target paper set,
- M2 has calibrated clip-level quality flags beyond metadata, source BVH discontinuity stats, and G1 velocity/root stats,
- M5 has final 30-body tensor dataset and a verified real training run,
- M6 has a real target-hardware latency report,
- M7 has real Isaac Lab replay/tracking evaluation.
