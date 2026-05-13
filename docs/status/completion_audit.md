# Completion Audit

Date: 2026-05-14.

Objective: "start implementing M1-M7; write live logs; make work traceable."

Verdict: not complete. The repo now has runnable scaffolds, real BONES-SEED smoke artifacts, live logs, and traceability surfaces, but formal M1-M7 completion is blocked by missing full paper deep-read, calibrated M2Q quality scans, final 30-body dataset extraction, real training, real 4090 latency, and Isaac Lab simulator evaluation.

## Prompt-to-Artifact Checklist

| Requirement | Evidence inspected | Status |
| --- | --- | --- |
| Implement M1 paper survey | `docs/research/paper_matrix.md`, `docs/research/literature_review.md`, `docs/research/bibliography.bib`, `docs/research/pdf_manifest.md`, `docs/research/papers/` | Partial. Core matrix exists, but deeper PDF reading and reference/citation expansion are still needed. |
| Include data filtering research | `docs/research/motion_quality_curation.md`, M2Q in `docs/milestones.md` | Partial. Paper-backed quality plan exists; source and G1 FK/contact smoke metrics now exist, but full calibration and simulator-backed validation are still pending. |
| Implement M2 data inventory/split/curation | `src/online_retarget/data/curation.py`, CLI `split-index`, real index under `runs/indices/...`, tests | Partial. Actor-heldout split and metadata curation are implemented; full quality gate remains incomplete. |
| Keep source data read-only | `.gitignore`, commands write under `runs/`, data root `/home/user/data/motion_data` only read | Satisfied for current work. |
| Separate different actors/skeletons | real split report: 522 actors, train/val/test actor split 417/52/53 | Satisfied at metadata split level. |
| Implement M2Q quality filtering | source/G1 scanners, source FK/contact scanner, G1 MJCF FK/contact scanner, threshold proposal, `merge-quality`, `worst_clips.csv`, train quality gate | Partial. Smoke quality pipeline exists; source BVH, source FK/contact, and G1 FK/contact stats now merge into the curated index, and curated reports include diversity-loss summaries. Full scans, self-collision, category thresholds, manual review, and simulator-backed labels remain pending. |
| Implement M3 schema/obs contract | `src/online_retarget/data/schema.py`, `src/online_retarget/data/windowed_builder.py`, `tests/test_schema.py`, `tests/test_windowed_builder.py`, real 30-body smoke artifact | Smoke path implemented. Formal-scale extraction, normalization policy, robot-state wiring, and online preprocessing are pending. |
| Implement M4 independent eval | `src/online_retarget/evaluation.py`, CLI `offline-eval`, `tests/test_evaluation.py` | Scaffold implemented. Real model predictions and simulator/contact metrics are pending. |
| Implement M5 supervised baseline | `scripts/train.py`, `src/online_retarget/data/supervised_builder.py`, `src/online_retarget/data/windowed_builder.py`, supervised JSONL artifacts | Partial. PyTorch optimizer loop exists and post-train prediction JSONL/offline-eval/WandB metadata hooks are coded, but current Python lacks torch and formal-scale 30-body training is pending. |
| Enforce quality before formal training | `scripts/train.py` quality gate, sample-builder gate, `tests/test_train_entry.py`, dry-run output, raw-debug negative check | Implemented for current training entry. Formal non-dry-run refuses missing quality metadata and raw debug sample artifacts. |
| DDP support | `scripts/train.py` reads `RANK`/`WORLD_SIZE` and reports them | Minimal scaffold only. Real distributed training not verified. |
| WandB traceability | `docs/experiment_tracking.md`, config project name, `scripts/train.py` optional WandB hooks, `tracking.wandb_mode` | Implemented as optional code path with default disabled mode. Real WandB artifact logging not executed in current no-torch environment. |
| Implement M6 latency gate | `scripts/benchmark_latency.py --dry-run` scaffold | Scaffold only. Real torch/CUDA/4090 benchmark pending. |
| Implement M7 Isaac Lab eval | `scripts/eval_isaac.py --dry-run` scaffold | Scaffold only. Real Isaac Lab/G1 replay task pending. |
| Write live logs | `docs/logs/implementation-log.md` | Satisfied for current implementation history. Keep updating during future work. |
| Make process/status readable | `docs/milestones.md`, `docs/status/m1_m7_status.md`, this audit | Satisfied as a living tracking surface, not final completion. |
| Verify current work | `PYTHONPATH=src:. python3 -m unittest discover -s tests` -> 44 tests OK; targeted `py_compile` -> OK; dry-run training -> OK with `samples_builder_is_formal=true`; source FK/contact and G1 MJCF FK/contact smoke scans -> OK; three-way `merge-quality` refreshed curated smoke artifact; raw-debug artifact formal training check fails as intended | Current scaffold verified. Not evidence of full M1-M7 completion. |

## Latest Verification Evidence

```bash
PYTHONPATH=src:. python3 -m unittest discover -s tests
# Ran 44 tests in 0.042s, OK.

PYTHONPATH=src python3 scripts/inspect_bones_seed.py scan-source-fk-quality \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --limit 100 \
  --ground-height 0.0 \
  --fps 30 \
  --frame-stride 2 \
  --max-frames 256
# Source FK/contact smoke wrote source_fk_quality_report.json with
# keep/downweight/quarantine = 42/20/38.

PYTHONPATH=src python3 scripts/inspect_bones_seed.py scan-g1-quality \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --limit 100 \
  --model-xml /home/user/repos/GMR/assets/unitree_g1/g1_mocap_29dof.xml \
  --frame-stride 2 \
  --max-frames 256
# G1 MJCF FK/contact smoke wrote g1_quality_report.json with
# keep/downweight/quarantine = 19/36/45 and flags including
# g1_foot_slide=70, g1_ground_penetration=41, g1_joint_limit_violation=18.

PYTHONPATH=src python3 scripts/inspect_bones_seed.py merge-quality \
  --split-index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --source-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_limit100/source_bvh_quality_stats.jsonl \
  --source-fk-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_fk_limit100/source_fk_quality_stats.jsonl \
  --g1-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit100/g1_quality_stats.jsonl \
  --output-root runs \
  --run-name smoke_source_g1_limit100
# Three-way curated smoke wrote curated_report.json with
# keep/downweight/quarantine/exclude = 71088/71048/83/1 and
# merged_source_rows=100, merged_source_fk_rows=100, merged_g1_rows=100.
# diversity_loss shows 0 lost actor/source-skeleton groups in this smoke policy.

PYTHONPATH=src:. python3 scripts/train.py --config configs/baseline_mlp.yaml --dry-run --limit 1
# quality_gate shows policy_id=smoke_source_g1_limit100, quality_report_exists=true,
# uses_curated_index=true, uses_merged_action=true, samples_builder=bvh_fk_30body_window,
# samples_builder_is_formal=true, and train_refs=112768.

PYTHONPATH=src:. python3 scripts/train.py --config configs/baseline_mlp.yaml \
  --samples-jsonl runs/supervised/train_merged-quality-action_h8_limit8/samples.jsonl \
  --max-steps 1
# Fails as intended before torch import because raw_bvh_channel_debug is not a formal
# bvh_fk_30body_window sample artifact.

PYTHONPATH=src python3 -m py_compile scripts/train.py scripts/inspect_bones_seed.py src/online_retarget/data/quality_merge.py src/online_retarget/data/source_fk_quality.py src/online_retarget/data/windowed_builder.py src/online_retarget/data/__init__.py src/online_retarget/cli.py tests/test_quality_merge.py
# OK

git diff --check
# OK
```

## Known Blockers

- Current base Python lacks torch/numpy, so real training and latency benchmarking cannot run in this environment.
- Isaac Lab/G1 replay or tracking task binding is not implemented.
- M2Q quality scanning is still smoke-scale. Source FK/contact and G1 MJCF FK/contact metrics exist; self-collision, full scans, calibrated category thresholds, and simulator-backed labels remain pending.
- M3/M5 now have a 30-body smoke sample builder, but formal-scale extraction, normalization, and robot-state wiring remain incomplete.
- WandB hooks, checkpoint report metadata, and automatic post-train offline eval are coded but not executed in a real torch training run yet.
