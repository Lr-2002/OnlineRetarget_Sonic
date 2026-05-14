# Completion Audit

Date: 2026-05-14.

Objective: "start implementing M1-M7; write live logs; make work traceable."

Verdict: not complete. The repo now has runnable scaffolds, real BONES-SEED smoke and representative M2Q artifacts, live logs, and traceability surfaces, but formal M1-M7 completion is blocked by missing full paper deep-read, calibrated/promoted M2Q policy, final 30-body dataset extraction, real training, real 4090 latency, and Isaac Lab simulator evaluation.

## Prompt-to-Artifact Checklist

| Requirement | Evidence inspected | Status |
| --- | --- | --- |
| Implement M1 paper survey | `docs/research/paper_matrix.md`, `docs/research/literature_review.md`, `docs/research/bibliography.bib`, `docs/research/pdf_manifest.md`, `docs/research/papers/`, especially `docs/research/papers/motion_filtering_deep_read.md` | Partial. Core matrix exists and filtering deep-read now covers NMR, PHUMA, GMR, KDMR, ReActor, OmniRetarget, KungfuBot, RoboForge, DynaRetarget, and SPIDER with PDF/OpenAlex/project/code evidence. Reference/citation expansion is still needed. |
| Include data filtering research | `docs/research/motion_quality_curation.md`, M2Q in `docs/milestones.md`, `docs/research/papers/motion_filtering_deep_read.md` | Partial. Paper-backed quality plan exists; source/G1 FK-contact/support/root-height/contact-correction-candidate/self-collision-proxy metrics, pair/provenance checks, upper/lower-tail grouped threshold proposals, stratified scan support, diversity-loss reports, and manual review manifests now exist. Deep-read adds jerk, support-base, pelvis/root-height, contact-mask correction, and simulator-refined provenance as required signals. Acceleration/jerk, support-base/pelvis-root height, and contact-mask correction candidates are implemented as metric-only scanner output; actual correction, simulator labels, and policy promotion remain pending. |
| Implement M2 data inventory/split/curation | `src/online_retarget/data/curation.py`, CLI `split-index`, real index under `runs/indices/...`, tests | Partial. Actor-heldout split and metadata curation are implemented; full quality gate remains incomplete. |
| Keep source data read-only | `.gitignore`, commands write under `runs/`, data root `/home/user/data/motion_data` only read | Satisfied for current work. |
| Separate different actors/skeletons | real split report: 522 actors, train/val/test actor split 417/52/53 | Satisfied at metadata split level. |
| Implement M2Q quality filtering | source/G1 scanners, source FK/contact/support/root-height/contact-correction-candidate scanner, G1 MJCF FK/contact/support/root-height/contact-correction-candidate/self-collision-proxy scanner, pair/provenance scanner, grouped upper/lower-tail threshold proposal, stratified scan support, `merge-quality`, `worst_clips.csv`, manual review manifest, train quality gate | Partial. Smoke and representative quality pipeline exists; source BVH, source FK/contact/support/root-height/contact-correction-candidate, G1 FK/contact/support/root-height/contact-correction-candidate/self-collision-proxy, and pair/provenance stats now merge into curated indexes, curated reports include diversity-loss summaries, and worst clips are exportable to JSONL/Markdown review manifests. The refreshed 560-row representative four-way merge at commit `e9054a4` records keep/downweight/quarantine/exclude = 70,858/70,878/477/7 after merging 560 source, 560 source-FK, 560 G1, and 560 pair rows; `worst_clips.csv` includes source/G1 root-height, support-distance, and contact-correction candidate columns. The representative scans record 6 source-FK and 172 G1 contact-correction candidates; 42 of the 100 worst rows carry a candidate, and the review manifest now includes a `contact_correction` family. BONES-SEED pair timing is recorded as 120 Hz with exact source/G1 frame-count agreement in the representative sample. Completed review decisions, formal category thresholds, actual correction, and simulator-backed labels remain pending. |
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
| Verify current work | `PYTHONPATH=src:. python3 -m unittest discover -s tests` -> 67 tests OK; targeted `py_compile` -> OK; dry-run training -> OK with `samples_builder_is_formal=true`; `git diff --check` -> OK; source FK/contact and G1 MJCF FK/contact/self-collision-proxy smoke scans -> OK; representative 560-row scans and upper/lower-tail grouped threshold artifacts generated; four-way `merge-quality` refreshed curated representative artifacts with pair/provenance stats; manual review manifests generated; raw-debug artifact formal training check fails as intended | Current scaffold verified. Not evidence of full M1-M7 completion. |

## Latest Verification Evidence

```bash
PYTHONPATH=src:. python3 -m unittest discover -s tests
# Ran 54 tests in 0.054s, OK.

PYTHONPATH=src python3 scripts/inspect_bones_seed.py scan-source-fk-quality \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --limit 100 \
  --ground-height 0.0 \
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
# G1 MJCF FK/contact/self-collision-proxy smoke wrote g1_quality_report.json with
# keep/downweight/quarantine = 18/36/46 and flags including
# g1_foot_slide=70, g1_ground_penetration=41, g1_joint_limit_violation=18,
# g1_self_collision_proxy=1.

PYTHONPATH=src:. python3 scripts/inspect_bones_seed.py scan-pair-quality \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --limit 560 \
  --sample-by category \
  --sample-by split \
  --expected-source-frame-time 0.008333333333333333 \
  --g1-fps 120 \
  --max-frame-count-delta 0 \
  --max-duration-delta-sec 0.001 \
  --target-provenance kinematic_g1_csv
# Pair quality wrote pair_quality_report.json with keep/quarantine = 494/66,
# max frame-count delta 0, and p95 absolute duration delta 0.001768 s.

PYTHONPATH=src:. python3 scripts/inspect_bones_seed.py scan-source-fk-quality \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --limit 560 \
  --sample-by category \
  --sample-by split \
  --ground-height 0.0 \
  --frame-stride 2 \
  --max-frames 256
# Refreshed source FK/support report records keep/downweight/quarantine/exclude =
# 223/235/100/2 and support/root-height metric summaries.

PYTHONPATH=src:. python3 scripts/inspect_bones_seed.py scan-g1-quality \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --limit 560 \
  --sample-by category \
  --sample-by split \
  --model-xml /home/user/repos/GMR/assets/unitree_g1/g1_mocap_29dof.xml \
  --root-position-scale 0.01 \
  --joint-angle-scale 0.017453292519943295 \
  --root-rotation-scale 0.017453292519943295 \
  --frame-stride 2 \
  --max-frames 256
# Refreshed G1 FK/support report records keep/downweight/quarantine =
# 37/167/356 and support/root-height metric summaries.

PYTHONPATH=src:. python3 scripts/inspect_bones_seed.py merge-quality \
  --split-index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --source-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_limit560_by-category-split/source_bvh_quality_stats.jsonl \
  --source-fk-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_fk_limit560_by-category-split/source_fk_quality_stats.jsonl \
  --g1-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit560_by-category-split/g1_quality_stats.jsonl \
  --pair-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_pair_limit560_by-category-split/pair_quality_stats.jsonl \
  --output-root runs \
  --run-name representative_source_g1_pair_limit560_by_category_split
# Refreshed four-way representative curated report wrote keep/downweight/quarantine/exclude =
# 70858/70878/477/7 with merged_source_rows=560, merged_source_fk_rows=560,
# merged_g1_rows=560, merged_pair_rows=560, and 0 lost actor/source-skeleton/category/split groups.

PYTHONPATH=src python3 scripts/inspect_bones_seed.py build-review-manifest \
  --worst-clips-csv runs/curated/representative_source_g1_pair_limit560_by_category_split/worst_clips.csv \
  --output-root runs/curated/representative_source_g1_pair_limit560_by_category_split \
  --run-name manual_review \
  --max-per-family 5
# Manual review representative artifact wrote review_manifest.jsonl and
# review_manifest.md with 35 review items across parser, mirror, jump,
# foot-slide, penetration, float, and joint-limit families.

PYTHONPATH=src python3 scripts/inspect_bones_seed.py propose-thresholds \
  --stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit560_by-category-split/g1_quality_stats.jsonl \
  --output-json runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit560_by-category-split/g1_threshold_proposals_grouped_p95.json \
  --metric max_abs_joint_velocity \
  --metric joint_jump_rate \
  --metric max_root_speed \
  --metric joint_limit_violation_rate \
  --metric max_joint_limit_violation \
  --metric mean_foot_clearance \
  --metric penetration_depth \
  --metric contact_slide_rate \
  --metric max_contact_slide_speed \
  --metric self_collision_proxy_rate \
  --lower-metric contact_frame_ratio \
  --group-by category \
  --group-by split \
  --percentile 0.95
# Representative G1 threshold proposals record high-is-bad upper-tail metrics
# plus low-is-bad lower-tail contact_frame_ratio; sample_count=560.

PYTHONPATH=src:. python3 scripts/train.py --config configs/baseline_mlp.yaml --dry-run --limit 1
# quality_gate shows policy_id=smoke_source_g1_limit100, quality_report_exists=true,
# uses_curated_index=true, uses_merged_action=true, samples_builder=bvh_fk_30body_window,
# samples_builder_is_formal=true, and train_refs=112768.

PYTHONPATH=src:. python3 -m unittest discover -s tests
# Ran 67 tests in 0.072s, OK.

PYTHONPATH=src:. python3 -m py_compile \
  src/online_retarget/data/thresholds.py \
  src/online_retarget/cli.py \
  tests/test_thresholds.py
# OK

git diff --check
# OK

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
- M2Q quality scanning is beyond first-N smoke scale for representative 560-row category/split samples, but no formal policy is promoted. Source FK/contact/support/root-height, source/G1 acceleration-jerk metric outputs, and G1 MJCF FK/contact/support/root-height/self-collision-proxy metrics exist, along with upper/lower-tail grouped threshold proposals, stratified scan support, diversity-loss summaries, and manual review manifests. Full scans or accepted representative policy, calibrated category thresholds, completed review decisions, contact-mask correction candidates, and simulator-backed labels remain pending.
- M3/M5 now have a 30-body smoke sample builder, but formal-scale extraction, normalization, and robot-state wiring remain incomplete.
- WandB hooks, checkpoint report metadata, and automatic post-train offline eval are coded but not executed in a real torch training run yet.
