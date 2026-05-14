# Completion Audit

Date: 2026-05-14.

Objective: "start implementing M1-M7; write live logs; make work traceable."

Verdict: not complete. The repo now has runnable scaffolds, real BONES-SEED smoke and representative M2Q artifacts, live logs, review-demo traceability surfaces, a stronger M1 deep-read set, and quality-gated training entry points, but formal M1-M7 completion is blocked by calibrated/promoted M2Q policy, final 30-body dataset extraction, real training, real 4090 latency, and Isaac Lab simulator evaluation.

## Prompt-to-Artifact Checklist

| Requirement | Evidence inspected | Status |
| --- | --- | --- |
| Implement M1 paper survey | `docs/research/paper_matrix.md`, `docs/research/literature_review.md`, `docs/research/bibliography.bib`, `docs/research/pdf_manifest.md`, `docs/research/papers/`, especially `docs/research/papers/motion_filtering_deep_read.md`, `docs/research/citation_usage_map.md`, and `docs/research/papers/tracking_latent_contact_deep_read.md` | Partial but stronger. Core matrix exists; filtering deep-read covers NMR, PHUMA, GMR, KDMR, ReActor, OmniRetarget, KungfuBot, RoboForge, DynaRetarget, and SPIDER with PDF/OpenAlex/project/code evidence; citation/usage map separates `cited`, `discussed`, and `used` relationships and records OpenAlex citation-graph limitations for fresh arXiv papers; the tracking/latent/contact note covers BeyondMimic, MaskedMimic, PHC, ProtoMotions, TMR, and Contact and Dynamics. This is enough to justify direct G1 output as the first baseline and to defer diffusion/partial-observation controllers until after M2Q/M5/M6 evidence. Final ablation selection still needs real training/latency results. |
| Include data filtering research | `docs/research/motion_quality_curation.md`, M2Q in `docs/milestones.md`, `docs/research/papers/motion_filtering_deep_read.md`, `docs/research/papers/tracking_latent_contact_deep_read.md` | Partial. Paper-backed quality plan exists; source/G1 FK-contact/support/root-height/contact-correction-candidate/self-collision-proxy metrics, pair/provenance checks, upper/lower-tail grouped threshold proposals, accepted threshold-policy artifacts, stratified scan support, diversity-loss reports, manual review manifests, reviewer CSV templates, review-decision ingest, and a one-command `quality-smoke` regression runner now exist. Deep-read adds jerk, support-base, pelvis/root-height, contact-mask correction, simulator-refined provenance, and contact-skate displacement as required signals. Acceleration/jerk, support-base/pelvis-root height, contact-skate displacement, and contact-mask correction candidates are implemented as metric-only scanner output; actual correction, simulator labels, and policy promotion remain pending. |
| Implement M2 data inventory/split/curation | `src/online_retarget/data/curation.py`, CLI `split-index`, real index under `runs/indices/...`, tests | Partial. Actor-heldout split and metadata curation are implemented; full quality gate remains incomplete. |
| Keep source data read-only | `.gitignore`, commands write under `runs/`, data root `/home/user/data/motion_data` only read | Satisfied for current work. |
| Separate different actors/skeletons | real split report: 522 actors, train/val/test actor split 417/52/53 | Satisfied at metadata split level. |
| Implement M2Q quality filtering | source/G1 scanners, source FK/contact/support/root-height/contact-correction-candidate scanner, G1 MJCF FK/contact/support/root-height/contact-correction-candidate/self-collision-proxy scanner, pair/provenance scanner, grouped upper/lower-tail threshold proposal, accepted threshold-policy artifact, stratified scan support, `merge-quality`, `worst_clips.csv`, manual review manifest/template, `merge-review-decisions`, policy-promotion audit/preflight, one-command `quality-smoke`, train quality gate, `docs/status/m2q_full_scan_handoff.md` | Partial. Smoke and representative quality pipeline exists; source BVH, source FK/contact/support/root-height/contact-correction-candidate, G1 FK/contact/support/root-height/contact-correction-candidate/self-collision-proxy, and pair/provenance stats now merge into curated indexes, curated reports include diversity-loss summaries, and worst clips are exportable to JSONL/Markdown review manifests plus reviewer-fillable CSV templates. Reviewer decisions can now be merged from CSV/JSONL into a new reviewed manifest plus `review_decision_report.json`, with validation for known IDs and legal actions. Threshold proposals can now be wrapped into a named `threshold_policy.json` with acceptance rationale and git evidence; `accept-curation-threshold-policy` discovers the matching proposal files from a curated run and refuses to overwrite an existing policy without `--overwrite`. Audit/preflight can consume this artifact instead of relying on a boolean CLI promise. The one-command smoke artifact `runs/curated/quality_smoke_limit24_by_category_split` scanned 24 category/split rows per source/G1/pair lane, generated grouped threshold proposals and 12 review items, and is intentionally `promotable=false` because threshold proposals are not accepted as a policy. The refreshed 560-row representative four-way merge at commit `e9054a4` records keep/downweight/quarantine/exclude = 70,858/70,878/477/7 after merging 560 source, 560 source-FK, 560 G1, and 560 pair rows; `worst_clips.csv` includes source/G1 root-height, support-distance, and contact-correction candidate columns. The representative scans record 6 source-FK and 172 G1 contact-correction candidates; 42 of the 100 worst rows carry a candidate, and the review manifest now includes a `contact_correction` family. `preflight-curation-policy` generated `policy_preflight.json` with `promotable=false`, auto-discovered five matching threshold proposal files, and blocked formal promotion on partial scan coverage, missing accepted threshold policy, and missing real manual-review decisions. BONES-SEED pair timing is recorded as 120 Hz with exact source/G1 frame-count agreement in the representative sample. Full pair scan is complete, but full G1 (`m2q_g1_full`) and fresh source BVH (`m2q_source_bvh_full_20260514`) scans are still running; the older 13,201-row source-full JSONL is explicitly treated as partial. Completed full source-FK/contact scan, real review decisions, formal category thresholds, actual correction, and simulator-backed labels remain pending. |
| Implement M3 schema/obs contract | `src/online_retarget/data/schema.py`, `src/online_retarget/data/windowed_builder.py`, `tests/test_schema.py`, `tests/test_windowed_builder.py`, real 30-body smoke artifact | Smoke path implemented. Formal-scale extraction, normalization policy, robot-state wiring, and online preprocessing are pending. |
| Implement M4 independent eval | `src/online_retarget/metrics.py`, `src/online_retarget/evaluation.py`, CLI `offline-eval`, `tests/test_metrics.py`, `tests/test_evaluation.py` | Scaffold implemented and extended with target-contact-aware FK artifact metrics for predicted foot float, foot slide speed, contact-skate displacement, ground penetration, clearance, and contact-frame ratio when JSONL rows include body positions plus foot body metadata. Real model predictions, simulator collision labels, and Isaac rollout metrics are pending. |
| Implement M5 supervised baseline | `scripts/train.py`, `src/online_retarget/data/supervised_builder.py`, `src/online_retarget/data/windowed_builder.py`, supervised JSONL artifacts | Partial. PyTorch optimizer loop exists and post-train prediction JSONL/offline-eval/WandB metadata hooks are coded, but current Python lacks torch, no policy audit is promotable yet, and formal-scale 30-body training is pending. |
| Enforce quality before formal training | `scripts/train.py` quality gate, policy audit gate, sample-builder gate, `tests/test_train_entry.py`, dry-run output, raw-debug negative check, blocked-audit non-dry-run check | Implemented for current training entry. Formal non-dry-run refuses missing quality metadata, missing/unpromotable policy audits, and raw debug sample artifacts. |
| DDP support | `scripts/train.py` reads `RANK`/`WORLD_SIZE` and reports them | Minimal scaffold only. Real distributed training not verified. |
| WandB traceability | `docs/experiment_tracking.md`, config project name, `scripts/train.py` optional WandB hooks, `tracking.wandb_mode` | Implemented as optional code path with default disabled mode. Real WandB artifact logging not executed in current no-torch environment. |
| Implement M6 latency gate | `scripts/benchmark_latency.py --dry-run` scaffold | Scaffold only. Real torch/CUDA/4090 benchmark pending. |
| Implement M7 Isaac Lab eval | `scripts/eval_isaac.py --dry-run` scaffold | Scaffold only. Real Isaac Lab/G1 replay task pending. |
| Implement web retarget preview | `scripts/run_web.py`, `src/online_retarget/web_app.py`, `src/online_retarget/web_pipeline.py`, `src/online_retarget/web_static/*`, `tests/test_web_pipeline.py`, `outputs/web_runs/1778729853-sample_upload-79da8236/pipeline_result.json`, `outputs/web_runs/1778730775-sample_smpl_like-3db7f515/pipeline_result.json` | Local standard-library web console implemented for BVH upload, approximate SMPL-like `.npz` preview from common pose/translation arrays, rule-based G1 preview retarget output, G1 MJCF kinematic preview, run artifacts, and explicit MuJoCo physics blocked/failed/ok status. A repo-local venv with `mujoco==3.8.1` completed real MuJoCo physics rollouts from both BVH and SMPL-like web flows against the G1 MJCF. Learned retarget checkpoints, full SMPL/SMPL-X body-model decoding, and controller-grade physical tracking remain pending. |
| Write live logs | `docs/logs/implementation-log.md` | Satisfied for current implementation history. Keep updating during future work. |
| Make process/status readable | `docs/milestones.md`, `docs/status/m1_m7_status.md`, this audit | Satisfied as a living tracking surface, not final completion. |
| Verify current work | Latest targeted evidence: `PYTHONPATH=src:. python3 -m unittest tests.test_quality_summary tests.test_quality_review_exports` -> 8 tests OK; `py_compile` passed for `quality_summary.py`, CLI, and its tests; `summarize-quality-jsonl --help` parsed; `summarize-quality-jsonl` wrote `runs/curated/g1_full_scan_progress_latest/progress_summary.json`; `export-balanced-quality-review` wrote a 16-row latest progress CSV; `export-review-clips` rendered 16/16 MuJoCo G1 MP4s under `runs/review_clips/g1_full_scan_progress_58975_balanced_render`. Earlier full-suite evidence: `PYTHONPATH=src:. python3 -m unittest discover -s tests` -> 106 tests OK, 1 skipped; `git diff --check` -> OK; targeted markdown/reference inspection found the new M1 deep-read note and source text artifacts; arXiv metadata check confirmed the Contact and Dynamics title/authors/arXiv ID; regenerated review-demo evidence remains available with 8 MuJoCo G1 videos, `render_status=ok`, `review_family`, and per-clip quality metrics in `README.md`, `summary.csv`, and `metadata.json`; full G1 scan is still running with no final `g1_quality_report.json` yet. Earlier verification also includes review-template tests, web-pipeline tests, policy-audit/preflight tests, blocked representative preflight, web MuJoCo smoke, dry-run training, source/G1/pair quality scans, representative 560-row scans, manual review manifests, blocked formal training on unpromotable policy audit, and raw-debug artifact rejection. | Current scaffold and review-demo traceability verified. Not evidence of full M1-M7 completion. |

## Latest Verification Evidence

```bash
PYTHONPATH=src:. python3 -m unittest discover -s tests
# Ran 106 tests in 0.135s, OK (skipped=1).

python3 -m py_compile \
  src/online_retarget/data/quality_review_exports.py \
  src/online_retarget/data/review_clips.py \
  tests/test_quality_review_exports.py \
  tests/test_review_clips.py
# OK

git diff --check
# OK

PYTHONPATH=src python3 scripts/inspect_bones_seed.py export-balanced-quality-review \
  --stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_full/g1_quality_stats.jsonl \
  --split-index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-csv runs/curated/g1_partial_quality_review/g1_partial_balanced_review.csv \
  --output-report-json runs/curated/g1_partial_quality_review/g1_partial_balanced_review_report.json \
  --flag g1_foot_slide \
  --flag g1_joint_limit_violation \
  --flag g1_unstable_start_end \
  --flag g1_ground_penetration \
  --flag joint_velocity_jump \
  --flag g1_foot_float \
  --flag g1_self_collision_proxy \
  --flag g1_low_foot_contact \
  --max-per-flag 1 \
  --include-downweight
# Refreshed a partial, non-promotable balanced review CSV with 8 rows.
# The CSV now includes max_start_end_root_speed and ranks
# g1_unstable_start_end by that metric.

PYTHONPATH=src outputs/web_mujoco_venv/bin/python scripts/inspect_bones_seed.py export-review-clips \
  --data-root /home/user/data/motion_data \
  --input-csv runs/curated/g1_partial_quality_review/g1_partial_balanced_review.csv \
  --output-root runs/review_clips \
  --run-name g1_partial_balanced_cli_render_check \
  --label g1_partial_balanced \
  --limit 8 \
  --render-g1 \
  --model-xml /home/user/repos/GMR/assets/unitree_g1/g1_mocap_29dof.xml \
  --render-max-frames 90 \
  --render-width 640 \
  --render-height 360 \
  --fps 120 \
  --root-position-scale 0.01 \
  --angle-scale 0.017453292519943295
# Refreshed runs/review_clips/g1_partial_balanced_cli_render_check with
# render_counts.ok=8. README.md, summary.csv, and per-clip metadata.json now carry
# review_family plus contact, penetration, joint-limit, start/end,
# velocity, float, and self-collision proxy metrics.

wc -l runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_full/g1_quality_stats.jsonl
# 37045 .../g1_quality_stats.jsonl
# tmux session m2q_g1_full still running; no final g1_quality_report.json yet.

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
# Ran 87 tests in final verification, OK.

PYTHONPATH=src:. python3 scripts/inspect_bones_seed.py merge-review-decisions \
  --review-manifest-jsonl runs/curated/representative_source_g1_pair_limit560_by_category_split/manual_review/review_manifest.jsonl \
  --decisions-file /tmp/ort-review-*/decisions.csv \
  --output-jsonl /tmp/ort-review-*/reviewed.jsonl \
  --output-report-json /tmp/ort-review-*/report.json
# Fixture evidence only: one valid decision merged for one representative review item,
# report recorded complete_decisions=1 and incomplete_decisions=41. No formal
# representative reviewed manifest was promoted or committed.

PYTHONPATH=src:. python3 scripts/inspect_bones_seed.py build-review-decision-template --help
# OK. The command writes a reviewer-fillable CSV from review_manifest.jsonl and
# refuses to overwrite existing template/report files unless --overwrite is used.

PYTHONPATH=src:. python3 scripts/inspect_bones_seed.py build-review-decision-template \
  --review-manifest-jsonl runs/curated/representative_source_g1_pair_limit560_by_category_split/manual_review/review_manifest.jsonl \
  --output-csv /tmp/ort-review-template-*/review_decision_template.csv \
  --output-report-json /tmp/ort-review-template-*/review_decision_template_report.json
# Fixture evidence only: wrote a 42-row reviewer CSV template under /tmp with
# review_id, failure family, source/G1 paths, flags, metric_summary, and blank
# decision fields. No formal review artifact was written under runs/.

PYTHONPATH=src:. python3 scripts/inspect_bones_seed.py accept-threshold-policy \
  --policy-id representative_source_g1_pair_limit560_by_category_split \
  --threshold-proposal-json runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_fk_limit560_by-category-split/source_fk_threshold_proposals_grouped_p95.json \
  --threshold-proposal-json runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit560_by-category-split/g1_threshold_proposals_grouped_p95.json \
  --output-json /tmp/ort-threshold-policy-*/threshold_policy.json \
  --accepted-by fixture \
  --rationale "Fixture only: validate threshold policy artifact schema; not a real accepted training policy." \
  --representative
# Fixture evidence only: wrote an accepted threshold-policy artifact summarizing
# 17 proposals across 1,120 proposal samples. It was not written under runs/.

PYTHONPATH=src:. python3 scripts/inspect_bones_seed.py preflight-curation-policy \
  --curated-run-dir runs/curated/representative_source_g1_pair_limit560_by_category_split \
  --output-json runs/curated/representative_source_g1_pair_limit560_by_category_split/policy_preflight.json
# Representative policy preflight is blocked as intended. It auto-discovered
# five threshold proposal files and reports next actions: full or accepted
# representative scans, accepted threshold policy, and 42 manual review decisions.

PYTHONPATH=src:. python3 -m py_compile \
  src/online_retarget/data/thresholds.py \
  src/online_retarget/cli.py \
  tests/test_thresholds.py
# OK

git diff --check
# OK

PYTHONPATH=src:. python3 scripts/train.py --config configs/baseline_mlp.yaml \
  --samples-jsonl runs/supervised/train_merged-quality-action_30b_h8_limit4/samples.jsonl \
  --max-steps 1
# Fails as intended before torch import because smoke_source_g1_limit100/policy_audit.json
# has promotable=false.

PYTHONPATH=src python3 -m py_compile scripts/train.py scripts/inspect_bones_seed.py src/online_retarget/data/quality_merge.py src/online_retarget/data/source_fk_quality.py src/online_retarget/data/windowed_builder.py src/online_retarget/data/__init__.py src/online_retarget/cli.py tests/test_quality_merge.py
# OK

git diff --check
# OK
```

## Known Blockers

- Current base Python lacks torch/numpy, so real training and latency benchmarking cannot run in this environment.
- Isaac Lab/G1 replay or tracking task binding is not implemented.
- M2Q quality scanning is beyond first-N smoke scale for representative 560-row category/split samples, but no formal policy is promoted. Source FK/contact/support/root-height, source/G1 acceleration-jerk metric outputs, and G1 MJCF FK/contact/support/root-height/self-collision-proxy metrics exist, along with upper/lower-tail grouped threshold proposals, accepted-threshold-policy artifacts, stratified scan support, diversity-loss summaries, manual review manifests, review-decision ingest, and promotion audits. Full scans or accepted representative policy, calibrated category thresholds, completed real review decisions, contact-mask correction decisions, and simulator-backed labels remain pending.
- M3/M5 now have a 30-body smoke sample builder, but formal-scale extraction, normalization, and robot-state wiring remain incomplete.
- WandB hooks, checkpoint report metadata, and automatic post-train offline eval are coded but not executed in a real torch training run yet.
