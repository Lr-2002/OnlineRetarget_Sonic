# Completion Audit

Date: 2026-05-14.

Objective: "start implementing M1-M7; write live logs; make work traceable."

Verdict: not complete. The repo now has runnable scaffolds, real BONES-SEED smoke artifacts, live logs, and traceability surfaces, but formal M1-M7 completion is blocked by missing full paper deep-read, calibrated M2Q quality scans, final 30-body dataset extraction, real training, real 4090 latency, and Isaac Lab simulator evaluation.

## Prompt-to-Artifact Checklist

| Requirement | Evidence inspected | Status |
| --- | --- | --- |
| Implement M1 paper survey | `docs/research/paper_matrix.md`, `docs/research/literature_review.md`, `docs/research/bibliography.bib`, `docs/research/pdf_manifest.md`, `docs/research/papers/` | Partial. Core matrix exists, but deeper PDF reading and reference/citation expansion are still needed. |
| Include data filtering research | `docs/research/motion_quality_curation.md`, M2Q in `docs/milestones.md` | Partial. Paper-backed quality plan exists; full implementation still lacks FK/contact metrics and calibrated thresholds. |
| Implement M2 data inventory/split/curation | `src/online_retarget/data/curation.py`, CLI `split-index`, real index under `runs/indices/...`, tests | Partial. Actor-heldout split and metadata curation are implemented; full quality gate remains incomplete. |
| Keep source data read-only | `.gitignore`, commands write under `runs/`, data root `/home/user/data/motion_data` only read | Satisfied for current work. |
| Separate different actors/skeletons | real split report: 522 actors, train/val/test actor split 417/52/53 | Satisfied at metadata split level. |
| Implement M2Q quality filtering | source/G1 scanners, threshold proposal, `merge-quality`, `worst_clips.csv`, train quality gate | Partial. Smoke quality pipeline exists; full scans, FK/contact, penetration/self-collision, and category thresholds remain pending. |
| Implement M3 schema/obs contract | `src/online_retarget/data/schema.py`, `tests/test_schema.py` | Scaffold implemented. Actual final 30-body tensor extraction is pending. |
| Implement M4 independent eval | `src/online_retarget/evaluation.py`, CLI `offline-eval`, `tests/test_evaluation.py` | Scaffold implemented. Real model predictions and simulator/contact metrics are pending. |
| Implement M5 supervised baseline | `scripts/train.py`, `src/online_retarget/data/supervised_builder.py`, supervised JSONL artifacts | Partial. PyTorch optimizer loop exists, but current Python lacks torch and final 30-body dataset/WandB/auto-eval are pending. |
| Enforce quality before formal training | `scripts/train.py` quality gate, `tests/test_train_entry.py`, dry-run output | Implemented for current training entry. Formal non-dry-run refuses missing quality policy metadata. |
| DDP support | `scripts/train.py` reads `RANK`/`WORLD_SIZE` and reports them | Minimal scaffold only. Real distributed training not verified. |
| WandB traceability | `docs/experiment_tracking.md`, config project name | Not implemented in training code yet. |
| Implement M6 latency gate | `scripts/benchmark_latency.py --dry-run` scaffold | Scaffold only. Real torch/CUDA/4090 benchmark pending. |
| Implement M7 Isaac Lab eval | `scripts/eval_isaac.py --dry-run` scaffold | Scaffold only. Real Isaac Lab/G1 replay task pending. |
| Write live logs | `docs/logs/implementation-log.md` | Satisfied for current implementation history. Keep updating during future work. |
| Make process/status readable | `docs/milestones.md`, `docs/status/m1_m7_status.md`, this audit | Satisfied as a living tracking surface, not final completion. |
| Verify current work | `PYTHONPATH=src:. python3 -m unittest discover -s tests` -> 28 tests OK; `py_compile` -> OK; `git diff --check` -> OK; dry-run training -> OK | Current scaffold verified. Not evidence of full M1-M7 completion. |

## Latest Verification Evidence

```bash
PYTHONPATH=src:. python3 -m unittest discover -s tests
# Ran 28 tests in 0.025s, OK

PYTHONPATH=src:. python3 scripts/train.py --config configs/baseline_mlp.yaml --dry-run --limit 1
# quality_gate shows policy_id=smoke_source_g1_limit100, quality_report_exists=true,
# uses_curated_index=true, uses_merged_action=true

PYTHONPATH=src python3 -m py_compile src/online_retarget/data/quality_merge.py scripts/train.py scripts/inspect_bones_seed.py src/online_retarget/cli.py
# OK

git diff --check
# OK
```

## Known Blockers

- Current base Python lacks torch/numpy, so real training and latency benchmarking cannot run in this environment.
- Isaac Lab/G1 replay or tracking task binding is not implemented.
- M2Q quality scanning is still smoke-scale for source/G1 stats and lacks FK/contact/penetration/self-collision signals.
- Final M3/M5 training samples still use raw BVH channels for data-path debugging, not the final 30-body observation contract.
- WandB logging, checkpoint artifact registration, and automatic post-train offline eval are pending.
