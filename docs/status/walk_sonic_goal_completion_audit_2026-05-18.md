# Walk SONIC Goal Completion Audit

Date: 2026-05-18.

Objective: "进行模型搭建和训练，以 walk task 为目标，直接用 BONES 里的 walk 数据作为数据，然后开始训练。"

## Concrete Deliverables

1. Identify a current BONES/SONIC walk subset.
2. Build train/eval samples from that walk subset.
3. Add a small model/training configuration.
4. Start and complete a real training smoke run.
5. Produce checkpoint, prediction, and evaluation artifacts.
6. Record limitations so the run is not mistaken for a formal retargeter or M2Q-gated policy.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Use BONES/SONIC data | Source index is `runs/indices/bones_sonic_index_full_v0/sonic_index.csv`; source data remains read-only under `/home/user/data/motion_data`. | Satisfied |
| Focus on walk task | Builder invoked with `--task-query walk`; train manifest records `candidate_clip_count=11571` walk clips, mirror clips excluded. | Satisfied |
| Build model/training path | Added `src/online_retarget/data/sonic_windowed_builder.py`, CLI `build-sonic-windowed-jsonl`, config `configs/walk_sonic_mlp_debug.yaml`, and reused `OnlineRetargetMLP`. | Satisfied |
| Generate training data | `runs/supervised/sonicbody_walk_train_h8_stride10_limit512/manifest.json` records 512 samples from 64 selected clips, input dim 1547, output dim 29. | Satisfied |
| Generate eval data | `runs/supervised/sonicbody_walk_val_h8_stride10_limit128/manifest.json` records 128 samples from 16 selected clips, actor-heldout split, input dim 1547, output dim 29. | Satisfied |
| Start training | `scripts/train.py` ran with `configs/walk_sonic_mlp_debug.yaml`, 20 CPU steps, batch size 64, `--allow-debug-data`. | Satisfied |
| Training completed | `runs/train/walk_sonic_mlp_debug_smoke/train_report.json` exists and records `final_train_mse=0.07438526302576065`; `checkpoint.pt` exists. | Satisfied |
| Eval completed | Train eval summary exists at `runs/train/walk_sonic_mlp_debug_smoke/eval/train_offline_eval/eval_summary.json`; val eval exists at `runs/eval/walk_sonic_mlp_debug_val/eval/offline_eval/eval_summary.json`. | Satisfied |
| Validate heldout actor split | Manifests record actor counts train/val/test = 240/30/30 and clip counts train/val/test = 9207/1242/1122. | Satisfied |
| Verification commands pass | `PYTHONPATH=src:. /home/user/repos/GR00T-WholeBodyControl/.venv_sim/bin/python -m unittest discover -s tests` ran 128 tests OK; `git diff --check` OK. | Satisfied |
| Limitations recorded | `docs/status/walk_sonic_training_smoke_2026-05-18.md` states this is `source_mode=sonic_body_pos`, a debug target-state baseline, not cross-skeleton retargeting and not M2Q-gated. | Satisfied |

## Completion Verdict

The immediate goal is achieved as a debug smoke: a walk-focused BONES/SONIC dataset was built, a compact MLP was trained, and train plus actor-heldout val evaluation artifacts were produced.

Not achieved, by design:

- formal M2Q-gated training;
- true cross-skeleton human-source retargeting;
- GPU/tmux long training;
- final benchmark-quality model.

The next goal should explicitly target `source_mode=soma_bvh` or a real SMPL/human-source lane if the desired deliverable is actual learned retargeting rather than a target-state debug baseline.
