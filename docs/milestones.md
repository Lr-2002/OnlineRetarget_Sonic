# Milestones

This project is evaluation-first: no model milestone is complete unless its data quality assumptions, split policy, metrics, and reproducibility evidence are explicit.

## M0 - Repository Initialization

Purpose: create a traceable baseline workspace.

Gate:

- Git repository initialized.
- Read-only data rule documented for `/home/user/data/motion_data`.
- Initial docs cover literature, data, architecture, milestones, and experiment tracking.
- Pure-Python smoke tests pass.

Status: mostly complete in baseline commit `aa84901`; keep updating docs as assumptions change.

## M1 - Paper Survey and Design Matrix

Purpose: understand existing learning-based, physics-aware, and kinematic retargeting methods before committing to model/input/output design.

Required reading coverage:

- User-provided references: PDF-HR, NMR, and related/cited works.
- Learning-based humanoid retargeting/tracking: BeyondMimic, GMR/Retargeting Matters, OmniTrack, ReActor, OmniRetarget, ProtoMotions-G1, ULTRA, shared-latent retargeting.
- For each paper/codebase, extract: observation, reward, loss, data, model, output, evaluation metrics, failure filtering, and implications for OnlineRetarget.

Gate:

- `docs/research/paper_matrix.md` contains a comparison table with obs/reward/loss/data/model/output/eval.
- `docs/research/bibliography.bib` and `docs/research/pdf_manifest.md` identify papers, PDFs, and local reading status.
- Per-paper notes exist for the first-pass core references.
- Open questions are separated from confirmed claims.
- Model choices in later milestones cite this matrix rather than unsupported intuition.

Stop condition: enough evidence exists to choose the first baseline and the first ablations without re-reading the same papers.

## M2 - Data Inventory, Motion Curation, and Splits

Purpose: build a reliable supervised/eval dataset without assuming the source motions or retargeted targets are perfect.

Core assumption: motion data can contain abnormal human motion, bad retargeting, simulator-invalid targets, skeleton-specific artifacts, and label noise. Split and training must happen after quality bookkeeping, not before.

Quality dimensions to track:

- Human motion quality: joint jumps, body twists, impossible velocities/accelerations, sit/contact ambiguity, float, ground penetration, self-penetration, missing or unstable frames.
- Humanoid motion quality: G1 joint jumps, joint-limit violations, foot float, foot slide, ground penetration, self-collision proxies, root discontinuities, unstable start/end frames.
- Pair quality: human-to-G1 temporal length mismatch, frame-rate mismatch, action/category mismatch, actor/skeleton metadata mismatch, mirrored-pair leakage.
- Physics quality: whether the motion is only kinematic, simulator-replayed, physics-refined, or policy-generated.

Gate:

- Loader can read BONES-SEED metadata and group by `actor_uid` / source skeleton.
- Actor-heldout train/val/test split is generated reproducibly.
- Mirrored clips and same-actor variants cannot leak across splits.
- Derived split/index files are written outside `/home/user/data/motion_data`.
- Data report includes actor/skeleton distribution, package/category distribution, missing-file checks, and quality-flag summary.
- Curation policy is explicit: keep, downweight, quarantine, or exclude per quality flag.

Stop condition: we can say exactly which clips are used for train/test/eval and why.

Current status: metadata-level actor split and curation index are implemented. Source BVH, source FK/contact, and G1 target quality scanners exist. G1 target scanning can optionally load a MJCF asset to compute FK contact, foot slide/float, ground penetration, and joint-limit smoke metrics. Full calibrated thresholds, larger scans, diversity-loss review, and self-collision/simulator-backed labels are tracked by M2Q, so M2 is not fully closed.

## M2Q - Motion Quality Curation Gate

Purpose: make "good enough motion" a measurable dataset property before the model is allowed to train on it.

Core assumption: neither the source human skeleton motion nor the provided G1 target motion should be treated as ground truth without inspection. A clip can be visually plausible but still harmful because of temporal jumps, wrong contact, floating feet, ground penetration, self-intersection, joint-limit stress, or simulator-infeasible dynamics.

Required reading coverage:

- NMR / CEPR: physics-aware human curation, humanoid motion curation, and RL physics refinement.
- PHUMA / PhySINK: root jerk, support-base checks, foot-contact score, pelvis-height filtering, joint feasibility, grounding, and skating losses.
- GMR / Retargeting Matters: retarget artifacts that degrade downstream BeyondMimic tracking, including foot sliding, penetration, self-intersections, floating, and scale-induced errors.
- OmniTrack, OmniRetarget, and KDMR: physics-consistent references, hard kinematic/contact constraints, and multi-contact dynamic feasibility.
- Contact/dynamics and foot-contact literature: foot contact detection, foot float, penetration, and skating definitions.

Quality signals:

- Source human motion: nonfinite frames, frame-count/channel mismatch, root velocity, root acceleration/jerk, joint/channel velocity spikes, twist spikes, contact plausibility, float/penetration when foot bodies are available.
- G1 target motion: joint position/velocity/acceleration spikes, joint-limit margin, root discontinuity, FK body-height artifacts, foot float, foot slide, ground penetration, self-collision proxy, unstable start/end frames.
- Pair quality: source-target length/fps mismatch, action/category mismatch, mirrored-pair leakage, actor/skeleton leakage, missing G1 target, motion provenance mismatch.
- Physics quality: kinematic-only, filtered kinematic, simulator-replayed, RL-refined, or policy-generated target labels must not be mixed silently.

Gate:

- `docs/research/motion_quality_curation.md` maps each quality signal to paper evidence, implementation status, and the current OnlineRetarget action.
- Quality scanners produce per-clip JSONL/CSV stats outside `/home/user/data/motion_data`.
- Thresholds are calibrated from dataset distributions and reported by split/category/actor/skeleton, not hard-coded from one smoke run.
- Curation action is one of `keep`, `downweight`, `quarantine`, or `exclude`; binary keep/drop is only allowed for unrecoverable parse/provenance errors.
- The curation report shows retained hours/clips, quarantined/excluded reasons, and diversity loss by actor, skeleton, package, category, and motion type.
- At least one manually inspectable worst-clip manifest exists for each major failure type.
- Training code refuses a formal run unless the selected curation policy ID and quality report path are recorded in the config/run metadata.

Stop condition: we can defend why each formal training clip is included, downweighted, quarantined, or excluded, and we can quantify the quality-diversity tradeoff.

## M3 - Dataset Schema and Observation Contract

Purpose: freeze the first trainable input/output contract while keeping future physics/IMU variants compatible.

Baseline input:

- Source skeleton body positions with temporal history in a source-local heading frame.
- Source velocities over the same history window.
- Actor morphology vector from metadata.
- Optional robot current state fields: G1 joint position, joint velocity, previous action, IMU/root orientation, angular velocity.

Baseline output:

- Direct 29D G1 joint target or joint delta.

Later output branches:

- Root plus 29D generalized coordinates for offline reference generation.
- Latent output through VAE/decoder after direct-output metrics are stable.
- Physics-consistent rollout state from Isaac Lab or another simulator-backed refinement stage.

Gate:

- Schema dataclasses or typed records define sample IDs, source frames, target frames, morphology, robot state, quality flags, and provenance.
- Dataset loader can create fixed-window samples from metadata/index without modifying the data root.
- All generated artifacts record data root, metadata file, split ID, git SHA, config, and timestamp.
- Unit tests cover split-safe sample construction on fixture data.

Stop condition: training/eval code can consume the same sample contract without per-script ad hoc parsing.

Current status: `MotionPairRef`, `ObservationSpec`, `RobotStateSpec`, and `OutputSpec` are implemented and tested. `build-windowed-jsonl` now generates a smoke 30-body BVH-FK position/velocity window artifact matching the 1,547D observation contract. Formal-scale extraction, normalization policy, robot-state wiring, and online preprocessing are still pending.

## M4 - Offline Evaluation Suite

Purpose: evaluate retargeting quality before simulator rollout so model failures are easy to diagnose.

Required metrics:

- MPJPE / body position error when body positions are available.
- G1 joint RMSE and per-joint error.
- Action similarity or motion-feature similarity.
- Joint jump rate and acceleration/smoothness penalties.
- Joint-limit violation rate.
- Foot float, foot slide, ground penetration, and self-collision proxy metrics where signals exist.
- Per-actor, per-skeleton, per-height-bin, per-package/category, and quality-flag breakdowns.

Gate:

- Metrics run independent of training.
- Evaluation outputs JSON/CSV summaries and a failure manifest.
- Worst clips are exportable for review by actor/category/failure type.
- Metric definitions cite the paper matrix when borrowed from BeyondMimic/GMR/NMR/PDF-HR-style evaluation.

Stop condition: a model checkpoint can fail clearly, not vaguely.

Current status: independent JSONL offline evaluation is implemented with summary, per-sample metrics, failure manifest, and actor/category/package/quality-flag aggregation. Simulator/contact metrics remain pending.

## M5 - Direct Supervised Baseline

Purpose: establish the simplest measurable learning baseline before adding latent/diffusion/flow complexity.

Baseline model:

- Compact temporal MLP or small temporal encoder.
- Direct G1 joint target/delta output.
- Optional history and morphology inputs from M3.

Gate:

- Small-subset training runs end-to-end.
- WandB logs config, git SHA, data split ID, quality filter policy, metrics, and artifact paths.
- DDP-compatible launch works on one node, even if first runs use one GPU.
- Checkpoints include enough metadata to reproduce eval.
- Offline eval from M4 runs automatically after training.

Stop condition: baseline is worse/better by concrete metrics, not subjective visual inspection alone.

Current status: training dry-run validates config, git state, DDP rank/world size, curated index, M2Q quality gate context, observation/output dimensions, and sample refs. Raw-BVH-channel and 30-body BVH-FK supervised JSONL builders exist and have produced real smoke artifacts. `scripts/train.py` has a PyTorch optimizer loop for supervised JSONL artifacts and refuses formal training without quality policy metadata. Formal-scale 30-body dataset generation, WandB logging, automatic offline eval after training, and actual torch-environment training execution are pending.

## M6 - Model Design Ablations and Latency Gate

Purpose: compare design choices under the online constraint instead of optimizing only loss.

Candidate branches:

- Direct MLP / temporal MLP.
- Tiny transformer or temporal convolution.
- VAE latent motion-action alignment.
- Single-step distilled flow/diffusion variant only if inference can satisfy the budget.
- PDF-HR-style pose prior as a scorer or regularizer.

Gate:

- Each branch has the same split, same curation policy, and same offline eval.
- Inference benchmark reports p50/p95/p99 latency and batch-size assumptions.
- The target for the model forward pass is under 1 ms on RTX 4090.
- Branches that miss latency or quality gates are documented and stopped.

Stop condition: one candidate is selected for simulator evaluation with a defensible quality/latency tradeoff.

Current status: latency benchmark scaffold exists and dry-run records dimensions/config. Real 4090 timing requires torch/CUDA environment.

## M7 - Physics Refinement and Simulator Evaluation

Purpose: account for the fact that kinematically good retargeting may still be physically bad.

Pipeline choices:

- Physics-aware human motion curation before model training.
- Humanoid motion curation after kinematic retargeting.
- Physics-based humanoid motion refinement through Isaac Lab tracking/expert policy rollout.

Gate:

- Isaac Lab G1 asset, joint limits, and replay interface are verified.
- Predicted references can be replayed or tracked in simulation.
- Simulator metrics include success/fall rate, episode length, world/root-relative MPJPE, foot slide/float, penetration, self-collision, and robustness to observation noise/latency.
- Physics-refined targets are labeled separately from kinematic targets.
- The project can compare direct supervised outputs against physics-refined outputs without mixing provenance.

Stop condition: we know whether the neural retargeter needs to predict kinematic G1 references, physics-refined G1 references, or simulator-executed state/action targets.

Current status: Isaac Lab eval scaffold exists and writes a status artifact in dry-run mode. Real simulator replay/tracking remains blocked on Isaac Lab/G1 task integration.

## M8 - Online Retargeting Demo and Handoff

Purpose: prove the system can run as an online retargeter rather than only an offline benchmark.

Gate:

- Streamed or recorded skeleton input runs through preprocessing, model inference, postprocessing, and G1 reference output.
- End-to-end p95 model inference satisfies the 1 ms model budget or clearly reports where the budget is spent.
- Output drives G1 simulator without manual per-clip tuning.
- Failure cases are logged with input, output, quality flags, metrics, model checkpoint, git SHA, and config.
- Documentation explains how another engineer can reproduce data indexing, training, evaluation, and demo playback.

Stop condition: a new actor/skeleton can be evaluated online with traceable metrics and reproducible artifacts.
