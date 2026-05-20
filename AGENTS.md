# OnlineRetarget Agent Contract

This repository targets a learning-based online retargeter from heterogeneous human skeleton motion to Unitree G1 robot motion.

## Non-Negotiables

- Treat `/home/user/data/motion_data` as read-only. Derived data, caches, checkpoints, and logs must go under `runs/`, `outputs/`, or another explicit repo-local/output path.
- Keep code simple before adding abstractions. Prefer direct supervised baselines and independent evaluation modules before generative or simulator-heavy variants.
- Cite papers or source code in docs when a design is derived from prior work.
- Keep training runs traceable: commit before meaningful training, log config and git SHA to WandB, and launch long training inside tmux.
- Before any remote training launch, verify the corresponding remote Git checkout is clean and at the latest upstream commit; launchers must refuse to run if this cannot be checked.
- Use Isaac Lab for simulation stages. This repo may contain adapters and configs, but simulator-specific assets should remain clearly separated.
- Preserve DDP readiness for training code: use rank/world-size aware logging, deterministic config capture, and output paths that do not collide across ranks.

## Current Technical Direction

- First output target: direct G1 generalized motion, beginning with joint position/velocity supervision from existing G1 targets.
- First model family: compact MLP/temporal MLP or small temporal transformer only after the MLP baseline is measured against the 1 ms inference budget on a 4090.
- First metrics: MPJPE/body position error, G1 joint RMSE, action similarity, joint jump rate, joint limit violation rate, and downstream tracking success once Isaac Lab integration exists.
- Latent/VAE, flow matching, and diffusion are research branches, not the baseline path, unless evaluation shows the compact direct model is insufficient.

## Web Retargeting / GMR / MuJoCo Lessons

- Validate the real end-to-end path before claiming success: upload input (`BVH` or `SMPL`) -> GMR retarget -> Unitree G1 qpos -> MuJoCo `Renderer` -> MP4. Do not rely on frontend point previews as proof of retarget or render correctness.
- Never use mock data or mock environments for user-facing retarget validation. Use the real local virtual environment, real GMR install, real G1 MJCF, real MuJoCo renderer, and the user's actual motion file.
- Preserve source frame count in the Web/API path. For BVH, process and render exactly the number of source frames by default; do not silently cap to 120 or any other debug limit.
- Preserve source timing as well as frame count. Read BVH `Frame Time` and encode the MuJoCo video at the corresponding FPS when feasible, so a 120 FPS motion does not become a slow 60 FPS video.
- Do not silently hide GMR failures behind a fallback. Reports must expose `selected_retargeter`, `src_human`, GMR status/details when relevant, `render_backend`, and whether the video was rendered by MuJoCo.
- Detect BVH source kind from skeleton joints, not just filename. For example, `LeftToeBase`/`RightToeBase` indicate the Nokov-style GMR BVH path even when the filename does not include `nokov`.
- For G1 ground/foot checks, use actual MuJoCo foot geoms and mesh vertices under the foot/ankle subtree. Do not infer foot contact from ankle or toe body centers; the G1 toe bodies may have no geoms.
- Keep retarget output and visualization correction separate. Ground alignment or root locking used for MuJoCo playback should not rewrite the raw GMR CSV unless that is explicitly requested.
- Avoid per-frame hard foot grounding for jogging/running motions. It can inject visible vertical jitter by pulling root Z up and down every frame. Prefer a sequence-level ground calibration for visualization, allowing natural flight phases.
- For kinematic playback of retargeted qpos in MuJoCo, set qpos and call `mj_forward` for validation/rendering. Do not add uncontrolled `mj_step` unless a real controller/physics rollout is part of the task.
- Quantify visual fixes with evidence: `load_frames`, `retarget_frames`, rendered frame count, `changed_frames`, `ffprobe` FPS/duration/frame count, foot geom min/max/mean height, and root-Z offset jump metrics.
- Keep debug-only controls out of the product path when they conflict with expected behavior. Internal test hooks may cap frames, but user-facing Web/API uploads should follow the source data by default.

## Documentation Discipline

- Update `docs/research/literature_review.md` when new papers materially change architecture or evaluation choices.
- Update `docs/research/data_inventory.md` when data layout, skeleton grouping, or loader assumptions change.
- Update `docs/milestones.md` when a milestone gate is completed or re-scoped.
