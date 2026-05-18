# NMR Baseline Status - 2026-05-18

## Environment

- Official NMR repo: `/home/user/repos/MakeTrackingEasy`
- NMR venv: `/home/user/repos/MakeTrackingEasy/.venv`
- Python: `3.10.20`
- PyTorch: `2.12.0+cu130`
- GPU check: `torch.cuda.is_available() == True`, device `NVIDIA GeForce RTX 4090`
- Driver: `580.142`, Canonical-signed module under Secure Boot

The NVIDIA driver stack is held at Ubuntu `580.142` packages to avoid the CUDA repo `580.159` DKMS path, which was rejected by Secure Boot.

## Installed NMR Assets

- Checkpoint: `/home/user/repos/MakeTrackingEasy/weights/epoch_30.pth`
- SMPL-X neutral model: `/home/user/repos/MakeTrackingEasy/assets/SMPLX_NEUTRAL.npz`

HuggingFace main endpoint was unreachable from this machine. The checkpoint was downloaded through `https://hf-mirror.com/RayZhao/NMR/resolve/main/weights/epoch_30.pth`. The SMPL-X model was copied from the local model cache at `/home/user/models_smplx_v1_1/models/smplx/SMPLX_NEUTRAL.npz`.

## Official Sanity Inference

Command:

```bash
cd /home/user/repos/MakeTrackingEasy
.venv/bin/python inference.py \
  --src examples/sample_motion.npz \
  --output-dir /home/user/project/OnlineRetarget/runs/nmr/sanity_example
```

Result:

- Input: `examples/sample_motion.npz`
- Output: `/home/user/project/OnlineRetarget/runs/nmr/sanity_example/sample_motion.npz`
- Runtime log: `90 frames @ 30 FPS -> 149 frames @ 50 FPS`, total `0.21s`
- Output schema: `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`, `fps`

## BONES Data Compatibility Finding

Official NMR inference input is SMPL-X/AMASS-style npz with one of these key sets:

- `trans`, `root_orient`, `pose_body`
- `transl`, `global_orient`, `body_pose`

Full scan of `/home/user/data/motion_data` found:

- `168,129` npz files scanned
- `0` files matched the NMR human SMPL-X input schema
- `bones_sonic`: `142,220` npz files, all G1 target-like (`joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `fps`)
- `AMASS/GMR_retarget_data`: npz/pkl files are also G1 target-like retargeted outputs, not SMPL-X inputs

Important distinction: original AMASS does provide SMPL-family body parameters, but the local `AMASS/GMR_retarget_data` copy is not raw AMASS. It is already converted/retargeted data and should not be used as the SMPL-X source lane for NMR.

The BONES source skeleton data is present as BVH inside:

- `/home/user/data/motion_data/soma_uniform.tar`
- `/home/user/data/motion_data/soma_proportional.tar`

Metadata file `/home/user/data/motion_data/metadata/seed_metadata_v003.csv` maps each motion to:

- `move_soma_uniform_path`
- `move_soma_proportional_path`
- `move_g1_path`
- actor-specific SOMA shape params

Example source paths confirmed inside tar:

- `soma_proportional/bvh/230711/explosion_reaction_R_001__A425_M.bvh`
- `soma_proportional/bvh/230711/cellpone_take_out_stand_R_001__A424_M.bvh`
- `soma_proportional/bvh/240918/body_check_001__A548.bvh`

## Implication

NMR is configured and runnable on this machine, but "NMR on BONES" is not a direct command yet. `bones_sonic` is the G1 target lane, not the human input lane. To run official NMR on BONES source motions, the next required adapter is:

```text
BONES/SOMA BVH + actor SOMA shape metadata -> SMPL-X/AMASS-style npz -> NMR -> G1 bmimic npz
```

Without that adapter or an existing SMPL-X/AMASS source lane, any NMR output can only be an unpaired sanity inference, not a supervised BONES baseline score.

## Public BONES-SEED SMPL Search Note

Search on 2026-05-18 found that the official `bones-studio/seed` public release documents SOMA Uniform BVH, SOMA Proportional BVH, actor SOMA shape metadata, and Unitree G1 MuJoCo-compatible trajectories. It does not list SMPL/SMPL-X motion npz files.

A third-party Hugging Face dataset `zirobtc/bone` appears to contain a `bones-seed-smplx.tar.gz` WebDataset with visible `betas`-like fields. This is not the official BONES-SEED release, has no dataset card, and needs separate validation before it can be trusted for a baseline.
