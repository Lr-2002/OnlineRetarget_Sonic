# LR-272 Evaluator Frame Consistency

## Status

The root/front route is stopped after the negative mixed10 gate. Walk100 should
remain blocked until the evaluator frame contract below is proven on the active
campaign artifacts.

## FK Metric Definitions

`fk_world_*` compares representative G1 FK body points directly in the shared
world frame:

```text
error_world(body, frame) = || p_pred_world - p_ref_world ||
```

This metric is expected to change under a known rigid root translation or yaw,
even when the local joint pose is unchanged.

`fk_rootrel_*` compares the same representative body points after mapping each
motion into its own root-aligned frame:

```text
p_root = R_root^T * (p_world - root_world)
error_rootrel(body, frame) = || p_pred_root - p_ref_root ||
```

This metric must stay near zero for an identity/null transform and for a known
rigid root transform when the local joint pose is unchanged. A subtract-root-only
definition is not sufficient because a global root yaw would rotate body offsets
and can create meter-scale root-relative errors from a pure frame change.

Body representatives are the mean of the FK points emitted for each MJCF body by
`online_retarget.data.g1_quality.g1_fk_body_positions`.

## Required Proof

Use `scripts/lr272_g1_evaluator_consistency.py` before adding any new candidate.
The proof writes:

- `identity_null_metrics.csv`: official G1 compared to itself.
- `known_rigid_root_transform_metrics.csv`: official G1 with a fixed root
  translation/yaw compared to the original official G1.
- `per_body_mixed10_sanity_table.csv`: pelvis, waist, feet, and hands body-group
  errors for identity, rigid-transform, and optional candidate CSV directories.
- `frame_consistency_report.json`: pass/fail checks and definitions.

Pass conditions:

- identity FK world/rootrel, DoF, root, and root-rot errors are approximately
  zero;
- known rigid root transform has nonzero FK world error;
- known rigid root transform keeps FK rootrel and DoF errors approximately zero.

Only after those checks pass should any train-split-frozen DoF map be attempted.
