# M2Q Full-Scan Handoff

Date: 2026-05-14.

Purpose: track long-running BONES-SEED `soma_proportional` / G1 quality scans without confusing progress checkpoints with a promotable curation policy.

## Active Sessions

| Session | Started | Purpose | Output | Status |
| --- | --- | --- | --- | --- |
| `m2q_g1_full` | 2026-05-14 15:17 CST | Full G1 MJCF FK/contact/joint-limit scan over the actor-heldout split | `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_full/g1_quality_stats.jsonl` | Running; no final `g1_quality_report.json` yet |
| `m2q_source_bvh_full_20260514` | 2026-05-14 19:32 CST | Fresh full source BVH parser/discontinuity scan over the same split | `runs/m2q_source_bvh_full_20260514/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_full/source_bvh_quality_stats.jsonl` | Running; no final `source_bvh_quality_report.json` yet |

## Current Checkpoints

These are progress checkpoints only.

| Artifact | Rows summarized | Action counts | Top flags |
| --- | ---: | --- | --- |
| `runs/curated/g1_full_scan_progress_latest/progress_summary.json` | 73,291 | keep 2,516; downweight 25,043; quarantine 45,732 | `g1_foot_slide`, `g1_unstable_start_end`, `g1_joint_limit_violation`, `g1_ground_penetration`, `joint_velocity_jump` |
| `runs/curated/source_bvh_full_rescan_progress_latest/progress_summary.json` | 405 | keep 181; quarantine 220; exclude 4 | `source_channel_jump`, `nonfinite_value`, `channel_width_mismatch`, `source_root_discontinuity` |

The raw JSONL files were still growing after these summaries were written. Re-run `summarize-quality-jsonl` before using the numbers in any decision.

## Known Non-Final Artifacts

- `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_full/source_bvh_quality_stats.jsonl` contains 13,201 rows and no report. It is a partial artifact, not a completed full scan.
- `runs/curated/g1_full_scan_progress_latest/*` and `runs/review_clips/g1_full_scan_progress_58975_balanced_render/*` are progress review evidence, not a final curation policy.
- No full source-FK/contact scan has been launched yet. It should wait until either the G1 full scan or source BVH full rescan finishes, because both source-BVH and source-FK scans read the 276GB `soma_proportional.tar` and would otherwise compete heavily for IO.

## Repro Commands

Refresh G1 progress:

```bash
PYTHONPATH=src:. python3 scripts/inspect_bones_seed.py summarize-quality-jsonl \
  --stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_full/g1_quality_stats.jsonl \
  --output-json runs/curated/g1_full_scan_progress_latest/progress_summary.json
```

Refresh source BVH progress:

```bash
PYTHONPATH=src:. python3 scripts/inspect_bones_seed.py summarize-quality-jsonl \
  --stats-jsonl runs/m2q_source_bvh_full_20260514/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_full/source_bvh_quality_stats.jsonl \
  --output-json runs/curated/source_bvh_full_rescan_progress_latest/progress_summary.json \
  --metric max_abs_channel_velocity \
  --metric max_root_speed \
  --metric max_abs_channel_acceleration \
  --metric max_root_acceleration \
  --metric max_root_jerk \
  --metric channel_jump_rate \
  --metric root_jump_rate \
  --metric fps \
  --metric frame_count \
  --group-by split \
  --group-by category
```

Check active scanners:

```bash
tmux ls
ps -eo pid,etime,pcpu,pmem,stat,cmd | rg 'scan-g1-quality|scan-source-quality|inspect_bones_seed' | rg -v 'rg '
wc -l \
  runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_full/g1_quality_stats.jsonl \
  runs/m2q_source_bvh_full_20260514/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_full/source_bvh_quality_stats.jsonl
```

## Next Actions

1. Let `m2q_g1_full` and `m2q_source_bvh_full_20260514` finish.
2. When each final report appears, refresh summaries and record exact keep/downweight/quarantine/exclude counts.
3. Start full source-FK/contact scan only after at least one current full scan finishes.
4. Merge full source BVH, source FK, G1, and pair stats into a new curated run only when all required full lanes are available.
5. Run threshold proposals, diversity-loss audit, manual review manifest, and policy preflight. Do not promote any policy until scan coverage, threshold acceptance, and manual review decisions pass.
