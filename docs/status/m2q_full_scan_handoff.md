# M2Q Full-Scan Handoff

Date: 2026-05-14.

Purpose: track long-running BONES-SEED `soma_proportional` / G1 quality scans without confusing progress checkpoints with a promotable curation policy.

2026-05-15 correction: this handoff is legacy archive evidence for `soma_proportional.tar + g1.tar`, not the active SONIC lane. The current SONIC data source is `/home/user/data/motion_data/bones_sonic`; use `docs/status/sonic_data_source.md` and `runs/indices/bones_sonic_index_full_v0` for current SONIC work. Do not use the G1 CSV review clips or quality counts below to answer SONIC data-quality questions.

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

The lane-level readiness artifact is:

- `runs/curated/m2q_full_scan_readiness_latest/readiness.json`

Latest readiness check is `blocked`: pair quality is ready at 142,220/142,220 rows; source BVH and G1 are partial and still missing final reports; source-FK full is missing.

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

Refresh lane readiness:

```bash
PYTHONPATH=src:. python3 scripts/inspect_bones_seed.py check-quality-readiness \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-json runs/curated/m2q_full_scan_readiness_latest/readiness.json \
  --source-stats-jsonl runs/m2q_source_bvh_full_20260514/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_full/source_bvh_quality_stats.jsonl \
  --source-report-json runs/m2q_source_bvh_full_20260514/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_full/source_bvh_quality_report.json \
  --source-fk-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_fk_full/source_fk_quality_stats.jsonl \
  --source-fk-report-json runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_fk_full/source_fk_quality_report.json \
  --g1-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_full/g1_quality_stats.jsonl \
  --g1-report-json runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_full/g1_quality_report.json \
  --pair-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_pair_full/pair_quality_stats.jsonl \
  --pair-report-json runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_pair_full/pair_quality_report.json
```

## Next Actions

1. Let `m2q_g1_full` and `m2q_source_bvh_full_20260514` finish.
2. When each final report appears, refresh summaries and record exact keep/downweight/quarantine/exclude counts.
3. Start full source-FK/contact scan only after at least one current full scan finishes.
4. Refresh `runs/curated/m2q_full_scan_readiness_latest/readiness.json`; merge full source BVH, source FK, G1, and pair stats into a new curated run only when all required lanes are `ready=true`.
5. Run threshold proposals, diversity-loss audit, manual review manifest, and policy preflight. Do not promote any policy until scan coverage, threshold acceptance, and manual review decisions pass.
