# WaymaxBoard

WaymaxBoard is a NuBoard-inspired browser for this repo's **Waymax evaluation artifacts**.

## Why it exists

NuBoard's strongest transferable idea is not the nuPlan-specific replay engine. It is the **file-backed experiment browser**:

- discover experiment bundles from disk
- summarize metric outputs across runs
- compare distributions across planners
- inspect a selected run's media and logs

That fits this repo directly because the local evaluation contract already writes:

- `run_manifest.json`
- `metrics.json`
- `config_snapshot.json`
- `stdout.log`
- `stderr.log`
- `vis/` media artifacts
- `suite_summary.json`

## What was borrowed from NuBoard

From `motional/nuplan-devkit` NuBoard:

- tabbed browser structure
- experiment-file discovery from disk
- separation between overview, metric distribution, and scenario/artifact inspection
- Bokeh server architecture with static file serving

## What was deliberately not copied

WaymaxBoard does **not** try to port:

- nuPlan's scenario builder integration
- simulation log deserialization
- map-layer and agent-layer playback from nuPlan objects
- cloud storage UI

Those pieces are tightly coupled to nuPlan's simulation serialization format. This repo does not currently emit Waymax runs in an equivalent replay format.

## Current tabs

### Overview

- completed run table
- suite summary table
- filters by model and tier

### Metrics

- run-level metric bar view
- simple histogram over the selected metric
- metrics supported now:
  - `AR[75:95]`
  - `mAR[75:95]`
  - `collision_rate`
  - `offroad_rate`
  - `progress_rate`

### Artifacts

- select a completed run
- inspect manifest JSON
- inspect stdout/stderr tails
- embed MP4/PDF/image artifacts from `vis/`

## Launch

```bash
python3 scripts/run_waymax_board.py --results-root "$PWD/results/runs" --port 5007
```

## Design boundary

WaymaxBoard is a **viewer over the existing experiment contract**. It is not yet a simulator replay tool.

That distinction matters:

- if you want final-result browsing, the current implementation is enough
- if you want frame-accurate multi-agent replay with per-step overlays, the repo must first start serializing richer Waymax rollout state

## Natural next steps

1. Persist per-scenario and per-batch metrics, not only run-level summaries.
2. Add side-by-side run comparison in the artifacts tab.
3. Add candidate-trajectory overlays once candidate dumping exists.
4. Add timeline-aware replay only after the rollout serialization contract is explicit.
