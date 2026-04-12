# Notebooks

Recommended single-notebook workflow:

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/achyutmorang/latentdriver-waymax-experiments/blob/main/notebooks/latentdriver_colab_runner.ipynb) `latentdriver_colab_runner.ipynb`: one shell-only Colab terminal-style launcher. Mount Drive first from the Colab Files sidebar if `/content/drive/MyDrive` is not present. It downloads and runs `scripts/colab_bootstrap.py`, then calls `scripts/colab_canary.py --profile ...` so notebook cells do not own experiment logic.

Default profile:

```text
full-eval-dry-run
```

Profiles that depend on upstream LatentDriver automatically run `scripts/bootstrap_upstream.py` first in fresh Colab runtimes.

Common profiles:

```text
full-preprocess-status
full-eval-dry-run
full-eval-reactive-single
full-eval-reactive
full-eval-non-reactive
plot-full-reactive
visualize-smoke
```

Legacy task-specific notebooks remain available for reference, but the runner notebook should be the default path:

Debug handoff:

```bash
python3 scripts/pull_latest_debug.py --which latest_failure
```

The helper pulls `debug_runs/LATEST_FAILURE.json` and `debug_runs/latest_failure/` from the Drive-backed project folder via the configured `gdrive_ro` rclone remote.

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/achyutmorang/latentdriver-waymax-experiments/blob/main/notebooks/latentdriver_assets_colab.ipynb) `latentdriver_assets_colab.ipynb`: clone/sync repo, mount Drive, patch upstream, download public evaluation checkpoints.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/achyutmorang/latentdriver-waymax-experiments/blob/main/notebooks/latentdriver_preprocess_val_colab.ipynb) `latentdriver_preprocess_val_colab.ipynb`: prepare smoke or full validation preprocessing artifacts from WOMD validation data, using either a Drive-local dataset root or authenticated GCS access.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/achyutmorang/latentdriver-waymax-experiments/blob/main/notebooks/latentdriver_full_eval_colab.ipynb) `latentdriver_full_eval_colab.ipynb`: older full-eval dry-run wrapper, retained for reference.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/achyutmorang/latentdriver-waymax-experiments/blob/main/notebooks/latentdriver_public_eval_colab.ipynb) `latentdriver_public_eval_colab.ipynb`: evaluate public checkpoints under standardized reactive/non-reactive Waymax tiers.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/achyutmorang/latentdriver-waymax-experiments/blob/main/notebooks/latentdriver_visualize_colab.ipynb) `latentdriver_visualize_colab.ipynb`: run one smoke visualization job and surface the generated MP4/PDF artifact.
