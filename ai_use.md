# AI Usage Disclosure

This file documents how AI tools were used in this project, in line with course and capstone expectations.

**Project:** Zero-Shot Learning Team (D05), forecasting-based electrical anomaly detection with synthetic data
**Last updated:** 2026-11-06

## Where AI Was Used

| Area | How AI was used |
|------|-----------------|
| Report: references | Helped organize references and draft BibTeX entries
| Report: introduction and abstract | Helped brainstorm and draft the introduction and abstract |
| Code | See below |

## Env initilization and repo structuring

Prompted with the project context, milestones, details of the sarsim0 paper. Conversed and decided to start with chornos-2. Wanted to establish baselines and settled on seasonal naive, a pre-trained chronos2.

Setup yaml configs and a cli to it’s easier to add more models, different tunings.

## Weights & Biases Addition

Prompted claude to understand what w&b is, and how we can implement it into our project. Worked on a separate branch, doing test pushes to W&B on smoke tests. Then imported our current results over to W&B using their library. It was here we also decided to split out readme into multiple markdown files in /docs so AI can better understand the project structure. This is also to mark project goals and prevent drift from docs and code. After renaming and organizing our training config yamls to a standard naming convention, we ran a sweep on all our different configs, which were synced and uploaded to w&b.

## Anomaly Detection
### 1. Anomaly-detection design and benchmark selection

### Important prompts

- “look through the repo and point out what needs to change to improve anomaly detection”
- “create a plan i can follow to implement the anomaly detection pipeline”
- “update the plan to focus on what we can improve and test before adding ucr”
- “rework the plan to include the code that needs to be written and how it should work, something i can give a coding agent”
- “check whether the ucr classification archive has the right labels for evaluating our anomaly detector”
- “read through the ucr/hexagon ml folder and explain how the files and labels work and how we would use them for evaluation”
- “look at where ucr integration should fit in the plan and what should be done first”

### Outputs

- A repository-grounded assessment of missing anomaly-detection functionality.
- An implementation plan for scoring, calibration, event grouping, metrics, and synthetic labels.
- A revised sequence prioritizing synthetic development before external evaluation.
- An explanation of the UCR classification archive versus the UCR/HexagonML anomaly archive.
- A plan for loading filename-encoded anomaly intervals.

### Validation

This stage produced design decisions and implementation requirements. Runtime testing occurred during the subsequent implementation.

### 2. Anomaly-pipeline implementation

### Important prompts

- “implement the attached anomaly detection improvement plan”
- “explain whats been done so far and what still needs to be done”
- “finish the remaining local work and run the checks”
- “look through the pipeline for anything else that needs to be added or improved, then give me the next steps after the push”

### Outputs

- Three anomaly-score functions.
- Rolling and block forecasting support.
- Forecast caching and cache validation.
- Custom labelled synthetic-anomaly injection.
- Clean-data threshold calibration.
- Event filtering, merging, and hysteresis.
- Point-level and event-level metrics.
- Development-only detector selection and locked evaluation.
- UCR loading and validation.
- CLI commands, configurations, saved results, diagnostics, and tests.

The implementation included `src/sarsim_fm/anomaly/`, `src/sarsim_fm/data/anomaly_injection.py`, and `src/sarsim_fm/data/ucr_anomaly.py`. Git history records the anomaly work in `9b47882`, later merged through PR #3.

### Testing

- The recorded test suite reached **54 passing tests**.
- `python -m pytest -q`, `ruff check src tests`, shell-syntax checks, and `git diff --check` passed.
- Tests covered score behavior, deterministic injection, forecast alignment, calibration isolation, event rules, metrics, UCR indexing, cache validity, writer locks, and locked selection.
- Both simple baselines, rolling MAD and Seasonal Naive, completed **all 250 UCR files**.
- These checks supported implementation correctness, rather than proving strong detection performance.

### 3. Local smoke tests and detector failure analysis

### Important prompts

- “give me a small local cpu smoke test for the anomaly pipeline”
- “run the same anomaly pipeline with pretrained chronos 2”
- “interpret these results and explain what was tested, what worked, and where the detector missed the anomalies”

### Outputs

- CPU smoke-test commands.
- A comparison of Seasonal Naive and pretrained Chronos-2 on the same synthetic subset.
- Diagnostic plots.
- Analysis of point flags versus retained events.
- Separate short CPU training checks.

### Testing and results

- The subset contained a spike series, a dip series, and a normal-only series.
- **Seasonal Naive:** 1 true event, 4 false events, and 1 missed event; event F1 0.2857; normal-point FPR 3.29%; runtime approximately 2.69 seconds.
- **Chronos-2:** 0 true events, 1 false event, and 2 missed events; event F1 0; normal-point FPR 1.91%; runtime approximately 20.71 seconds.
- Saved-score inspection identified isolated anomaly flags removed by the two-consecutive-flags requirement before gap merging.
- Two separate CPU `sfm check` invocations each completed two optimizer steps with finite reported losses.
- Those checks did not save checkpoints or continue one another, and did not explicitly establish that every gradient was finite.

### 4. UCR power-demand evaluation

### Important prompts

- “run the anomaly detector on power demand data from ucr”
- “give me the commands to run the evaluation locally on cpu”
- “explain what test was run, what settings it used, and what the results mean”
- “check whether this ran on one power demand series or the full subset”

### Outputs

- Commands for local pretrained Chronos-2 evaluation on UCR power-demand data.
- Interpretation of event metrics, false alarms, delay, and localization.
- Clarification of the evaluation’s actual scope.

### Testing and results

- One series was evaluated: PowerDemand1, ID 152.
- Runtime was approximately 82.2 seconds.
- Results: 1 true event, 20 false events, and no missed labelled event.
- Event precision was 4.76%; recall was 100% from one labelled event; event F1 was 9.09%.
- Normal-point FPR was 1.74%; detection delay was 333 readings.
- Top-one localization failed.
- No training or new anomaly injection occurred, and completion of the full 11-series Chronos evaluation was not recorded.

### 5. Understanding and reviewing the implementation

### Important prompts

- “explain how the current anomaly pipeline works start to finish, including the data, scores, thresholds, events, and evaluation”
- “walk through how the injection code changes a series and records where the anomaly is”
- “point out which parts come from sarsim0 and which parts were added by us”
- “explain what evidence we have that these changes represent real anomalies and what still needs to be checked”
- “explain why we need the scoring, threshold, and event steps instead of just looking at where the forecast is wrong”
- “explain how our detector defines an anomaly and how that differs from the labels we evaluate against”

### Outputs

- Explanations of the complete detector workflow.
- An explanation of the anomaly score and calibration threshold.
- A distinction between SarSim0 base generation and custom labelled injection.
- An explanation of anomaly definitions, event formation, and evaluation labels.
- Discussion of synthetic realism and the limitations of the available results.
- Repository-based meeting notes and technical summaries.


## NRP Configuration and Setup

Asked claude to investigate how the nrp platform worked, and how we could integrate our project to run on the platform. Looked into creating shell scripts that would clone our repo and allow us to run the same CLI commands on a kubernetes pod. We also got a PVC (Persistent Volume Claim) on K8s so that our datasets, library cache, and results can be saved from job to job. All of these configurations are saved onto the repo so once we auth into the namespace on local machines. We can run the script and it will create a job on K8s automatically.
