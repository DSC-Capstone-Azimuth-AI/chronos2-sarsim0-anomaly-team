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

## NRP Configuration and Setup

Asked claude to investigate how the nrp platform worked, and how we could integrate our project to run on the platform. Looked into creating shell scripts that would clone our repo and allow us to run the same CLI commands on a kubernetes pod. We also got a PVC (Persistent Volume Claim) on K8s so that our datasets, library cache, and results can be saved from job to job. All of these configurations are saved onto the repo so once we auth into the namespace on local machines. We can run the script and it will create a job on K8s automatically.
