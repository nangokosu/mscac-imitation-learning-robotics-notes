# Waabi Interview Prep: E2E Learning & Modern Architectures

Interview-focused study notes covering end-to-end learning, offline RL, diffusion-based planning, visual imitation at scale, and representation learning — the five core topics Waabi hires for from CSC2626 (Imitation Learning for Robotics, U of T, Fall 2026).

**📖 Read the notes:** [Waabi Interview Prep](https://claude.ai/artifact/5Eh9TePpNpmwVvLmAbWPoG) — a browsable study guide covering five weeks, publicly viewable with no sign-in required. The markdown in this repo is the same content in plain-text form.

## Coverage

This project focuses exclusively on the five weeks most relevant to Waabi's work on autonomous systems:

| Week | Topic | What it covers |
|---|---|---|
| **1** | **Imitation Learning Fundamentals** | Behavioral cloning, covariate shift, DAgger, and why long-horizon IL needs more than supervised learning. |
| **3** | **Offline & Batch RL** | Learning from fixed datasets without online interaction — distribution shift, policy constraints (BEAR, TD3+BC), and value conservatism (CQL). |
| **4** | **Diffusion Models for Action Generation** | Using diffusion models to generate action sequences instead of direct regression — cutting-edge architecture. *(In development)* |
| **10** | **Learning from Visual Demonstrations** | Scaling via internet video, keypoint tracking, and visual foundation models — the path to real-world deployment. *(In development)* |
| **11** | **Representation Learning & Generalization** | Contrastive learning, self-supervised pretraining, and robustness — why representation matters in deployment. *(In development)* |

**Not covered:** Classical optimal control (Week 2), hierarchical task decomposition, reward learning, and shared autonomy — these are outside Waabi's core focus on E2E learning.

## Files

- [`week1-study-notes.md`](./week1-study-notes.md) — IL Fundamentals
- [`week3-study-notes.md`](./week3-study-notes.md) — Offline RL  
- [`glossary.md`](./glossary.md) — Quick-lookup glossary organized by topic
- [`CLAUDE.md`](./CLAUDE.md) — Project guidelines (how these are structured and maintained)

## Purpose

These notes prepare for Waabi interviews by teaching interview-ready depth on:
- Why behavioral cloning fails and how DAgger fixes it
- The offline RL problem: distribution shift and how to solve it (CQL, policy constraints)
- Modern architectures for action generation (diffusion models)
- Scaling visual IL and learning from internet video
- Why representation learning is essential for robustness

Each concept is explained from first principles with formulas, worked examples, and diagrams.

## Source

Course: CSC2626 (Imitation Learning for Robotics, U of T, Fall 2026)  
https://csc2626.github.io/2026F_website/
