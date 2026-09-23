# Waabi Interview Prep: E2E Learning & Modern Architectures

## Purpose

This project distills the core concepts from CSC2626 (Imitation Learning for Robotics) that are
directly relevant to **Waabi's focus on end-to-end learning and cutting-edge neural architectures**.
It covers only the weeks and papers that matter for autonomous systems engineering at Waabi: from
behavioral cloning fundamentals, to offline learning for safe deployment, to the latest diffusion
models and vision-based imitation learning approaches that scale via internet video.

**Goal:** develop interview-ready depth on E2E learning, policy learning from demonstrations, and
modern architectures — what Waabi actually hires for.

## Who these notes are for

You're a complete beginner to imitation learning, robotics, and reinforcement learning/control —
including ordinary robotics vocabulary, not just theory. You already know: backpropagation,
MLPs/CNNs, loss functions, standard supervised training loops. That's assumed; we don't re-derive it.

**What we define from scratch:**
- Robotics/hardware vocabulary: end-effector, gripper, manipulator/robot arm, degrees of freedom
  (DOF), joint space vs. task space, workspace, kinematics, proprioception, RGB-D camera, force/torque
  sensing, embodiment, teleoperation, kinesthetic demonstration.
- RL/control/IL theory: MDP, policy, state/observation/action space, reward/cost, on-policy vs.
  off-policy, covariate shift, horizon, rollout, trajectory, distribution shift, offline learning.
- Modern architectures you'll see at Waabi: diffusion models applied to action generation, energy-based
  policies, contrastive representation learning, vision transformers for robotics, video foundation
  models applied to imitation learning.

**How we teach it:**
- Plain-language analogy first, formal definition second. Never assume terms are "obviously known."
- Inline definitions where terms first appear; glossary is a lookup index, not the primary source.
- For formulas: show derivation intuition, not just the result. Why does the formula have that shape?
- Disambiguate overloaded terms (e.g., "policy" as function vs. distribution) each time they recur.
- No dense blocks of text — long paragraphs signal a concept needs restructuring. Split at idea
  boundaries, pull algorithm walkthroughs into their own blocks.
- Diagrams for every concept that's hard to visualize in prose alone: the DAgger loop, how diffusion
  models generate actions, how visual features flow through a policy network, how offline RL constraints
  work, comparisons between E2E and modular pipelines. Diagrams show the actual robotics task
  (camera → network → action) and the architecture explicitly, never just abstract boxes.

## Writing procedure — concepts and architectures first, papers as evidence

These notes teach concepts and neural-network architectures ground-up from first principles as the
primary content — papers anchor already-taught concepts with concrete implementation and empirical
results, never organize the notes themselves. Notes are never drafted directly off the slides or
assembled as paper summaries stitched end-to-end.

**For the five Waabi-relevant weeks (1, 3, 4, 10, 11):**

1. **Identify all concepts the notes depend on**, prioritizing what's most relevant to autonomous
   systems: behavioral cloning and why it fails at scale (compounding error), offline RL and how it
   enables safe deployment, diffusion models applied to action generation, visual imitation learning
   at scale (video-based learning), and modern representation learning for robustness.
2. **Identify the underlying math/algorithm each concept assumes** — without assuming familiarity.
   Behavioral cloning's quadratic-in-horizon error, DAgger's regret bound, diffusion model
   reverse-process intuition, contrastive loss formulations — each gets full from-scratch treatment.
3. **Map the ground-up dependency graph** — what's prerequisite, what formalizes what, what
   builds on what — before writing a sentence. This graph, not the lecture's slide order, determines
   exposition order and where Paper Breakdowns get placed.
4. **Write bottom-up narrative flow**: each concept introduced only after its dependencies, with
   explicit connective tissue showing how they relate. Make clear how Week 1 foundational ideas
   (behavioral cloning, DAgger) feed into Week 3 offline learning, which feed into Week 4 diffusion
   models, which connect to Week 10's visual approaches and Week 11's representation learning.

## Paper Breakdowns

Week 1 (and most weeks after it) leans heavily on primary papers — going through them to understand
key concepts and training methods is a defining part of this course. **Every paper linked for a
given week counts as core material — required and optional/appendix alike, with no distinction in
how thoroughly these notes cover it.** "Optional" describes how the lecture frames a reading, not
how much depth it gets here: whenever the lecture leans on *any* linked paper, required or
appendix, that paper gets its own clearly-marked subsection, placed inline where the narrative first
needs it (never in a trailing bibliography dump, and never left as a bare name-drop the way
appendix methods sometimes are in the slides themselves), with this fixed shape.

A Paper Breakdown is never the primary structuring device (see "Writing procedure" above): its job
is only to anchor already-taught material with this specific paper's concrete numbers,
implementation, and results. If a week's notes read as Paper Breakdowns stitched together with
little concept-teaching in between, that's a signal the concept-first procedure wasn't followed for
that paper's underlying idea, and the section needs a from-scratch conceptual lead-in written first.

The fixed shape:

- **Problem/motivation** — what gap or failure mode the paper is responding to.
- **Key idea/innovation** — the one-sentence core insight.
- **Method & training procedure** — the actual algorithm, loss function, and/or architecture, in
  enough depth to reproduce the core idea, not just cite it.
- **Results** — what the paper empirically showed. Never invent or round a number that isn't
  actually in the paper.
- **How it connects** — what earlier concept in the notes it builds on, and what later concept or
  paper it feeds into.

## Explaining novel neural network architectures

Whenever a lecture or any covered paper (required or optional/appendix) introduces a new, novel, or
meaningfully-improved network architecture beyond the ordinary MLP/CNN/standard-training-loop
baseline already assumed known (see "Who these notes are for"), that architecture gets dedicated
treatment, inline at first use:

- If the architecture is the paper's own contribution, fold the treatment into that Paper
  Breakdown's "Method & training procedure" subsection.
- If it's a baseline/comparison architecture a paper only compares against rather than introduces
  (e.g. an "LSTM-GMM" baseline named only by score in a results table), give it its own short
  subsection immediately adjacent instead of forcing it into someone else's Paper Breakdown
  template. This applies even to architectures mentioned only as a baseline — a baseline's
  structure is exactly what explains why it under/overperforms the method being studied, so it
  deserves the same structural treatment, not just a name-drop of its score.

Either way, the treatment must explicitly cover:

- **Structure** — how the architecture is actually built as a neural network: what network(s),
  heads, or sub-modules it's composed of and how they connect, not just the abstract mathematical
  idea.
- **Input/output/label/loss** — what exactly is fed in as input, what exactly comes out as output,
  what serves as the training label/target, and what loss function is used.
- **What's genuinely innovative** about it relative to other architectures already covered
  elsewhere in the notes — e.g. contrast an energy-based model's "evaluate-and-search" design
  against an ordinary regression head's "predict-directly" design, against a diffusion model's
  "iteratively denoise" design, against a Mixture Density Network's "predict distribution
  parameters directly" design.
- **Build bottom-up from the assumed MLP/CNN baseline, never top-down from the named concept.** The
  target reader knows only ordinary MLPs/CNNs, backprop, and standard supervised training loops (see
  "Who these notes are for"). An autoencoder, a variational/conditional autoencoder, a
  transformer/self-attention layer, a diffusion/denoising process, and a contrastive/InfoNCE
  objective are each themselves unfamiliar, advanced ideas to this reader — not small steps past
  that baseline — doubly so when applied outside the text/image contexts they're normally taught in.
  Give each its own plain-language analogy before formalism (same as any other new term under "Who
  these notes are for"), showing what problem it solves, what concretely changes relative to what
  the reader already has, and — when applied to this course's non-standard data — an explicit
  bridge (e.g. what plays the role a word/token plays in a transformer, here). The failure mode to
  avoid: naming the concept and glossing what it does, however accurately, without this treatment.
- **Use diagrams liberally, not as occasional decoration.** Favor a diagram wherever it would show
  any of the following, and default to including one rather than leaving it to prose alone:
  - The architecture concretely **applied to the actual robotics task** — real inputs (a camera
    image, a joint-state reading, etc.) flowing through the network to a real robot action, not an
    abstract box diagram detached from what the robot is actually doing.
  - The paper's **core conceptual innovation in one picture** — e.g. an energy landscape's two
    separate valleys versus a regression surface forced to interpolate between them.
  - A **side-by-side contrast against a standard or baseline policy network**, so the reader sees
    exactly what structurally changed (what moved from output to input, what got added, what got
    removed) rather than having to infer it from prose.
  - When the architecture predicts a multi-step *sequence* (e.g. a chunk of future actions), make
    temporal ordering visible on the diagram itself: which output slot corresponds to which future
    timestep, and — when overlapping predictions get combined — that they're aligned by absolute
    timestep, not by position within their own chunk.

## Grouping architecture families, not paper order

Notes must teach concepts and architectures ground-up, in logical groupings — not read as a
sequence of paper summaries ordered by when the lecture happens to introduce each one. This is the
lesson of Week 1's §6/§7 restructuring (EBM policies and action-chunking policies were originally
scattered across unrelated sections purely because their source papers first appeared under
different lecture topics) and applies to every future week:

- **When two or more of a week's covered papers (required or optional/appendix) implement different
  versions of the same underlying idea** (e.g. several energy-based-model variants; several
  action-chunking architectures), group and teach them together as one shared unit, even if their
  source papers were originally introduced under different lecture topics (e.g. one under hardware/data
  collection, another under generative policies) — rather than following the order the
  lecture/syllabus happens to introduce the papers in. This explicitly supersedes "Explaining novel
  neural network architectures" above's default of folding an architecture into its own paper's
  Paper Breakdown, for this specific case: pull the shared architecture teaching into one dedicated
  subsection placed before the first relevant Paper Breakdown, have each paper's own Paper
  Breakdown reference that shared subsection instead of re-deriving the mechanism, and keep only
  genuinely paper-specific details (its own loss formula, its own hyperparameters/results) inline
  in that paper's own breakdown.
- **A mechanism referenced by name across multiple papers without being fully explained anywhere**
  (the concrete example that prompted this rule: "negative sampling," named in both the IBC and
  Ranking-NCE Week 1 breakdowns but never explained) must get a full, from-scratch, plain-language
  explanation with a worked example the first time it's used — a Wikipedia link plus a one-clause
  gloss is not sufficient on its own.

The bottom-up-baseline and liberal-diagram rules in "Explaining novel neural network architectures"
above apply here too, on every architecture in the grouped family — grouping several papers
together is not a reason to explain any one of them more thinly.

## Source material & coverage scope

**Weeks covered:** 1 (Imitation Learning Fundamentals), 3 (Offline & Batch RL), 4 (Diffusion Planning
& Hybrid IL+RL), 10 (Visual Demonstrations), 11 (Representation Learning & Generalization).

- Course site: https://csc2626.github.io/2026F_website/ — fetch required and optional readings for
  each of these five weeks from the syllabus. Required readings get full Paper Breakdown treatment;
  optional readings are covered only if they directly advance the core concepts (no peripheral papers).
- Fetch papers directly (arXiv, venue PDFs, project pages) — don't rely on the lecture's summary.
  Paper Breakdown claims must trace to the paper itself.
- Prioritize papers that are most actionable for Waabi: ones teaching E2E learning, scaling via
  data, modern architectures, or practical deployment. Skip classical control, hierarchical task
  decomposition, reward learning, and shared autonomy material.

## What goes in, what doesn't

**In these notes:**
- Core concepts: behavioral cloning, DAgger, offline RL, diffusion models for action generation,
  visual IL, representation learning, contrastive methods, distribution shift.
- Paper Breakdowns showing how each paper's innovation concretely solves a real robotics problem.
- Algorithm walkthroughs and worked examples for anything a Waabi interviewer might ask you to
  explain on a whiteboard (DAgger's loop, diffusion model reverse process, contrastive loss).
- Diagrams illustrating architectures applied to real robotics tasks.

**Not in these notes:**
- Course logistics, assignments, grading.
- Classical optimal control theory (LQR, MPC) beyond the basic idea that they exist.
- Hierarchical task decomposition or program induction.
- Reward learning (inverse RL, IRL), preference learning, human-in-the-loop methods.
- Assistive robotics or shared autonomy (not Waabi's focus).

## File structure

- `CLAUDE.md` — this file (project guidelines for Waabi interview prep).
- `week1-study-notes.md` — Imitation Learning Fundamentals (behavioral cloning, DAgger, compounding
  error, loss functions, multi-modal generative models).
- `week3-study-notes.md` — Offline & Batch RL (offline constraints, CQL, IQ-Learn, safe deployment,
  data-driven learning without online interaction).
- `week4-study-notes.md` — Hybrid IL+RL & Diffusion Planning (guided policy search, diffusion models
  for action generation, tree search integration, planning as generation).
- `week10-study-notes.md` — Visual Imitation Learning (keypoint tracking, video-based learning,
  internet video scaling, causal confusion, action inference from video).
- `week11-study-notes.md` — Representation Learning & Generalization (contrastive learning, self-
  supervised pretraining, generalization guarantees, representation as core to robustness).
- `glossary.md` — Waabi-focused glossary, grouped by topic (not alphabetically) covering: IL concepts,
  offline RL concepts, visual learning concepts, representation learning concepts, architecture terms.
  Entries are beginner-level anchors: one sentence plus cross-references. Never delete an entry;
  deepen it as later weeks cover related ideas.

## Published artifact — single interactive study guide

A single Claude Artifact is the published study guide, covering all five Waabi-relevant weeks. It's
updated in place as weeks are added (never create a second artifact for a new week).

**Structure:** one HTML page with a `<section class="week-block" id="week-N" data-week-number="N"
data-week-title="...">` per covered week (only weeks 1, 3, 4, 10, 11). Each week-block contains:
- Numbered subsections mirroring the concept-dependency order (not the lecture's slide order).
- Paper Breakdowns inline where first needed.
- Worked examples and algorithm walkthroughs in their own callout blocks.
- One `.selfcheck` block at the end (what should I be able to explain in a Waabi interview after
  this section?).

A persistent `<section id="glossary">` follows, with one `.week-card` per covered week. A nav script
builds the sidebar/mobile nav and displays "5 of 5 Waabi-focus weeks" or similar progress indicator.

**Diagrams:** live only in the artifact, never in markdown `.md` files. Include a diagram for any
concept that's hard to visualize in prose alone:
- Showing the actual robotics task (camera → network → robot action), never just abstract boxes.
- Comparing architectures side-by-side: what structurally changed from one to the next?
- Illustrating the DAgger loop, diffusion model sampling, visual feature flow, offline RL constraints.
- Figure captions numbered "Fig. N" sequentially across the whole page. Use original inline SVG with
  CSS custom properties for theme-awareness.

**Math rendering in the artifact:** Use native HTML and Unicode only — **never external math libraries** (KaTeX, MathJax) or CDN dependencies. Artifact viewers may block external scripts entirely. Instead:
- **Superscripts/subscripts:** Use HTML `<sup>` and `<sub>` tags: `T<sup>2</sup>`, `a<sub>t</sub>`.
- **Greek letters and symbols:** Use Unicode directly: π, α, β, ε, σ, μ, ∝, ≤, ≥, ∈, →, etc. All are already in the content; just strip LaTeX delimiters.
- **Display formulas:** Wrap in `<pre>` tags with monospace font and padding for readability.
- **Inline formulas:** Wrap in `<code>` tags for monospace rendering inline with text.
- **Complex expressions:** Use `<sup>`/`<sub>`, Unicode symbols, and spaces/ASCII to approximate: `ℝ^D`, `∇_θ`, `μ<sub>z</sub>`, etc.

This ensures the artifact is **completely self-contained** with zero external dependencies—no CDN blocking issues, no script execution failures, pure native browser HTML.

**Hyperlinks:** every technical term, formula, robotics component, or concept introduced in prose
or tables gets linked on first mention in each section:
- Artifact: inline `<a href="https://en.wikipedia.org/wiki/...">`.
- Markdown notes: `[Term](https://en.wikipedia.org/wiki/...)`.
- Wikipedia only — never course site, paper project pages, or external refs (Paper citations stay
  as plain unlinked text, showing arXiv/venue link).

## Git workflow

Every change made in this project — new week notes, glossary updates, CLAUDE.md edits, artifact
republishes, anything — is committed and pushed to GitHub (`origin/main`) automatically, one
discrete step at a time, never batched into a single end-of-session commit:

1. Finish one discrete step of work: a single file edit, or one logically-grouped set of edits
   belonging to the same step (e.g. "write week N notes," "update glossary for week N," "update
   CLAUDE.md").
2. Immediately stage only the file(s) that step just changed, by name (`git add <path> ...`) —
   never a broad `git add -A` or `git add .` that could sweep in unrelated changes.
3. Commit with a message specific to that step alone.
4. Push to `origin/main` before moving on to the next piece of work.
5. Repeat for the next discrete step. Never let two unrelated edits (e.g. a CLAUDE.md policy
   change and a week's notes) share one commit, and never start the next step while a local commit
   from a previous step is still unpushed.

Don't leave changes sitting uncommitted or unpushed for the user to handle separately.

## Interview preparation roadmap

These five weeks are ordered for interview-readiness progression:

1. **Week 1 → Week 3 → Week 4:** Build deep understanding of the full IL+RL pipeline
   - Week 1 teaches why behavioral cloning fails and introduces DAgger.
   - Week 3 shows how to safely learn offline (Waabi's core: data-driven, no live interaction).
   - Week 4 shows modern planning: diffusion models applied to control.
   - Interview expectation: explain the progression from imitation to offline RL to generative planning.

2. **Week 10 + Week 11:** Modern visual architecture and scaling
   - Week 10: how visual IL scales via internet video (actionable at scale).
   - Week 11: why representation learning matters (generalization and robustness for deployment).
   - Interview expectation: articulate why visual features + representation learning = robust E2E systems.

**Common interview questions these weeks prepare you for:**
- "Explain behavioral cloning and its failure modes" (Week 1).
- "Why does DAgger work? What's the regret bound?" (Week 1).
- "How is offline RL different from online RL? Why can't you just use any RL algorithm on fixed data?" (Week 3).
- "What makes a good offline RL algorithm?" (Week 3).
- "How do diffusion models generate actions? Why use them over direct regression?" (Week 4).
- "How can you learn robot policies from internet videos?" (Week 10).
- "Why is representation learning important? What goes wrong without it?" (Week 11).

## Continuous improvement: refinement workflow

When existing week notes need refinement (to add clarity, fix errors, or deepen coverage):

1. Read the current `.md` file and the artifact's matching `.week-block` (via `action: "read"` on the artifact URL).
2. Identify the gap: missing depth, unclear explanation, missing diagram, inaccurate formulation?
3. Revise the `.md` file to fix the issue, following the writing procedure above (concept-first, then evidence).
4. If a diagram is needed, add it to the artifact only (not the markdown).
5. Update the artifact (read first, republish to the same URL).
6. Commit and push the markdown edit, then the artifact edit separately.
