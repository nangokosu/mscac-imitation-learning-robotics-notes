# CSC2626 Imitation Learning for Robotics — Study Notes Project

## Purpose

This project produces running study notes for CSC2626 (Imitation Learning for Robotics, Fall
2026, University of Toronto): one markdown file per lecture week plus a cumulative glossary, to
help the user learn the field from scratch over the term — not to solve assignments or the course
project.

## Who these notes are for

The user is a complete beginner to imitation learning, robotics, and reinforcement
learning/control — including ordinary robotics vocabulary itself, not just the theory layered on
top of it. The one exception: the user already knows general neural network architecture
(backpropagation, MLPs/CNNs, loss functions, standard training loops) — that part is assumed, not
re-derived.

- **Do not** re-explain basic deep-learning machinery (what an MLP/CNN is, backprop, cross-entropy,
  ordinary supervised training loops) — assume it.
- **Do** define everything else from scratch on first use, with a plain-language analogy before the
  formal definition — including robotics/hardware vocabulary that might sound ordinary but can't be
  assumed: end-effector, gripper, manipulator/robot arm, degrees of freedom (DOF), joint space vs.
  task space, workspace, kinematics, proprioception, RGB-D camera, force/torque sensing, embodiment,
  bimanual manipulation, leader-follower arm, teleoperation, kinesthetic demonstration. Treat none
  of these as "obviously known" just because they sound like everyday words.
- Also define every RL/control/IL-theory term from scratch: MDP, policy, state/observation/action
  space, reward/cost, on-policy vs. off-policy, regret/no-regret learning, covariate shift, horizon,
  rollout, receding-horizon control, energy-based model, action chunking, epistemic vs. aleatoric
  uncertainty, etc.
- Put definitions inline where a term first appears, not only in the glossary — a reader should
  never have to jump elsewhere to follow the current sentence. The glossary is a lookup-speed index;
  its entry can be a shorter echo of the inline definition.
- For formulas, show the derivation intuition — why the formula has that shape — not just the
  result.
- Cross-reference where a concept reappears or gets formalized in a later week.
- Disambiguate overloaded terms explicitly every time they recur (e.g. "policy" as a function vs.
  a distribution; "loss" in the imitation sense vs. the RL-reward sense) — state in-line which is
  meant.
- **No dense blocks of text** — a long unbroken paragraph signals a concept needs restructuring, not
  just tighter prose. Split multi-idea paragraphs at idea boundaries, pull worked examples and
  algorithm walkthroughs into their own callout-style blocks, and use lists for enumerations (e.g.
  DAgger's algorithm steps, the five parts of a Paper Breakdown).
- Illustrate geometric/procedural ideas (the DAgger interaction loop, embodiment transfer across
  robots, an energy landscape, error-growth curves, etc.) with a diagram proactively, whenever prose
  alone would be hard to picture — see "Published artifact" below for how these are built.

## Writing procedure — concepts and architectures first, papers as evidence

These notes teach concepts and neural-network architectures ground-up from first principles as the
primary content of every week — no paper, required or optional/appendix, ever organizes the notes.
Instead, a paper's role is to *anchor* an already-taught concept or architecture: it supplies the
concrete implementation, numbers, and results that make the abstract idea concrete. Notes are never
drafted directly off the slides, or assembled as a sequence of paper summaries stitched end to end.
For each new week:

1. **Identify all concepts the lecture actually depends on**, prioritizing anything a current or
   upcoming assignment leans on for the deepest treatment, even if the lecture itself only mentions
   it in passing. This step also reads that week's entry in `weekly-attention-points.md` (see
   "Professor's weekly attention-points checklist" below) alongside the slides and readings — every
   bullet listed there for the week must map onto a concept in the dependency graph built next, not
   just get name-dropped somewhere in the prose.
2. **Identify the underlying math/algorithm each concept assumes**, without assuming familiarity —
   DAgger's regret bound, behavioral cloning's quadratic-in-horizon error argument, an energy-based
   model's energy formulation each get their own from-scratch treatment, not just a citation.
3. **Map the ground-up relationships between concepts** — prerequisite, formalizes, builds-on —
   before writing a single sentence of exposition. This dependency graph, not the lecture's own
   slide order, determines section order and where each Paper Breakdown gets inserted.
4. **Only then write the narrative flow**, still bottom-up: each concept introduced only after what
   it depends on, with the connective tissue between concepts made explicit.

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

## Professor's weekly attention-points checklist

`weekly-attention-points.md` holds the professor's own per-week "Things to pay attention to" list,
originally circulated as a Google Doc. It is a required source for every week, supplementing (never
replacing) the lecture slides and readings under "Source material" below — the slides and readings
supply the depth and evidence, this checklist supplies the professor's own list of what must not be
missed. Every bullet listed for a given week is a mandatory coverage item: the corresponding
`weekN-study-notes.md` and the artifact's matching `week-N` section must each explicitly name and
explain it, at the level of depth the bullet implies (a narrow sub-claim like "DAgger's guarantees
on compounding error as a linear function of the horizon" needs that specific claim stated, not
just general DAgger coverage that a reader would have to infer it from).

The source Google Doc cannot be reliably re-fetched by tooling (it requires Google sign-in, and
automated fetching only returns a paraphrased view rather than the exact text), so
`weekly-attention-points.md` is not auto-synced from it. If the professor updates the doc during
the term, the user re-pastes the new text and the file is updated by hand at that point — always
check whether a given week's entry might have been revised since the notes for that week were last
touched.

## Source material

- Course site: https://csc2626.github.io/2026F_website/ (schedule and full syllabus on `index.html`;
  lecture slides under `/lecs/wNN/lecNN.html`; required/optional readings linked per-topic under
  `index.html#week-N`). Fetch and cover *both* lists in full — see "Paper Breakdowns" above.
- Fetch papers directly (arXiv abstract/PDF, ACM/project pages) rather than relying on the lecture's
  own summary of them — a Paper Breakdown's claims must trace to the paper itself.
- Skip course-logistics slides (schedule, grading, staff, policies) — notes start from the first
  technical content slide of each lecture.

## What never goes in these notes

Never provide a working solution to a graded assignment or the course project (e.g. runnable
DAgger/behavioral-cloning implementation code for this course's actual assignment environments) —
explain the underlying algorithm and math fully, but leave the specific implementation/deliverable
to the assignment.

## File structure

- `CLAUDE.md` — this file.
- `weekly-attention-points.md` — the professor's verbatim per-week "things to pay attention to"
  checklist; see "Professor's weekly attention-points checklist" above.
- `weekN-study-notes.md` — one per lecture week, following the structure established in
  `week1-study-notes.md`: numbered sections mirroring the concept-dependency order (not the
  lecture's own slide order), Paper Breakdowns inline at first use, worked examples/algorithm
  walkthroughs in their own callout-style blocks, inline Wikipedia hyperlinks on first mention of
  each technical term (same convention and scope as the artifact's "Hyperlinks" rule below — notes
  just stay plain markdown text/tables, with no embedded diagrams/SVG figures), and a "Part A" that
  ends with a pointer to the shared glossary rather than repeating it.
- `glossary.md` — the single, cumulative, beginner-facing glossary, grouped by the course week each
  term is actually covered (not flat A–Z). Update it every time a new week's notes are written:
  promote that week's terms out of any "forward-looking preview," and add genuinely new
  forward-looking terms for weeks further out if the new lecture introduces them. Never delete an
  existing entry — later weeks may deepen a definition, but the original beginner-level anchor
  stays.

## Published artifact — one running notebook, not one per week

A single Claude Artifact covers the whole course, updated in place every time a new week is added
(never create a second artifact for a new week). It follows the same mechanics as the sibling
`computational-imaging-notes` project's artifact, with its own fresh visual identity (this is a
different course/subject, not a reskin):

It's one HTML page with a `<section class="week-block" id="week-N" data-week-number="N"
data-week-title="...">` per lecture week, each containing that week's own numbered subsections
(including its Paper Breakdowns) and ending in its own `.selfcheck` block, followed by a persistent
`<section id="glossary">` holding one `.week-card` per syllabus week. A script builds the
sidebar/mobile nav and the "N of 13 lecture weeks logged" progress indicator from whatever
`.week-block`/`.week-card` elements exist — adding a week just means adding a new `.week-block` (and
a matching glossary card, if one doesn't already exist as a forward-looking preview).

Diagrams live only in the artifact, never in the `weekN-study-notes.md` files, matching the pattern
set by Week 1. Any geometric or procedural idea worth illustrating gets an original inline SVG
`<figure class="chart-fig">`, styled with the artifact's own CSS custom properties so it stays
theme-aware — never a screenshot of the actual lecture slide. Figure captions are numbered "Fig. N"
sequentially in document order across the whole page (not per week), so inserting a new figure
ahead of existing ones means renumbering the ones after it. Keep the nav script's coding standard:
JSDoc per function, fully descriptive names, no unexplained hardcoded numbers.

**Hyperlinks:** every technical term, named concept, instrument, formula, phenomenon, robotics
component, or historical figure introduced in prose or tables — in the artifact **and** in the
`weekN-study-notes.md` files alike — gets linked on first mention within its section: an inline
`<a href="https://en.wikipedia.org/wiki/...">` in the artifact's HTML, a markdown
`[Term](https://en.wikipedia.org/wiki/...)` in the notes' plain-text prose (the notes' own
plain-text/tables-only/no-diagrams rule above is unaffected — hyperlinks are just inline markdown
links, not embedded HTML or figures). Wikipedia is the only link target ever used — never the
course site, a paper's own project page, or other external references (Paper Breakdown citations
are the one exception — the paper's own arXiv/venue link stays as plain, unlinked citation text,
not a Wikipedia link). Link to the specific page/anchor that actually matches the concept as used,
not just a plausibly-named page — this is exactly what the fact-auditor's link-resolution check
verifies.

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

## When asked to add a new week

1. Fetch/read that week's lecture slides and any linked readings the same way Week 1 was researched
   (see "Source material" above), and read that week's entry in `weekly-attention-points.md` (see
   "Professor's weekly attention-points checklist" above) alongside them. Treat every bullet listed
   there for the week as a mandatory coverage item for both the notes and the artifact section
   written in steps 2 and 4 below.
2. Write `weekN-study-notes.md` following the Week 1 structure and tone, arrived at through the
   "Writing procedure" sequence above rather than by drafting prose directly off the slides.
   Wikipedia-link each technical term on first mention the same way as Week 1 (see "Hyperlinks"
   above) — the notes carry the same links as the artifact, just as plain markdown, not HTML. Before
   treating the notes as done, check them against that week's `weekly-attention-points.md` entry
   bullet by bullet and confirm each is explicitly named and explained, not just implied.
3. Update `glossary.md`: move that week's terms into a dated `## Week N` section, keeping the
   beginner-level one-sentence-plus-cross-reference format.
4. Update the running artifact (URL in `README.md`) by reading it first (`action: "read"`), then
   republishing with `url` set to that same address: append a new `.week-block` after the previous
   one (before the `<hr class="div"/>` that precedes `#glossary`), and refresh the matching
   `#glossary` `.week-card` so it reflects what was actually covered rather than a forward-looking
   guess. Reuse the established visual language exactly — don't redesign per week. Wikipedia-link
   every newly introduced term the same way as Week 1 (see "Hyperlinks" above). Keep the nav
   script's coding standard (JSDoc per function, fully descriptive names, no unexplained hardcoded
   numbers). Load the `artifact-design` skill before touching the HTML.
5. Commit and push after each of steps 2–4 individually, per "Git workflow" above — notes, then
   glossary, then artifact, each its own staged-by-name commit pushed before the next step starts.
