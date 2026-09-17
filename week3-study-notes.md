# CSC2626 Imitation Learning for Robotics — Week 3 Study Notes

**Topic:** Offline / Batch Reinforcement Learning
**Source:** Lecture 3 slides (F. Shkurti, CSC2626, Fall 2026) plus the week's required and optional/appendix readings, each summarized in its own Paper Breakdown or Reading write-up below, plus a secondary-topic reading on transitioning from offline to online RL.
**Scope:** Logistics slides skipped — notes start from the course's actual technical content.

---

## 0. Framing: a third way to get a policy

Week 1 ([`week1-study-notes.md`](./week1-study-notes.md)) learned a policy purely from expert demonstrations, with no reward signal at all: behavioral cloning (BC) fits `π_θ(o) → a` directly to (observation, action) pairs. Week 2 ([`week2-study-notes.md`](./week2-study-notes.md)) went the opposite direction and *computed* a policy from an explicit cost function and a known (or learned) dynamics model, using optimal control.

This week's setting borrows a piece of each, but is genuinely a third thing: a fixed, static dataset of **(state, action, reward, next-state)** transitions — collected once, by some unknown policy, and never added to again — from which a policy must be learned that does as well as possible, without ever touching the real system again during training, and without any dynamics model. This is **offline** (equivalently: **batch**) **[reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning)**.

| | Behavioral cloning (Wk 1) | Optimal control (Wk 2) | Offline RL (this week) |
|---|---|---|---|
| What's given | (observation, action) demonstrations | A cost function + a dynamics model | A fixed dataset of (state, action, **reward**, next-state) transitions |
| Reward/cost used during training? | No | Yes (hand-specified) | Yes (given in the dataset) |
| Dynamics model needed? | No | Yes | No |
| Further environment interaction during training? | No | No (control laws are computed, not learned from interaction) | **No** — this is the defining constraint |
| Can it ever beat the data that generated it? | No — bounded by the demonstrator (§4.2 below) | N/A (no demonstrator to bound it) | **Potentially yes** — via reward-driven "trajectory stitching" (§4.2) |

Because offline RL uses rewards and Bellman-style value backups (Week 2 §1's machinery, not re-derived here) but — like BC — never interacts with the environment during training, most of this week's difficulty comes from a single question: *what goes wrong when you try to run ordinary reinforcement-learning algorithms on a dataset instead of a live environment, and how do you fix it?* Sections 1–3 build the RL vocabulary and off-policy value-learning machinery needed to state that question precisely; §5 states the problem; §6–7 present the two families of fixes.

---

## 1. RL terminology from scratch

### 1.1 The Markov Decision Process, formalized

Week 1 and Week 2 used "state," "action," and "reward" informally. This week needs the full, precise object every RL algorithm is defined over: the **[Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process)** (MDP), a tuple `(S, A, T, r, γ, d_0)`:

- **`S`** — the state space (all possible states `s`).
- **`A`** — the action space (all possible actions `a`).
- **`T(s' | s, a)`** — the transition function: the probability of landing in state `s'` after taking action `a` in state `s`. This is Week 2's dynamics model `f`/`p(x_{t+1}|x_t,u_t)`, renamed and generalized to allow arbitrary stochasticity.
- **`r(s, a)`** — the reward function (RL's sign convention: bigger is better, the negative of Week 2's cost `c`).
- **`γ ∈ (0, 1]`** — the discount factor, exactly as in Week 2 §1.
- **`d_0(s)`** — the initial-state distribution: how the very first state of an episode is drawn.

A **trajectory** `τ = (s_0, a_0, s_1, a_1, ..., s_T)` is one full rollout; its probability under a policy `π` factors as `p_π(τ) = d_0(s_0) · Π_t π(a_t|s_t) · T(s_{t+1}|s_t,a_t)` — the initial state, times every action the policy chose, times every transition the environment produced. The **return** of a trajectory is the discounted sum of rewards, `Σ_t γ^t r(s_t, a_t)`, and the RL objective is to find `π` maximizing its expected value, `J(π) = E_{τ~p_π(τ)}[Σ_t γ^t r(s_t,a_t)]`.

**Disambiguating "policy" again.** Following Week 2 §1's convention, "policy" is used here in its RL sense — a distribution `π(a|s)` an action is sampled from — unless a specific paper below defines a deterministic control law, which will be flagged explicitly.

### 1.2 State-visitation distribution

One more piece of notation is needed before §5 can state the offline-RL problem precisely: the **state-visitation distribution** `d^π(s)`, the distribution over states induced by running policy `π`. Concretely: if `π` were rolled out for many episodes and every visited state were recorded into a histogram (each timestep's state weighted by `γ^t`, then normalized), `d^π(s)` is what that histogram converges to. Two different policies `π_1 ≠ π_2` in general visit *different* parts of the state space — `d^{π_1} ≠ d^{π_2}` — and §5 will show this simple fact is the root of offline RL's central difficulty.

### 1.3 Seven dichotomies that carve up "RL methods"

The lecture organizes the landscape of RL algorithms along seven independent axes. Each is a genuine binary choice a method-designer makes; a single algorithm is describable as one point in this seven-dimensional space of choices.

- **Episodic vs. non-episodic (continuing).** An episodic task has a natural end (a robot either succeeds, fails, or times out) and the return is a finite-horizon sum `Σ_{t=0}^{T} γ^t r_t`. A non-episodic (continuing) task never terminates, and the return is an infinite discounted sum `Σ_{t=0}^{∞} γ^t r_t` — the discount `γ < 1` is what keeps this sum finite.
- **[Tabular](https://en.wikipedia.org/wiki/Q-learning) vs. function-approximation.** Tabular methods store one number per state (or state-action pair) in an explicit lookup table — exact, but impossible once `S`/`A` are continuous or astronomically large. Function-approximation methods instead fit a parameterized function (a neural network `Q_θ(s,a)`) that generalizes across states it has never exactly seen — the only option for image-based robotics.
- **[Exploration vs. exploitation](https://en.wikipedia.org/wiki/Exploration-exploitation_dilemma).** Exploitation takes the action currently believed best; exploration deliberately takes a possibly-suboptimal action to gather information that might improve future decisions. Standard online RL must balance both; §4 below shows offline RL sidesteps this axis entirely, since there is no live decision to make during training at all.
- **Model-based vs. model-free.** Model-based RL (Week 2 §5) learns or is given a dynamics model `T(s'|s,a)` and plans through it; model-free RL learns a value function or policy directly from experienced transitions, never building an explicit model of the world's dynamics. Every algorithm in this week's notes is model-free.
- **Policy optimization vs. value-function estimation.** Policy-optimization methods directly search over policy parameters `θ` to maximize `J(π_θ)` (§2 below). Value-estimation methods instead estimate a value function `Q(s,a)`/`V(s)` and derive a policy from it (e.g. act greedily: `π(s) = argmax_a Q(s,a)`) — §3 below.
- **On-policy vs. off-policy.** Deepened in §2.3: on-policy methods can only learn about the exact policy currently generating data; off-policy methods can learn about a *different* target policy than whatever policy produced the data being trained on.
- **Batch (offline) vs. online.** Deepened in §4.1: online methods continually collect new data with the very policy being updated, so the training data distribution shifts round to round; batch/offline methods train on one fixed, unchanging dataset collected in advance, with zero new interaction.

---

## 2. From values to actions: policy gradients and actor-critic

### 2.1 Policy gradients, derived from scratch

The most direct way to do policy optimization is **[policy gradient](https://en.wikipedia.org/wiki/Policy_gradient_method)** ascent: compute `∇_θ J(π_θ)` and step `θ` in that direction. The obstacle is that `J(θ) = E_{τ~p_θ(τ)}[R(τ)]` is an expectation over trajectories *sampled from* `π_θ` itself — the sampling operation isn't differentiable, so the gradient can't be pushed through it the way backpropagation pushes through an ordinary differentiable layer.

The fix is the **log-derivative trick**, a single line of calculus (`∇_θ p_θ(τ) = p_θ(τ) · ∇_θ log p_θ(τ)`, which follows directly from `∇ log f = ∇f / f`) that rewrites the gradient of an expectation as an expectation of a (now differentiable) quantity:

```
∇_θ J(θ) = ∇_θ ∫ p_θ(τ) R(τ) dτ
         = ∫ ∇_θ p_θ(τ) R(τ) dτ
         = ∫ p_θ(τ) ∇_θ log p_θ(τ) R(τ) dτ
         = E_{τ~p_θ(τ)} [ ∇_θ log p_θ(τ) · R(τ) ]
```

Since `log p_θ(τ) = log d_0(s_0) + Σ_t log π_θ(a_t|s_t) + Σ_t log T(s_{t+1}|s_t,a_t)`, and only the middle term depends on `θ`, everything involving the (unknown, un-differentiated) transition function `T` drops out of the gradient entirely:

```
∇_θ J(θ) = E_{τ~p_θ(τ)} [ ( Σ_t ∇_θ log π_θ(a_t|s_t) ) · R(τ) ]
```

This is **[REINFORCE](https://en.wikipedia.org/wiki/Reinforcement_learning)**: sample trajectories by actually running `π_θ`, then push up the log-probability of every action taken, weighted by how good the whole trajectory turned out (`R(τ)`) — actions from high-return trajectories get reinforced, actions from low-return ones get suppressed. No model of the environment is ever needed, only the ability to sample from it.

### 2.2 On-policy actor-critic: a learned baseline

Raw REINFORCE has famously high variance: `R(τ)` is one noisy sample of a whole trajectory's return, and every action in that trajectory gets the exact same credit/blame regardless of whether it actually contributed. **Actor-critic** methods reduce this variance by learning a second function — the **critic**, an estimate `V_φ(s)` of the value function — and using it to compute an **advantage** `A(s,a) = Q(s,a) − V(s)`, replacing the raw return in the policy-gradient formula:

```
∇_θ J(θ) ≈ E [ ∇_θ log π_θ(a_t|s_t) · A(s_t, a_t) ]
```

`A(s,a)` answers a more targeted question than the raw return does: *was this specific action better or worse than what this state's average action would have achieved?* — subtracting the state-dependent baseline `V(s)` removes noise common to every action from the same state without introducing bias (a standard variance-reduction fact for this kind of estimator). The **actor** (`π_θ`) is updated by the policy-gradient rule above; the **critic** (`V_φ`) is updated by ordinary [temporal-difference](https://en.wikipedia.org/wiki/Temporal_difference_learning) regression toward `r + γV_φ(s')`. Both networks are trained using data generated by the current `π_θ` — this is what makes vanilla actor-critic **on-policy**: the critic is only ever asked to evaluate actions the current policy would plausibly take.

### 2.3 Off-policy actor-critic: behavior policy vs. target policy

The vocabulary every remaining section of this week depends on: split "the policy" into two distinct roles.

- **Behavior policy, `π_β(a|s)`** — whatever policy actually generated the data being trained on (a human teleoperator, a scripted controller, an earlier partially-trained agent — anything).
- **Target policy, `π_θ(a|s)`** — the policy currently being learned/improved.

On-policy methods require `π_β = π_θ` at all times (the data must come from the exact policy being evaluated). **Off-policy** methods allow `π_β ≠ π_θ`: the critic can be trained on data from *any* behavior policy and still produce a valid estimate of the target policy's value, because a Bellman backup (§3.1) only needs to know what reward and next-state *actually resulted* from a given `(s,a)` — it never needs to know what policy chose that `a` in the first place. This is precisely the property that makes offline RL possible at all: a fixed dataset generated by an unknown `π_β` can still, in principle, be used to learn about a much better `π_θ`.

---

## 3. Value-based off-policy learning: Q-learning, Fitted Q-Iteration, and QT-Opt

### 3.1 Q-learning and the Bellman optimality backup

Week 2 §1 defined `Q*(x,a)` as the value of taking action `a` now and acting optimally forever after, satisfying `Q*(s,a) = r(s,a) + γ E_{s'}[max_{a'} Q*(s',a')]`. **[Q-learning](https://en.wikipedia.org/wiki/Q-learning)** turns this identity into a learning rule: given a transition `(s,a,r,s')`, push the current estimate `Q(s,a)` toward the **Bellman optimality backup**:

```
Q(s,a) ← r + γ · max_{a'} Q(s', a')
```

The `max_{a'}` is what makes this update off-policy: the target depends only on the transition `(s,a,r,s')` itself, never on what action the data-generating policy actually took next. This is exactly why Q-learning can, in principle, learn from data generated by any behavior policy at all — and, as §5 will show, exactly why it breaks when that data is all there is.

### 3.2 Fitted Q-Iteration: Q-learning as repeated supervised regression

**[Fitted Q-Iteration](https://en.wikipedia.org/wiki/Q-learning)** (FQI) turns the Bellman backup into an explicit batch procedure, alternating two steps over a fixed set of transitions: (1) using the *current* Q-function estimate, compute a Bellman-backup target value `r + γ max_{a'} Q(s',a')` for every transition in the batch; (2) fit a new Q-function — via ordinary supervised regression — to those target values. Repeating this outer loop is exactly Q-learning, just organized as batches of supervised-learning problems rather than one-transition-at-a-time updates — a structure that generalizes naturally to a fixed, static dataset, since nothing about it requires interacting with a live environment.

> **Fitted Q-Iteration in practice.** Classical [neural-network](https://en.wikipedia.org/wiki/Artificial_neural_network)-based FQI predates the deep-offline-RL methods covered later this week by over a decade.

### Paper Breakdown: Neural Fitted Q-Iteration for RoboCup soccer

*Citation: Martin Riedmiller, Thomas Gabel, Roland Hafner, Sascha Lange, "Reinforcement Learning for Robot Soccer," Autonomous Robots 27(1):55–73, 2009.*

**Problem/motivation.** Learning robot control policies directly through trial-and-error interaction is expensive on real hardware — every failed episode costs real time and risks damage — so the paper asks whether a *batch*, off-policy value-based method can learn effective robot-soccer skills from a modest amount of stored, reusable interaction data, rather than the huge sample counts online RL typically needs.

**Key idea/innovation.** A general three-module **batch RL** framework — (1) sample experience, (2) generate a training-pattern set via a dynamic-programming target-value computation, (3) batch-mode supervised learning (using the Rprop training algorithm) to fit a multilayer perceptron to that pattern set — repeated across outer-loop passes. **Neural Fitted Q-Iteration (NFQ)** is the concrete instance: the target for transition `(s,a,s')` at outer-loop iteration `k+1` is `Q_{k+1}^{target}(s,a) := c(s,a,s') + γ min_{b∈A(s')} Q̃_k(s',b)` (a cost-minimization form of the Bellman backup from §3.1, since the paper poses control as cost minimization rather than reward maximization). NFQ is model-free, off-policy, and reuses the *entire* aggregated experience set on every outer-loop pass, since a stored transition's validity for the target computation never depends on which policy originally produced it.

**Method & training procedure.** Three case studies, all on the **[RoboCup](https://en.wikipedia.org/wiki/RoboCup)** MidSize-league robot platform (or its 2D-simulation counterpart):

- **Aggressive defense behavior (ADB)**, 2D simulation league: learning to disturb/tackle a ball-dribbling opponent, 9-dimensional state, 76 discretized actions, trained via fitted policy iteration with Monte-Carlo policy evaluation.
- **DC motor speed control**, real MidSize robot wheel motors: a 4-dimensional state (current, integrating output, speed error, angular velocity), NFQ trained directly on real hardware.
- **Ball dribbling**, real MidSize robot: a 6-dimensional state (robot velocity, rotation speed, relative ball position, heading error), a 9-input/2×20-hidden-neuron/1-output MLP fit via NFQ.

**Results.**
- ADB: after ~700 training episodes, success rate reached 83.8% (vs. 52.8% for the previous hand-coded controller), with full-failure rate dropping from 19.7% to 8.2%; the ADB policy contributed to the team's **RoboCup world championships in 2007 and 2008**.
- Motor control: an effective controller emerged from only **~198 seconds** of real-motor interaction time, after just 30 NFQ iterations.
- Ball dribbling: after 132 real-robot trials (~30 minutes of actual interaction time), the learned controller made sharper, faster turns than the prior hand-tuned routine, and became "the first behavior on our MidSize robot that was completely learned on the real robot," used in competition from 2007 onward. Overall, up to **80% of in-game decisions** by the team's competition agent were driven by learned neural networks by the end of the project, which won **five RoboCup world championships**.

**How it connects.** NFQ is the direct historical ancestor of every offline-RL algorithm in this week's notes — same core loop (batch Bellman-target computation, then supervised regression), just with a tabular-era MLP instead of the deep, image-conditioned Q-networks the rest of this week uses, and without any of the safeguards against extrapolation error that §5–§7 introduce (the paper's small, low-dimensional state spaces largely sidestepped that failure mode).

### 3.3 QT-Opt: Q-learning made continuous

Q-learning's greedy policy, `π(s) = argmax_a Q(s,a)`, is trivial to compute when `A` is small and discrete: just evaluate `Q(s,a)` for every possible `a` and take the max. Robot manipulation actions are **continuous** (a gripper's Cartesian velocity, rotation, open/close command) — there is no way to enumerate "every possible action" to find the max. **QT-Opt** is the lecture's example of a genuine architecture built to solve exactly this problem, so it earns full architecture treatment even though it is not one of this week's explicitly linked readings (its underlying paper is fetched directly here for accuracy, since the professor's attention points name it explicitly).

*Citation: Dmitry Kalashnikov et al., "QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation," CoRL 2018.*

**Structure.** The network takes as input an observation `(I_t, gripper-aperture, gripper-height)` — `I_t` a 472×472 RGB image crop from an over-the-shoulder camera — and an action `a = (translation, rotation, open/close, termination)`. The image passes through 7 convolutional/max-pool layers; in parallel, the action (plus gripper state) passes through two fully-connected layers and is **broadcast-added** into the image branch's feature map (tiled spatially so the same action vector is added at every spatial location); the fused representation then passes through 9 more convolutional layers, two fully-connected layers, and a final sigmoid, producing a **single scalar Q-value in `[0,1]`** for that specific `(image, action)` pair.

**Input/output/label/loss.** Input: `(image, action)`. Output: one scalar `Q(s,a) ∈ [0,1]`. Label/target: the standard Bellman backup `r + γ · max_{a'} Q(s',a')` (§3.1) — reward 1 at episode end if an object was successfully lifted, 0 otherwise, with a small per-step penalty. Loss: ordinary squared temporal-difference error between the network's current output and this target.

**The innovation: solving `max_{a'}` with the cross-entropy method.** Since there is no vector of per-discrete-action outputs to take a max over, QT-Opt instead treats `argmax_{a'} Q(s',a')` as a small continuous *search* problem, solved at both training time (computing Bellman targets) and inference time (choosing an action to execute) via the **[cross-entropy method](https://en.wikipedia.org/wiki/Cross-entropy_method)** (CEM) — the same optimizer already introduced for trajectory optimization in Week 2 §6, here repurposed to search over a single action rather than a whole action sequence. The procedure: sample a batch of 64 candidate actions, evaluate `Q(s,a)` for each, fit a Gaussian to the best 6 of them, then resample the next batch of 64 from that Gaussian — repeated for 2 iterations. This is precisely CEM's "sample, evaluate, refit-to-the-elites, resample" loop from Week 2 §6, just scoring actions with a learned Q-network instead of a known cost function.

**What's genuinely innovative, contrasted with prior architectures.** An ordinary discrete-action Q-network (e.g. classic DQN) outputs one Q-value *per action* in a single forward pass and takes an exact max — impossible here because the action space is continuous. A standard actor-critic architecture instead trains a *second* network (the actor) whose entire job is to amortize this argmax by directly outputting the best action. QT-Opt needs no actor network at all: CEM performs the search freshly every time, at the cost of extra forward passes through the Q-network rather than the cost of training and maintaining a second network. In this sense QT-Opt's Q-network plays a role structurally similar to Week 1's energy-based-model policies (§7 of `week1-study-notes.md`) — a single scalar "compatibility/goodness" function over `(state, action)` pairs that must be *searched* rather than directly inverted to produce an action — as opposed to a regression head that outputs an action in one forward pass.

**Distributed training and results.** Data was collected from **7 real KUKA robot arms** running nearly continuously over ~4 months (~800 robot-hours), accumulating 580,000 off-policy plus 28,000 on-policy grasp episodes; training used a distributed replay buffer, roughly 1,000 parallel "Bellman updater" jobs computing CEM-based targets, and 10 training workers across 10 GPUs, needing up to 15 million gradient steps. The resulting policy achieved a **96% grasp success rate**, compared to 78% for a prior comparable-scale method trained on a similarly sized dataset; using only the off-policy portion of the data (no on-policy fine-tuning at all) still reached 87%.

---

## 4. Offline RL vs. behavior cloning

### 4.1 Precise definition

**Offline (batch) reinforcement learning**: given a single fixed dataset `D = {(s_i, a_i, r_i, s_i')}` generated once by an unknown behavior policy `π_β`, learn a policy `π` maximizing `J(π)`, using *only* `D` — with zero further interaction with the environment during training (§1.3's batch/online dichotomy, now made precise). This is strictly harder than ordinary off-policy RL (§2.3), which typically still allows *some* fresh interaction; offline RL allows none at all.

### 4.2 Why this isn't just behavioral cloning

BC (Week 1) only ever sees `(s, a)` pairs with no reward, and is trained by pure supervised regression to reproduce the demonstrator's actions. Its performance is fundamentally bounded by the demonstrator's own performance — a BC policy can, at best, match the demonstration data's average quality, never systematically exceed it, since nothing in its training signal ever says "this action was good" versus "this action merely happened."

Offline RL's dataset carries a **reward** at every transition, and its Bellman-backup machinery (§3) can, in principle, do something BC structurally cannot: **trajectory stitching** — combining fragments of several separately-suboptimal trajectories into a policy better than *any single trajectory* in the dataset. If one logged trajectory shows how to get from A to B, and a completely different logged trajectory (from a different, unrelated episode) shows how to get from B to C, a Bellman backup can propagate value backward from C through the second trajectory into state B, and from there into the first trajectory — yielding a policy that goes A→B→C even though no single demonstration ever did. BC has no analogous mechanism: it can only reproduce a trajectory shape it was actually shown end to end.

### Reading write-up: NeurIPS 2020 Offline RL Tutorial

*Citation: Sergey Levine, Aviral Kumar, "Offline Reinforcement Learning: From Algorithms to Practical Challenges," NeurIPS 2020 Tutorial.*

**What this reading covers.** A slide/video tutorial (with an accompanying interactive Colab exercise) motivated by the tension that "RL requires a fundamentally online learning paradigm," and framing offline RL as the attempt to remove that requirement, turning static logged datasets into "powerful decision-making engines." It surveys batch/offline RL methods, the deep-RL-specific challenges of the offline setting, theoretical foundations, and applications (robotics, autonomous driving, healthcare).

**How it connects.** This tutorial is presented by the same two lead authors, in the same year, as the companion survey immediately below, and the two should be read as a matched slide/paper pair covering the same material at different levels of formality.

### Reading write-up: "Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems"

*Citation: Sergey Levine, Aviral Kumar, George Tucker, Justin Fu, arXiv:2005.01643, 2020.*

**What this reading covers.** Formalizes the offline RL problem exactly as in §4.1 above, over the MDP notation of §1.1, with the offline dataset's states/actions drawn from `π_β`'s stationary visitation distribution `d^{π_β}(s)` (§1.2). It organizes the space of distribution-shift-mitigation strategies into three families that map directly onto this week's structure: **(a) importance-sampling / off-policy-evaluation methods** (correcting for the mismatch between `π_β` and `π_θ` via density-ratio reweighting — not covered further in this week's notes), **(b) dynamic-programming methods**, itself split into **policy-constraint methods** (§6 below) and **conservative/pessimistic value estimation** (§7 below), and **(c) model-based offline RL** (using conservative planning atop a learned dynamics model — outside this week's scope). Its open-problems discussion highlights the lack of known "dataset sufficiency" conditions guaranteeing any offline algorithm recovers a near-optimal policy, and the general gap between theory (mostly tabular/linear) and practice (deep function approximation) — a gap §7.4's critique papers make concrete.

**How it connects.** This survey's own taxonomy is exactly the (a)/(b)/(c) split this week's notes follow (minus model-based methods, out of scope), making it the organizing reference for §6–§7 below.

### Paper Breakdown: When Should We Prefer Offline RL Over Behavioral Cloning?

*Citation: Aviral Kumar, Joey Hong, Anikait Singh, Sergey Levine, "When Should We Prefer Offline Reinforcement Learning Over Behavioral Cloning?," ICLR 2022.*

**Problem/motivation.** Prior empirical results directly conflicted: some studies showed offline RL considerably beating BC (especially on tasks needing trajectory stitching, §4.2); others showed BC or filtered-BC beating offline RL on expert or near-expert data. No prior work had rigorously characterized *when* to expect one to beat the other.

**Key idea/innovation.** Formalize "how much distribution shift exists between the dataset and the optimal policy" via a **concentrability coefficient `C*`** — the smallest constant such that the optimal policy's state-action visitation never exceeds `C*` times the data distribution's own visitation, everywhere. `C* = 1` means the data exactly covers the optimal policy's own visitation (i.e., the data already *is* expert data with no extra coverage); larger `C*` means more of a mismatch.

**Method & training procedure.** Three regimes are analyzed:
- **`C* = 1`** (pure expert data, no distribution shift): a matching information-theoretic lower bound is proved showing *no* learner — offline RL or BC — can do better than the other here. With truly expert, unshifted data, offline RL has no structural advantage.
- **Data with "critical states"** (still near-expert, but the environment has a small fraction of states where only a narrow set of actions succeeds, versus most states having many "good-enough" actions): offline RL can beat BC even on expert data, because BC must imitate everywhere with equal precision (including at non-critical states, where doing so is wasted effort), whereas reward-driven offline RL can quickly identify "good enough" at non-critical states and concentrate its learning capacity on the states that actually matter.
- **Noisy data with coverage**: if the (suboptimal) data distribution still puts non-negligible probability mass near the optimal trajectory, offline RL attains `Õ(√H)` suboptimality (`H` = horizon) versus BC's `Õ(H)` — a gap that grows with task length. This directly motivates a practical recipe: collecting an equal amount of *noisy* data (not necessarily expert) and running offline RL on it can beat cloning an equal-sized *expert* dataset outright, especially on long-horizon tasks.

**Results.** On simulated robot-manipulation tasks (state-based), holding dataset size fixed, offline RL (a tuned CQL, §7.2 below) trained on **noisy-expert** data substantially beat BC trained on **expert** data of the same size: 85.7% vs. 14.5% success on one manipulation task, 90.3% vs. 17.4% on a second, 92.4% vs. 33.2% on a third — with the performance gap growing as task horizon increased, exactly matching the `Õ(√H)` vs. `Õ(H)` theoretical prediction. On pure expert data with no distribution shift, naive offline RL was often comparable to or worse than BC, confirming the `C*=1` no-advantage result — offline RL's advantage only appeared once *tuned* (offline-selected checkpoints, regularization) and once the data had either critical-state structure or noisy coverage.

**How it connects.** This paper is the direct empirical resolution of §0's open question ("can offline RL ever beat the data that generated it?") — yes, but only under specific, now-precisely-characterized conditions, and only with the conservatism/constraint machinery of §6–§7 properly tuned.

---

## 5. The central obstacle: distribution shift

### 5.1 Why offline Q-learning breaks

Section 3.1's Bellman backup, `Q(s,a) ← r + γ max_{a'} Q(s',a')`, requires evaluating `Q` at `(s', a')` for whatever `a'` currently maximizes it — but nothing constrains that `a'` to be an action the dataset ever demonstrated from `s'`. Online, this is self-correcting: if `Q` overestimates some untried action, the agent eventually tries it, observes a disappointing real return, and the estimate gets corrected. **Offline, there is no such correction step** — the dataset is fixed, the "mistaken" `(s', a')` pair is never actually visited, and the erroneously high Q-value simply gets bootstrapped forward into every other value estimate that depends on it, compounding through repeated backups (exactly the recursive structure of Fitted Q-Iteration in §3.2). The policy that results, `argmax_a Q(s,a)`, then systematically chases these overestimated, never-validated actions.

### Paper Breakdown: Batch-Constrained deep Q-learning (BCQ)

*Citation: Scott Fujimoto, David Meger, Doina Precup, "Off-Policy Deep Reinforcement Learning without Exploration," ICML 2019.*

**Problem/motivation.** Formalizes exactly the failure mode of §5.1 as **extrapolation error**: "a phenomenon in which unseen state-action pairs are erroneously estimated to have unrealistic values," arising from three distinct sources — **Absent Data** (a state-action pair is simply missing or underrepresented in the batch, so nothing constrains its estimated value), **Model Bias** (sampling transitions only from a finite fixed batch gives a biased estimate of the true transition dynamics), and **Training Mismatch** (the Bellman/regression loss weights updates by the *batch's* visitation frequency rather than the *current policy's*, biasing the learned Q-function away from accuracy on exactly the actions the (possibly-extrapolating) policy would actually pick).

**Key idea/innovation.** **Batch-constrained reinforcement learning**: rather than letting the policy greedily maximize `Q` over the *entire* action space (risking selection of an out-of-batch action whose value is an extrapolation artifact), restrict the policy's candidate actions to ones similar to what the batch actually contains — pushing the agent to behave close to on-policy with respect to the batch, while still optimizing Q-values among that restricted set.

**Method & training procedure.** A conditional [VAE](https://en.wikipedia.org/wiki/Variational_autoencoder) `G_ω` is trained to model the batch's own state-conditional action distribution (standard VAE reconstruction-plus-KL loss), so sampling from it produces only plausible, previously-seen-style actions. A small **perturbation model** `ξ_φ(s,a,Φ)`, constrained to output an adjustment in `[-Φ,Φ]`, is trained (via deterministic policy gradient) to nudge a VAE-sampled action toward a higher `Q`-value without straying far from it. At action-selection time, `n` candidate actions are sampled from the VAE, each perturbed, and the perturbed action maximizing `Q` is chosen:

```
π(s) = argmax_{a_i + ξ_φ(s,a_i,Φ)}  Q_θ(s, a_i + ξ_φ(s,a_i,Φ)),    {a_i ~ G_ω(s)}
```

Bellman targets use a clipped-double-Q-style combination of two Q-networks, weighting their minimum more heavily than their maximum (a tunable coefficient controlling how strongly future uncertainty is penalized) — reducing to standard clipped double Q-learning at one extreme of that weighting.

**Results.** Evaluated on MuJoCo continuous-control tasks (Hopper, HalfCheetah, Walker2d) under three named batch-construction settings: **"final buffer"** (all transitions from one noisy DDPG run, for broad coverage), **"concurrent"** (a batch identical to what a live DDPG agent would see, trained alongside it), and **"imitation"** (a batch of pure expert transitions) — plus an **"imperfect demonstrations"** variant testing robustness to noisy/suboptimal data. BCQ was the only tested algorithm to match or outperform the behavior policy across every setting; standard off-policy methods (DDPG, DQN) diverged and performed dramatically worse than the behavior policy even in the "concurrent" setting, where the data distribution exactly matched what a live agent would have seen — demonstrating the failure is intrinsic to offline bootstrapping, not merely a symptom of narrow data.

**How it connects.** BCQ is the founding paper of the **policy-constraint** family developed in §6 below — its "restrict the action space to stay near the batch" idea is the shared mechanism every method in that section refines.

---

## 6. Solution family 1 — Policy constraints

### 6.1 The shared mechanism: constrain the policy to the behavior policy

Every method in this family attacks §5.1's problem the same general way: prevent the learned policy `π_θ` from ever proposing an action far from what the behavior policy `π_β` would have proposed, so the Bellman backup's `max_{a'}` (§3.1) is never evaluated on a genuinely out-of-distribution action in the first place. BCQ (§5.2) is the founding instance of this idea, restricting `π_θ`'s candidate actions to VAE-sampled, behavior-like ones.

A subtlety the lecture flags explicitly: if the constraint is phrased as "make `π_θ` close to `π_β`" via a **[KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)**, the *direction* of that KL term matters, because KL divergence is asymmetric. Minimizing the **forward KL**, `KL(π_β ‖ π_θ)`, forces `π_θ` to place probability mass everywhere `π_β` does — including, if `π_β` is multimodal (e.g. a human teleoperator who sometimes goes left around an obstacle and sometimes right), the low-density region *between* those modes, since averaging over both modes is exactly what minimizes this direction of KL. This is "mass-covering." Minimizing the **reverse KL**, `KL(π_θ ‖ π_β)`, instead only penalizes `π_θ` for placing mass where `π_β` has none — it can safely collapse onto a *single* mode of `π_β` and pay no penalty, so long as it never invents unsupported actions. This is "mode-seeking." (This exact multimodality trade-off is why Week 1 §7's energy-based and diffusion policies exist — the same tension between covering all of a multimodal expert distribution versus safely picking one mode recurs there in a different guise.) BEAR (§6.2) sidesteps this asymmetry question entirely by constraining only the *support* of `π_θ`, not matching either KL direction.

### Paper Breakdown: BEAR (support matching via MMD)

*Citation: Aviral Kumar, Justin Fu, George Tucker, Sergey Levine, "Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction," NeurIPS 2019.*

**Problem/motivation.** Names the failure mode of §5.1 **bootstrapping error**, and formalizes how it accumulates: with `ζ_k(s,a) = |Q_k(s,a) − Q*(s,a)|` the true error at iteration `k` and `δ_k(s,a) = |Q_k(s,a) − 𝒯Q_{k-1}(s,a)|` the fresh regression error introduced at this step, `ζ_k(s,a) ≤ δ_k(s,a) + γ · max_{a'} E_{s'}[ζ_{k-1}(s',a')]` — errors on out-of-distribution actions are never directly minimized during training, yet still propagate forward, discounted, through every future backup.

**Key idea/innovation.** BCQ's action-distribution matching is, in the authors' framing, "similar to behavioral cloning" and overly restrictive. BEAR instead constrains only the **support** of `π_θ` — requiring `π_θ(a|s) = 0` wherever `π_β(a|s)` is negligible, while leaving the *relative probabilities within that support* completely free. A policy can therefore look nothing like `π_β`'s exact shape and still satisfy the constraint, so long as it never assigns probability to actions `π_β` would essentially never take.

**Method & training procedure.** The support constraint is enforced via a sample-only, kernel-based two-sample test called **[Maximum Mean Discrepancy](https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions)** (MMD), introduced here from scratch since it has no prior appearance in these notes: given samples `x_1,...,x_n ~ P` and `y_1,...,y_m ~ Q`, and a kernel `k(·,·)` measuring pointwise similarity,

```
MMD²({x_i}, {y_j}) = (1/n²)Σ_{i,i'} k(x_i,x_i') − (2/nm)Σ_{i,j} k(x_i,y_j) + (1/m²)Σ_{j,j'} k(y_j,y_j')
```

— average within-`P` similarity, minus twice the average cross `P`–`Q` similarity, plus average within-`Q` similarity. Intuitively: if every sample from `P` is, on average, just as similar to other `P` samples as it is to `Q` samples (and vice versa), the two distributions are indistinguishable by this test; for a suitable kernel, `MMD = 0` exactly when `P = Q`. BEAR's policy-improvement step maximizes Q-values over `π`, subject to `E_{s~D}[MMD(D(s), π(·|s))] ≤ ε` — the dataset's own action samples at each state versus the policy's sampled actions — solved via dual gradient descent on a Lagrange multiplier. The dataset side of this comparison, since typically only one action per state exists in `D`, is approximated by fitting an auxiliary tanh-Gaussian behavior-policy model and sampling from it. Bellman targets use the same kind of min/max convex combination across an ensemble of Q-functions that BCQ used, again trading off how heavily uncertainty at future states is penalized.

**Results.** No numeric result tables appear in the paper (all results are learning curves) — but the qualitative pattern is a clean and direct contrast with BCQ: on medium-quality (partially-trained-policy) data, BEAR consistently outperforms both BCQ and naive off-policy RL by large margins, and the authors note "the performance of BCQ often tracks the performance of the BC baseline, suggesting that BCQ primarily imitates the data." On random and on near-optimal expert data specifically, BEAR is reported as the only method of the two that succeeds at both extremes — naive RL fails on optimal data (it never sees mistakes to learn from) and BCQ's tighter distribution-matching constraint fails on random data (it forces near-imitation of a bad policy).

**How it connects.** BEAR is a direct refinement of BCQ's shared mechanism (§6.1): same underlying goal (stay near `π_β`), a fundamentally different and less restrictive way to formalize "near."

### Paper Breakdown: TD3+BC — a minimalist policy constraint

*Citation: Scott Fujimoto, Shixiang Shane Gu, "A Minimalist Approach to Offline Reinforcement Learning," NeurIPS 2021.*

**Problem/motivation.** Argues that offline-RL methods (the paper's own complexity ledger names CQL, §7.2 below, and Fisher-BRC specifically) accumulate substantial extra machinery on top of an ordinary off-policy algorithm — added regularizers, generative models, extra hyperparameters, removed baseline components — creating real implementation and tuning burden.

**Key idea/innovation.** Add a single behavior-cloning regularization term directly to [TD3](https://en.wikipedia.org/wiki/Model-free_(reinforcement_learning))'s ordinary deterministic policy-gradient objective — nothing else about TD3 changes.

**Method & training procedure.** Where vanilla TD3 maximizes `E[Q(s,π(s))]`, TD3+BC maximizes

```
E_{(s,a)~D} [ λ · Q(s, π(s)) − (π(s) − a)² ]
```

— an ordinary squared-error BC term added directly alongside the Q-maximization term, gently pulling the policy's output toward the batch's own recorded action `a` at that state (this is exactly the mechanism §6.1 calls "constraining `π_θ` toward `π_β`," implemented in the simplest possible way: a direct regression penalty rather than a distributional or support-based constraint). Two small extra pieces: state features are normalized (`s_i ← (s_i − μ_i)/(σ_i + ε)`, `ε = 10⁻³`), and the trade-off weight `λ` is itself normalized by the minibatch's mean absolute Q-value, `λ = α / ((1/N)Σ|Q(s_i,a_i)|)` with `α = 2.5` fixed across all experiments — keeping the BC and Q terms on comparable numeric scales without per-task tuning.

**Results.** On the D4RL (§8 below) MuJoCo continuous-control suite, TD3+BC's total normalized score across all tested tasks (979.3) matched the considerably more complex CQL (764.3) and Fisher-BRC (974.6), while training in **39 minutes** versus CQL's **4 hours 11 minutes** on identical hardware — the paper's headline point being that this near-total simplicity costs essentially nothing in final performance.

**How it connects.** TD3+BC belongs to the policy-constraint family by mechanism (§6.1) even though its own paper frames its contribution as a complexity contrast against the *conservatism* family (§7) rather than against BCQ/BEAR directly — a reminder that the two solution families' boundary is about mechanism, not about which papers a given paper happens to benchmark against.

### Paper Breakdown: Benchmarking Batch Deep RL Algorithms

*Citation: Scott Fujimoto, Edoardo Conti, Mohammad Ghavamzadeh, Joelle Pineau, "Benchmarking Batch Deep Reinforcement Learning Algorithms," Deep RL Workshop, NeurIPS 2019.*

**Problem/motivation.** Prior batch-RL results used inconsistent environments and data distributions, making claims across papers hard to compare directly — this paper re-runs several methods under one unified, controlled protocol to clarify when and why batch RL fails.

**Key idea/innovation.** Rather than reusing BCQ's original continuous-control "final buffer/concurrent/imitation" settings (§5.2), this paper works entirely in the **discrete-action Atari** domain: a single DQN behavioral policy is trained online for 10 million timesteps, and *every* transition it ever experienced — mixing high-exploration and near-greedy episodes together — becomes one fixed 10-million-transition batch, reused identically across every compared algorithm.

**Method & training procedure.** A **discrete-action adaptation of BCQ** is introduced (simplifying BCQ's continuous VAE-plus-perturbation machinery to a thresholded argmax over a generative action model, since discrete action spaces don't need a perturbation model at all), compared against **KL-Control** (a KL-regularized policy method), **QR-DQN** (a distributional value method), **REM** (§7.3 below — its full mechanism explained there), and online DQN as a reference point.

**Results.** Standard off-policy algorithms performed poorly in this fixed-batch setting, several underperforming even the noisy behavioral policy that generated the data; the discrete BCQ adaptation outperformed every other method in every tested game (though typically only matching, not exceeding, online DQN's performance); QR-DQN was the strongest non-BCQ baseline. Performance drops were tied directly to divergence in the algorithms' own value estimates — the same extrapolation-error mechanism BCQ's original paper identified (§5.2), now confirmed in a discrete, image-based domain.

**How it connects.** This is the policy-constraint family's empirical bridge into §7: REM, one of value conservatism's central methods, is evaluated here too, on exactly the same Atari batch as the paper's own discrete-BCQ baseline — direct evidence that both solution families are being tested under comparable conditions, even though their underlying mechanisms (§6.1's constraint vs. §7.1's pessimism) are unrelated.

---

## 7. Solution family 2 — Value conservatism

### 7.1 Pessimism as the alternative lever

Rather than constraining *which actions the policy is allowed to propose* (§6), a second family of fixes changes what gets learned instead: make the Q-function itself **pessimistic** — deliberately biased *low* on actions the data doesn't support — so that a policy optimizer downstream naturally avoids them without any explicit constraint on the policy's distribution or support at all.

### Paper Breakdown: Conservative Q-Learning (CQL)

*Citation: Aviral Kumar, Aurick Zhou, George Tucker, Sergey Levine, "Conservative Q-Learning for Offline Reinforcement Learning," NeurIPS 2020.*

**Problem/motivation.** The same overestimation-on-out-of-distribution-actions failure diagnosed in §5.1, but addressed by directly re-shaping the Q-function's training objective rather than restricting the policy.

**Key idea/innovation.** Add a term to the ordinary Bellman-error loss that *pushes down* Q-values on actions sampled from a distribution seeking out high-Q (and therefore likely out-of-distribution) actions, while simultaneously pushing *up* Q-values on the actions actually observed in the data — driving the learned Q-function toward being a provable **lower bound** on the true Q-function, rather than trying to keep the policy itself close to `π_β`.

**Method & training procedure — the CQL objectives.** The general form, **CQL(ρ)**, is a min-max problem:

```
min_Q max_μ   α · ( E_{s~D, a~μ(a|s)}[Q(s,a)] − E_{s~D, a~π̂_β(a|s)}[Q(s,a)] )
            + (1/2) E_{s,a,s'~D}[ (Q(s,a) − Bellman-backup target)² ]
```

— `μ(a|s)` is an adversarially-chosen action distribution being maximized over (it seeks out actions where pushing Q down matters most), `π̂_β` is the empirical behavior policy estimated from the dataset, and `α` weights the conservative penalty against the ordinary Bellman-error term. Choosing `μ`'s regularizer to be a maximum-entropy penalty gives the practical variant used in experiments, **CQL(H)**, where the inner adversarial maximization has a closed form (`μ(a|s) ∝ exp(Q(s,a))`), collapsing the min-max into a single minimization with a **log-sum-exp** term standing in for "the max over the whole action space":

```
min_Q   α · E_{s~D}[ log Σ_a exp(Q(s,a)) − E_{a~π̂_β(a|s)}[Q(s,a)] ]
      + (1/2) E_{s,a,s'~D}[ (Q(s,a) − Bellman-backup target)² ]
```

The first term is the conservative penalty (push down a soft-max proxy over *all* actions; push up only the actually-observed ones); the second is the ordinary squared TD error from §3.1. `α` is tuned automatically via Lagrangian dual gradient descent for continuous-control experiments, and held fixed for discrete (Atari) experiments. The paper proves that for `α` large enough, the learned Q-function is a **pointwise lower bound** on the true Q-function everywhere in the dataset's support, and — the practically important version — that the estimated *value* of the learned policy under this Q-function lower-bounds its *true* value, provided `α` exceeds a threshold that scales with reward/dynamics-error constants, the discount, and how much the learned policy's action distribution diverges from the behavior policy's.

**Results.** Evaluated on the D4RL suite (§8) across locomotion, AntMaze, Adroit, and Kitchen tasks, plus low-data (1%) Atari; CQL is reported to substantially outperform prior offline-RL baselines, with its largest margins on datasets that mix data from multiple policies of differing quality (the D4RL "-medium-expert"/"-mixed"-style datasets) — exactly the kind of heterogeneous data offline RL is meant to handle.

**How it connects.** CQL is this family's foundational method — REM (§7.3) and the critique papers immediately below engage directly with the conservatism idea it establishes, and Cal-QL (§11) is a direct extension of CQL's own objective.

### Paper Breakdown: REM — ensembling instead of an explicit penalty

*Citation: Rishabh Agarwal, Dale Schuurmans, Mohammad Norouzi, "An Optimistic Perspective on Offline Reinforcement Learning," ICML 2020.*

**Problem/motivation.** An empirical challenge to the premise that off-policy deep RL algorithms necessarily perform poorly offline: trained purely on a large, fixed, logged dataset with no further interaction, do standard off-policy algorithms actually fail, or can they work well given enough data diversity?

**Key idea/innovation.** Train an ensemble of `K` Q-value "heads," but instead of training each head against its own fixed target (ordinary ensembling) or using a single fixed average of the ensemble, draw a fresh **random convex-combination weight vector** over the `K` heads at every training step, and require the Bellman equation to hold for *that* randomly-weighted mixture. Note this mechanism is **not** the same as CQL's log-sum-exp penalty: REM contains no explicit push-down term on any particular action at all — its robustness comes entirely from forcing Bellman-consistency across a randomly varying family of ensemble combinations, an implicit regularization/ensembling effect, not an explicit conservative penalty.

**Method & training procedure.** Given `K` Q-heads `Q^1_θ,...,Q^K_θ`, a random mixture weight `α` is drawn each step by sampling `K` values i.i.d. from `Uniform(0,1)` and normalizing them to sum to 1 (landing `α` on the probability simplex). The training loss enforces the Bellman equation for the resulting mixture:

```
Δ^α_θ(s,a,r,s') = Σ_k α_k Q^k_θ(s,a) − r − γ · max_{a'} Σ_k α_k Q'^k_{θ'}(s',a')
L(θ) = E_{(s,a,r,s')~D} [ E_{α} [ Huber-loss( Δ^α_θ(s,a,r,s') ) ] ]
```

The comparison baseline the paper itself uses to isolate this mechanism's effect is **Ensemble-DQN**: each head trained against its *own* target (no random mixing), with the plain average used only at evaluation time.

**Results.** On the Atari DQN Replay Dataset (roughly 50 million logged transitions per game from a fully-trained-online DQN agent, across 60 games), offline REM reached a median normalized score of 123.8% (beating the fully-trained online DQN reference on 49 of 60 games), ahead of offline QR-DQN (118.9%, 45 games) and offline Ensemble-DQN (111.0%, 39 games) — direct evidence that, given enough data diversity, off-policy value methods trained purely offline can exceed the very online agent that generated their training data.

**How it connects.** REM is deliberately kept adjacent to, rather than merged into, CQL's shared-mechanism treatment above: both belong to the broad "value conservatism/robustness" family in spirit, but REM's ensembling-based robustness and CQL's explicit penalty-based pessimism are mechanically distinct ideas that happen to target the same failure mode from different directions. §6.4 above shows REM evaluated directly alongside a policy-constraint method (discrete BCQ) on the same Atari batch, underscoring that "which family a method belongs to" is about mechanism, not about which benchmark it appears in.

### A critical perspective on both families

Two papers push back on assumptions the rest of this section relies on:

### Paper Breakdown: Why Should I Trust You, Bellman?

*Citation: Scott Fujimoto, David Meger, Doina Precup, Ofir Nachum, Shixiang Shane Gu, "Why Should I Trust You, Bellman? The Bellman Error Is a Poor Replacement for Value Error," ICML 2022.*

**Problem/motivation.** Nearly every method in this week's notes is trained by minimizing some form of Bellman error, on the implicit assumption that lower Bellman error means a more accurate value function. This paper asks directly: is that assumption actually true?

**Key idea/innovation.** No. Bellman error and true value error can diverge for two distinct reasons: **cancellation** — since Bellman error is a *difference* of value errors on consecutive state-action pairs, correlated errors partially cancel, producing deceptively low Bellman error even when the underlying value function is very wrong — and **non-uniqueness under finite data**: with a finite dataset, the Bellman equation can be satisfied *exactly* by infinitely many wrong value functions, because the "missing" successor transitions the equation would otherwise need to constrain simply aren't in the data.

**Method & training procedure.** A toy two-state MDP makes both failure modes concrete: a transition `(s_0, a) → s_1`, zero reward everywhere (so the true value of everything is 0), with `(s_1, a)` never appearing in the dataset. Setting `Q(s_0,a) = C`, `Q(s_1,a) = C/γ` gives **zero Bellman error** on the one available transition, yet **value error `C`** — arbitrarily large, and entirely invisible to a Bellman-error-based loss. The paper further proves a tight multiplicative bound relating average Bellman error to average value error, `C_avg/(1+γ) ≤ E[|value error|] ≤ C_avg/(1−γ)`: for `γ = 0.99`, an average Bellman error of exactly 1 is consistent with an average value error anywhere from about 0.5 to 100 — a hundred-fold range of genuine uncertainty hiding behind one Bellman-error number.

**Results.** On MuJoCo policy-evaluation experiments comparing Bellman Residual Minimization (BRM, which directly minimizes Bellman error) against Fitted Q-Evaluation (FQE, an iterative Bellman-backup method), FQE consistently achieved *lower* value error than BRM while having *dramatically higher* Bellman error — in one off-policy HalfCheetah setting, FQE's Bellman error was over a thousand times BRM's, while FQE's value error was still the lower of the two. The paper concludes this isn't because FQE is a better loss metric, but because it's a better *optimization process* (iterative bootstrapping under a generalization condition), independent of what its raw loss value says.

**How it connects.** This is a direct caution against reading D4RL/offline-RL training curves at face value: low training loss, on its own, is not evidence of an accurate value function for *any* method in §6 or §7 — exactly the motivation for §8's standardized, held-out evaluation protocol.

### Paper Breakdown: Instabilities of Offline RL with Pre-Trained Neural Representation

*Citation: Ruosong Wang, Yifan Wu, Ruslan Salakhutdinov, Sham M. Kakade, "Instabilities of Offline RL with Pre-Trained Neural Representation," ICML 2021.*

**Problem/motivation.** Modern pre-trained neural representations are extremely effective in supervised and transfer-learning settings. This paper asks whether that effectiveness carries over to offline RL: is a good, even near-oracle, pre-trained representation enough to make offline value estimation stable?

**Key idea/innovation.** No — even a representation that exactly realizes the target policy's true Q-function (the strongest possible notion of "good representation") is not sufficient. What matters is a much stronger condition — **policy completeness** (whether the feature space is rich enough that every Bellman-backup target is itself representable) plus **low distribution shift** between the data distribution's feature covariance and the one-step-lookahead feature covariance — and typical pre-trained (or random-feature) representations generally fail to satisfy it.

**Method & training procedure.** Using linear function approximation on top of a fixed feature map, the paper shows Fitted Q-Iteration's error obeys `θ_T − θ* ≈ Σ_t (γL)^{t-1}(...)`, where `L` involves the ratio of the data's feature covariance to the one-step-lookahead feature covariance — if this operator `L` is not appropriately contractive, estimation error **grows exponentially** with the number of FQI iterations, regardless of how good the representation otherwise is. A synthetic experiment (100-dimensional features drawn i.i.d. from a standard Gaussian, by construction perfectly realizing the target Q-function) shows exactly this exponential blow-up in practice — even though the feature covariance itself is well-conditioned, ruling out "bad/collinear features" as the explanation.

**Results.** On real Gym tasks (MountainCar, CartPole, Ant, HalfCheetah, Hopper, Walker2d), using the last hidden layer of an online-trained DQN/TD3 network as the feature representation — a genuinely task-relevant, non-arbitrary representation — Fitted Q-Iteration and LSTD value estimates degraded sharply as even a modest amount of additional random-trajectory data was mixed into an otherwise-clean target-policy dataset, and in most environment/policy-comparison pairs tested, FQI could not reliably distinguish the target policy's value from a substantially worse policy's value at all.

**How it connects.** This is CQL's and REM's cautionary counterpart from the representation-learning side: §7.2–§7.3's methods assume the underlying Q-network's representation is adequate; this paper shows that assumption can silently fail even when the representation looks good by every ordinary standard, independent of which conservatism or constraint mechanism sits on top of it.

---

## 8. Standardized evaluation: D4RL

Given §7.4's warning that training losses can't be trusted at face value, offline RL specifically needs *held-out*, standardized benchmarks — not just a training curve — to know whether a method actually works.

### Paper Breakdown: D4RL

*Citation: Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine, "D4RL: Datasets for Deep Data-Driven Reinforcement Learning," arXiv:2004.07219, 2020.*

**Problem/motivation.** Most prior offline-RL evaluations simply rolled out a partially-trained online policy (e.g. an early-stopped SAC agent) to generate a dataset — narrow, single-modal data that doesn't reflect the heterogeneous, passively-logged, or human-generated data seen in real applications, and which can mask real algorithmic weaknesses.

**Key idea/innovation.** A benchmark suite of tasks and datasets deliberately engineered so each domain stresses a *specific* realistic property offline RL must handle, rather than reusing narrow online-RL-derived data.

**Method & training procedure.** Domains include **Gym-MuJoCo locomotion** (`random`/`medium`/`medium-replay`/`medium-expert` variants, testing narrow and mixed-quality data from deterministic partially-trained policies), **Maze2D** and **AntMaze** navigation (sparse-reward, undirected, explicitly designed to require "stitching" — combining segments of separate trajectories, exactly §4.2's mechanism), **Adroit** dexterous-hand manipulation (small human-demonstration datasets, `human`/`expert`/`cloned` variants, testing narrow high-dimensional data), **Franka Kitchen** (undirected multitask data), and **offline CARLA** driving (image-based, testing partial observability and non-neural hand-designed behavior policies).

**Results.** Across baselines including BC, SAC, BEAR, BRAC, BCQ, and CQL, the paper reports that many algorithms struggled precisely on the tasks designed to stress realistic properties — passively logged data, narrow distributions, limited human demonstrations — with CQL generally the strongest and most consistent performer among the tested baselines on the harder AntMaze and Kitchen tasks, and several baselines scoring near zero on the largest, sparsest AntMaze maze.

**How it connects.** D4RL is the shared measuring stick nearly every other paper in this week's notes (CQL, BCQ, BEAR-family comparisons, TD3+BC, IQ-Learn, Cal-QL) reports results on — a direct response to §4.2's stitching claim and §7.4's warning that training-time metrics alone can't be trusted.

---

## 9. Offline RL at robot-manipulation scale

The methods above are validated mostly on MuJoCo/Atari benchmarks. This section turns to real (or photorealistic simulated) robot manipulation, where the offline dataset is genuinely messy, multi-source, and often human-collected.

### Paper Breakdown: IRIS

*Citation: Ajay Mandlekar, Fabio Ramos, Byron Boots, Silvio Savarese, Li Fei-Fei, Animesh Garg, Dieter Fox, "IRIS: Implicit Reinforcement without Interaction at Scale for Learning Control from Offline Robot Manipulation Data," arXiv:1911.05321, 2019/2020.*

**Problem/motivation.** Large, crowdsourced demonstration datasets are diverse but often suboptimal and multimodal — conventional imitation learning degrades sharply on this kind of data, while RL faces a heavy exploration burden and reward-shaping cost that a purely offline setting can't easily supply.

**Key idea/innovation.** Factorize control into a **low-level, goal-conditioned imitation controller** (reproduces short demonstrated sub-sequences toward a fixed nearby goal) and a **high-level goal-selection mechanism** (proposes and value-scores candidate subgoals), so the system can selectively imitate only the good local segments of a large, suboptimal dataset, recombining fragments of different trajectories rather than imitating any one of them wholesale.

**Method & training procedure.** The low-level controller is a goal-conditioned RNN `π_θ(a|s, s_g)`, where `s_g` is a state observed `T` steps ahead in a training sub-sequence, trained by ordinary BC regression: `Σ_k ‖a_k − π_θ(s_k|s_g)‖²`. The high-level mechanism has two parts: a **conditional VAE** that learns to propose plausible future goal states `s_g` given the current state (trained on `(s_t, s_{t+T})` pairs from the data), and a **value function** `Q(s,a)` trained via a BCQ-style batch-constrained update (§5.2's mechanism, reused here rather than re-derived) to score how promising each proposed goal is. At test time: sample several candidate goals from the VAE, pick the one the value function scores highest, hand it to the low-level RNN as a fixed target for `T` steps, then repeat.

**Results.** On a real-robot-derived pick-and-place task (RoboTurk-collected, crowdsourced, intentionally suboptimal demonstrations), IRIS reached 81.3% success versus BC's 13.7%, BC-RNN's 16.7%, and BCQ's 18.0% — plain imitation and plain batch-constrained Q-learning both struggled badly on this diverse data, while IRIS's factorized approach reached success rates 4–6× higher.

**How it connects.** IRIS reuses BCQ's policy-constrained value function as a sub-component rather than treating goal-conditioned imitation and offline RL as competitors — a direct example of the two solution families (§6, and offline RL generally) composing rather than substituting for each other.

### Paper Breakdown: robomimic — what actually matters

*Citation: Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio Savarese, Yuke Zhu, Roberto Martín-Martín, "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation," CoRL 2021.*

**Problem/motivation.** The offline-learning-from-demonstration literature has no consistent datasets or reproducible protocols, making it hard to know which design choices actually drive performance rather than incidental implementation differences.

**Key idea/innovation.** A large, controlled empirical study — not a new algorithm — benchmarking six algorithms (BC, BC-RNN, HBC, BCQ, CQL, IRIS) across five simulated and three real-world manipulation tasks, with datasets of varying quality (a single proficient teleoperator, six teleoperators of mixed skill, and machine-generated mixtures of expert/suboptimal rollouts).

**Method & training procedure.** Beyond the algorithm comparison, the study isolates several design axes: policy-selection criteria (choosing a checkpoint by validation loss vs. actual task success), observation-space choices (raw low-dimensional state vs. images, with vs. without auxiliary velocity features), and image-augmentation choices (pixel-shift randomization, wrist-camera inclusion).

**Results.** Offline RL (BCQ, CQL) clearly **underperformed** a temporally-abstracted BC variant (BC-RNN) on human-collected demonstration data across nearly every task — e.g. on a harder assembly task, BC-RNN reached 84.0% versus BCQ's 50.0% and CQL's 5.3%, and the gap widened further on longer-horizon tasks. The paper's own conclusion: "neither BCQ nor CQL performs particularly well on these human-generated datasets," a direct call for improving offline RL's ability to learn from suboptimal *human* data specifically, since both algorithms had worked well on suboptimal *agent*-generated data in their original papers. Two further headline findings: choosing checkpoints by best-observed task success rather than validation loss mattered enormously (a 10–100% relative swing in success rate), and design choices tuned in simulation transferred directly to the real-robot tasks without retuning.

**How it connects.** This is the empirical counterpoint to §4.2's optimistic BC-vs-offline-RL framing: offline RL's theoretical advantages (§4.2's paper) depend on the kind of data conditions (critical states, noisy-but-covering data) that clean, expert-quality human teleoperation demonstrations often don't actually exhibit.

### Paper Breakdown: Reward sketching and batch RL at scale

*Citation: Serkan Cabi, Sergio Gómez Colmenarejo, Alexander Novikov, Ksenia Konyushkova, et al., "Scaling Data-Driven Robotics with Reward Sketching and Batch Reinforcement Learning," RSS 2020.*

**Problem/motivation.** Real robots cannot generate data faster than real time, and, unlike simulation, real-world tasks have no naturally-occurring reward signal — hand-engineering a reward function for every new task doesn't scale, and BC requires large, consistent, task-specific demonstration sets that can't reuse data collected for other tasks.

**Key idea/innovation.** Combine three pieces: cheap human **reward sketching** (a person retroactively draws a rough reward curve while scrubbing through a recorded video, rather than labeling every frame), automatic retroactive annotation of *all* stored historical robot experience (regardless of what task it was originally collected for) using the resulting learned reward model, and batch RL trained on this newly-relabeled, task-agnostic data pool.

**Method & training procedure.** A human watches a video episode and sketches a continuous progress curve `s(x_t) ∈ [0,1]` by scrubbing through it. A reward model is trained not by direct regression to this curve, but via an **intra-episode pairwise ranking loss**: frames the human rated as higher-progress must score higher than lower-progress frames from the *same* episode, plus an auxiliary loss anchoring absolute reward near the labeled success/failure regions. Once trained, this reward model retrospectively relabels the entire "NeverEnding Storage" (NES) — a persistent, task-agnostic archive of *all* robot experience ever collected, from teleoperation, scripted policies, and prior learned policies alike. A D4PG-style (distributional, recurrent, distributed actor-critic) batch-RL algorithm then trains offline on this relabeled data.

**Results.** On real Sawyer-arm manipulation tasks, the resulting policies reached 80% success on a basic lifting task and 60% on a stacking task under normal conditions, with graceful (though reduced) performance on harder variants and previously unseen objects; ablations confirmed that removing either the broader task-agnostic data pool or the distributional-RL component collapsed performance to near zero on the harder variants.

**How it connects.** This is offline RL applied to the *reward-labeling* problem itself, not just the policy-learning problem — complementary to every other method in §9, which all assume rewards are already present in the dataset.

### Paper Breakdown: COG — stitching skills via Bellman backups

*Citation: Avi Singh, Albert Yu, Jonathan Yang, Jesse Zhang, Aviral Kumar, Sergey Levine, "COG: Connecting New Skills to Past Experience with Offline Reinforcement Learning," CoRL 2020.*

**Problem/motivation.** A policy trained only on a small, task-specific demonstration set is narrow: e.g. a robot trained to grasp an object from an *open* drawer fails outright if the drawer starts *closed*, because that initial condition was never demonstrated.

**Key idea/innovation.** Directly illustrates §4.2's trajectory-stitching claim: combine a small, reward-labeled, task-specific demonstration set with a much larger task-agnostic prior dataset (unlabeled, reward set to zero), and let CQL-style offline RL's Bellman backups propagate value from the labeled task set backward into the unlabeled prior data — even when *no single trajectory* in either dataset shows the complete task.

**Method & training procedure.** Trained with Conservative Q-Learning (§7.2's exact mechanism, reused rather than re-derived) over the union of both datasets. The stitching argument, in the authors' own framing: "Q-learning propagates information backwards through a trajectory... state-action pairs at the end of a trajectory with a high reward are assigned higher values, and these values propagate to states further back in time" — applied across the combined dataset, a prior-dataset trajectory ending in a state that happens to also appear in a task-demo trajectory gets its value updated based on that demo's eventual reward, even though the two trajectory segments came from entirely different collection episodes. Plain imitation learning has no analogous mechanism, since BC only fits the action distribution conditioned on states actually present in its own (small) demonstration set, with no cross-trajectory value-propagation step at all.

**Results.** On a simulated grasp-from-drawer task, when the drawer started in an initial condition absent from the small task-specific demo set (closed, or blocked by another object), COG reached 68–78% success while BC, plain offline RL trained only on the task-specific set, and SAC all scored at or near 0%. On a real-robot version of the drawer task, COG succeeded in 7 of 8 trials from a closed-drawer start, while a BC baseline never succeeded at all.

**How it connects.** COG is the cleanest empirical demonstration in this week's notes of exactly what §4.2 argued in the abstract: trajectory stitching via Bellman backups genuinely extends effective policy coverage beyond what any single demonstrated trajectory shows.

### Paper Breakdown: Targeted Environment Design from Offline Data

*Citation: Izzeddin Gur, Ofir Nachum, Aleksandra Faust, "Targeted Environment Design from Offline Data," NeurIPS 2021 DeepRL Workshop.*

**Problem/motivation.** Simulators are cheap and safe to train in but rarely match a real target environment exactly (the sim-to-real gap); pure offline RL avoids that mismatch but generalizes poorly outside the exact support of its fixed dataset.

**Key idea/innovation.** Rather than training a policy directly from an offline dataset the way every other method in this section does, learn a **distribution over simulator parameters** that, when a (jointly learned) behavior policy is rolled out inside it, reproduces the offline dataset's own state-action distribution — then hand the resulting *calibrated simulator* to an ordinary online RL algorithm.

**Method & training procedure.** The simulator's parameters and an auxiliary behavior policy are optimized jointly so that simulated rollouts' induced state-action distribution matches the target offline dataset's distribution as closely as possible; a fully-trained RL agent is then produced by standard online training *inside* this calibrated simulator, not by any direct offline learning step.

**Results.** The paper reports learning effective simulator calibrations from only a handful of offline demonstrations, though this note deliberately omits specific benchmark numbers, since the primary text could not be independently verified against the offline dataset's OpenReview page for this pass — a gap to close before treating any specific figure from this paper as citable fact.

**How it connects.** This paper's *purpose* is categorically different from every other paper in §9: IRIS, robomimic, reward sketching, and COG all turn an offline dataset directly into a deployable policy; this paper instead turns an offline dataset into a *calibrated simulator*, deferring policy learning to a separate, standard online-RL step run afterward — a data-generation/simulator-calibration tool that sits upstream of policy learning, rather than an offline-imitation or offline-RL method in its own right.

---

## 10. Reusing the Q-function for imitation: IQ-Learn

### Paper Breakdown: IQ-Learn

*Citation: Divyansh Garg, Shuvam Chakraborty, Chris Cundy, Jiaming Song, Stefano Ermon, "IQ-Learn: Inverse Soft-Q Learning for Imitation," NeurIPS 2021.*

**Problem/motivation.** Prior adversarial inverse-reinforcement-learning-style imitation methods (e.g. GAIL) cast imitation as a nested min-max problem over a reward function *and* a policy trained against it — typically unstable, sensitive to hyperparameters, and hard to get right in practice.

**Key idea/innovation.** Because the optimal maximum-entropy policy is a simple closed-form function of a soft Q-function (`π*(a|s) ∝ exp(Q*(s,a))`), the nested min-max collapses into a **single concave maximization over one function — the Q-function itself** — with no separate reward network, no separate policy network, and no adversarial training loop.

**Method & training procedure.** The continuous-control training objective is `J(π,Q) = E_{ρ_E}[φ(Q(s,a) − γE_{s'}[V^π(s')])] − (1−γ)E_{ρ_0}[V^π(s_0)]`, alternated with a soft-actor-critic-style policy update. With a `χ²`-divergence choice of `φ`, the objective in the offline/no-reward-network simplification reduces to a form the authors note is identical to CQL's Q-learning objective with the reward set to zero — a direct structural link back to §7.2. Rewards can be recovered after training via `r(s,a,s') = Q(s,a) − γV^π(s')`.

**Results.** With a single expert demonstration on MuJoCo continuous-control tasks, IQ-Learn matched or exceeded expert-level return (e.g. 5227.1 on Humanoid vs. an expert score of 5312.8), outperforming GAIL, DAC, and ValueDICE on every tested task; on Atari with 20 expert demonstrations, it reached expert-level performance on several games roughly 3× faster (in environment steps) than Q-learning-based RL baselines, while GAIL and ValueDICE remained near random performance even after far more training.

**How it connects.** IQ-Learn sits at the intersection of this week's offline-RL machinery and Week 1's imitation-learning framing — its own objective reduces to CQL's, but its motivating problem (imitation from demonstrations, no offline reward given) is closer to Week 1's behavioral-cloning setting than to this week's reward-labeled offline-RL setting. It is flagged here as a forward pointer: the fuller inverse-reinforcement-learning and maximum-entropy framing this method builds on is developed properly in a later week's notes, not here.

---

## 11. From offline to online: Cal-QL

### Paper Breakdown: Cal-QL

*Citation: Mitsuhiko Nakamoto, Yuexiang Zhai, Anikait Singh, Max Sobol Mark, Yi Ma, Chelsea Finn, Aviral Kumar, Sergey Levine, "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning," NeurIPS 2023.*

**Problem/motivation.** A natural-seeming recipe — pretrain with CQL offline, then fine-tune online — often fails in practice: CQL's Q-values are, by design, deliberately smaller than the true return of any valid policy (§7.2's whole point). Once online fine-tuning starts, newly-collected actions — even objectively *worse* than the offline policy's — produce real returns that look larger than these deliberately deflated Q-estimates, so the policy optimizer abandons the good offline initialization to chase these spuriously "superior" actions, and performance craters until the Q-function's scale catches up with reality.

**Key idea/innovation.** A Q-function is **calibrated** with respect to some easily-estimated reference policy `μ` (typically the behavior policy that generated the offline data) if `E_{a~π}[Q(s,a)] ≥ V^μ(s)` for every state in the dataset — i.e. conservatism is allowed to push Q-values down, but never below a known, reliable floor. Modifying CQL's push-down term to respect this floor prevents exactly the online-fine-tuning collapse described above.

**Method & training procedure.** Cal-QL's regularizer clamps CQL's out-of-distribution push-down term: `E_{s~D,a~π}[max(Q_θ(s,a), V^μ(s))] − E_{s,a~D}[Q_θ(s,a)]` — a one-line change to CQL's objective (§7.2) that masks the conservative penalty whenever the Q-function is already at or below the reference floor `V^μ(s)`, estimated in practice by simple Monte-Carlo return-to-go regression on the offline dataset.

**Results.** Across 11 tasks (D4RL AntMaze, FrankaKitchen, Adroit binary-reward tasks, and a vision-based pick-and-place task), Cal-QL's average post-fine-tuning normalized score (90) exceeded both plain CQL (71) and IQL (69), with the largest gaps on the hardest tasks — e.g. on one AntMaze task, CQL went from 62 (offline) to 98 (after fine-tuning) while Cal-QL went from 54 to 97, but on a harder relocate-binary Adroit task, CQL barely improved (6→69) while Cal-QL improved dramatically (3→98). Cal-QL achieved the best fine-tuned performance on 9 of the 11 tasks tested.

**How it connects.** Cal-QL is a direct, minimal extension of CQL's own objective (§7.2), addressing this week's secondary topic — transitioning from offline pretraining to online fine-tuning — by fixing exactly the pathology that plain conservatism (built for a purely offline setting) introduces once online interaction resumes.

---

## 12. A robotics-safety vignette

One further lecture topic is included briefly, as illustration rather than core material (it is neither a linked reading nor a professor's attention-point item): **Constrained Safety Critics** apply CQL-style conservative value estimation to a *safety* constraint rather than the task reward alone — maximizing task value subject to a learned "accident value" staying below a threshold, and constraining each policy update's KL-divergence from the previous policy (citing the unrelated *Constrained Policy Optimization* line of work for that mechanism). The motivating idea is that a robot-RL policy should be safe *at every iteration during learning*, not merely once training has converged — and using CQL-style conservative updates for the *safety* critic specifically avoids the accident-value estimate itself becoming falsely confident about states the policy hasn't actually visited, the same extrapolation concern §5 raised for ordinary task-value Q-functions.

---

## 13. Synthesis

Four questions this week's material answers, worth holding together explicitly:

**Is offline RL just BC with extra steps?** No — §4.2 and the dedicated Paper Breakdown in §4 show reward-driven Bellman backups can, under specific and now precisely characterized conditions (critical states, noisy-but-covering data, long horizons needing trajectory stitching), do something BC structurally cannot: exceed the quality of any single trajectory in the training data. But §9's robomimic study is the necessary caution — on clean, expert-quality human teleoperation data (as opposed to the noisy or agent-generated data offline RL's theory is built around), plain offline RL often still loses outright to a well-designed BC variant.

**Policy constraints or value conservatism — which is "the" fix?** Neither dominates; they're two different levers on the same underlying failure (§5.1's uncorrected extrapolation on out-of-distribution actions). Policy constraints (§6) restrict what the policy is *allowed to propose*; value conservatism (§7) instead makes the *value function* pessimistic about anything unfamiliar, leaving the policy free to optimize normally against that pessimistic estimate. TD3+BC's minimalism (§6) and CQL's log-sum-exp penalty (§7) reach broadly comparable final performance through genuinely different mechanisms — a reminder that "which mechanism" is a design choice, not a settled question.

**How much should any of this be trusted?** §7.4's two critique papers are the load-bearing caution for the whole week: Bellman error is not a reliable proxy for value accuracy, and even a well-trained, task-relevant representation doesn't guarantee stable offline value estimation. This is exactly why §8's standardized, held-out D4RL evaluation — not training-loss curves — is the field's actual arbiter of what works.

**What's still open?** The Levine/Kumar survey (§4) names the central unresolved issue plainly: no known "dataset sufficiency" condition guarantees any offline algorithm recovers a near-optimal policy, and most theoretical guarantees remain tabular or linear while practical methods use deep, generalizing function approximators — precisely the gap §7.4's instability results make concrete. Cal-QL's fix (§11) for the offline-to-online transition, and IQ-Learn's forward pointer (§10) into inverse-RL/imitation framings, are this week's two bridges into where the course goes next.

---

# Part B — Glossary

Moved to the project's running, cumulative glossary so terminology stays in one place across all weeks: see [`glossary.md`](./glossary.md).

---

## Quick self-check

Can you explain, without looking back at §5–§7, exactly *why* an offline Q-learning update can go wrong in a way an online one cannot — tracing the failure all the way from the Bellman backup's `max_{a'}` operator, through the absence of any correction step, to why the resulting policy actively seeks out the mis-estimated actions rather than merely tolerating them? And can you state, in one sentence each, what distinguishes a policy-constraint fix from a value-conservatism fix, using BCQ and CQL as your two concrete anchors — and then explain why TD3+BC and Cal-QL each count as a fix from one of those two families, even though neither paper is best known for benchmarking directly against the family's founding method?
