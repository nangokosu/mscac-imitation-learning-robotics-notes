# CSC2626 Imitation Learning for Robotics — Week 1 Study Notes

**Topic:** Imitation Learning vs. Supervised Learning
**Source:** Lecture 1 slides (F. Shkurti, CSC2626, Fall 2026) plus the week's required-reading papers (each summarized in its own Paper Breakdown below).
**Scope:** Logistics slides (staff, grading, forum links) skipped — notes start from the course's actual technical content.

---

## 0. Why does imitation learning need its own field?

Picture teaching someone to drive by having them watch you drive for an afternoon, then putting them alone behind the wheel in traffic. Everything they saw was *you* driving well — they never saw what to do the moment the car drifted half a lane off-center, because you never let it drift that far. If they make one small steering error, they're suddenly in a situation nothing in their afternoon of watching prepared them for.

That is, in miniature, the central problem of **imitation learning (IL)**: training a controller (for a robot, a simulated agent, a self-driving car) to reproduce the behavior of an expert **demonstrator**, using only recorded examples of the expert acting — no hand-written rule for "what to do in each situation," and (in the simplest version) no live guidance during training either.

A few pieces of vocabulary the rest of these notes lean on immediately:

- A **[robot policy](https://en.wikipedia.org/wiki/Robot_control)** is the function that decides what a robot should do next. It takes in whatever the robot currently senses — an **observation** (camera images, joint sensor readings, etc.) — and outputs an **action** (a motor command). "Training a policy" means fitting this function's parameters (in this course, almost always a neural network's weights) so it produces good actions.
- An **[expert](https://en.wikipedia.org/wiki/Imitation_learning)** or **demonstrator** is whoever/whatever generated the training data: a human physically operating the robot, a human driving a car, or even a slow, expensive planning algorithm you'd like to replace with a fast learned policy.
- A **demonstration** or **trajectory** is one recorded run: a sequence of (observation, action) pairs from start to finish of a task.
- The **[Markov Decision Process (MDP)](https://en.wikipedia.org/wiki/Markov_decision_process)** is the standard mathematical model this whole course sits on top of: an agent occupies a **state**, takes an **action**, the world responds with a new state (according to some transition rule) and (in the reinforcement-learning framing) a **reward**. A **[rollout](https://en.wikipedia.org/wiki/Markov_decision_process)** is one full run of a policy through an MDP, start to finish. The **horizon**, written **H** or **T** throughout this literature, is the number of decision steps in one rollout/episode — e.g. a 10-second task controlled at 10 actions/second has horizon 100.
- Imitation learning normally trains on states/actions only, with **no reward signal** — this is what distinguishes it from reinforcement learning (RL), covered starting Week 2. (Some later methods, like the "cost-to-go" idea in §4 and the inverse-reinforcement-learning unit in Week 6, blur this line by inferring a reward/cost from demonstrations — but the base case in this lecture assumes none is given.)

The lecture frames imitation learning for robotics around three "guiding principles": robots shouldn't have to learn every skill from scratch through blind trial-and-error; humans should be able to hand a robot new skills by *demonstrating* them rather than writing new code; and robots should be able to learn by watching *others* — other robots, other people's demonstrations, prior experience — not exclusively their own.

**Why this needs its own theory**, in one sentence, anticipating §3: the driving-instructor scenario above is not standard supervised learning, because the *policy's own errors change what situations it sees next* — and untangling exactly what that does to error, and how to fix it, is what most of this week's readings are about.

---

## 1. Basic robotics vocabulary this course assumes you don't already know

Before touching any algorithm, a short primer on ordinary hardware terms this course uses constantly. None of these are "obvious" just because they sound like everyday words — treat them the way you'd treat any new technical vocabulary.

| Term | Plain-language picture | Formal meaning |
|---|---|---|
| **[Manipulator](https://en.wikipedia.org/wiki/Robotic_arm) / robot arm** | A mechanical version of a human arm: a chain of rigid segments (**links**) connected by motors (**joints**) that can bend or rotate. | A serial chain of links and actuated joints ending in a tool. |
| **[End-effector](https://en.wikipedia.org/wiki/Robot_end_effector)** | Whatever's at the very tip of the arm doing the actual work — a hand, a suction cup, a welding torch. | The device mounted at the distal end of a manipulator that interacts with the environment. |
| **[Gripper](https://en.wikipedia.org/wiki/Robotic_gripper)** | The simplest possible "hand": usually just two flat fingers that open and close, like a pair of pliers. | A simple (often 1-DOF) end-effector for grasping, contrasted below with more complex multi-fingered hands. |
| **[Degrees of freedom (DOF)](https://en.wikipedia.org/wiki/Degrees_of_freedom_(mechanics))** | Count the independent ways something can move — a hinge (like your elbow) has 1 DOF (it only bends one way); your shoulder, which can swing in any direction, has 3. | The number of independent parameters needed to fully specify a mechanism's configuration. A typical robot arm has 6–7 DOF; a human-like multi-fingered hand can have 15–20+. |
| **[Forward / inverse kinematics](https://en.wikipedia.org/wiki/Robot_kinematics)** | Forward: "if I set each joint to this angle, where does the fingertip end up?" (easy — just chain the geometry through). Inverse: "I want the fingertip *here* — what joint angles get me there?" (harder — often many solutions, or none). | Forward kinematics maps joint angles → end-effector pose; inverse kinematics (IK) solves the reverse, usually via numerical optimization. |
| **Joint space vs. task space** | Joint space: describing the robot by its own joint angles (elbow bent 40°, wrist twisted 10°...). Task space (also "Cartesian space" or "end-effector space"): describing it by where the *tool tip* is in the room (x, y, z position + orientation), ignoring exactly how the joints got it there. | Two coordinate systems for the same configuration, related by forward/inverse kinematics. Which one a policy predicts actions in is a real design choice you'll see argued over below (e.g. §6.2's ALOHA vs. §7.4's Diffusion Policy). |
| **[Workspace](https://en.wikipedia.org/wiki/Robot_kinematics)** | The full bubble of physical space the end-effector can actually reach. | The reachable set of end-effector poses given the arm's link lengths and joint limits. |
| **[Proprioception](https://en.wikipedia.org/wiki/Proprioception)** | The sense of where your own limbs are without looking — close your eyes and you still know roughly where your hand is. | In robotics: internal sensing of the robot's own joint angles/velocities/forces, as opposed to external sensing (cameras) of the world. |
| **Force/torque sensing** | A sense of touch — how hard is the gripper squeezing, how much resistance is the arm pushing against. | Sensors (often mounted at the wrist) that measure contact forces and torques, distinct from vision. |
| **[RGB-D camera](https://en.wikipedia.org/wiki/RGB-D_camera)** | A regular color camera that *also* measures, per pixel, how far away that point in the scene is. | A camera producing an aligned color image (RGB) plus a per-pixel depth map (D). |
| **[Telerobotics / teleoperation](https://en.wikipedia.org/wiki/Telerobotics)** | Puppeteering a robot in real time — a human's live movements are captured and immediately replayed on the robot. | Real-time human control of a remote or co-located robot, used throughout §6 as the standard way to *collect* demonstrations for imitation learning. |
| **Kinesthetic demonstration** | Literally grabbing the robot's arm with your hands and physically guiding it through a motion, with the robot recording its own joint sensors as you move it. | A demonstration-collection method where a human directly, physically moves the robot (as opposed to teleoperation through a separate controller device). |
| **Embodiment** | The specific physical "body" a policy is controlling — a robot with a 6-DOF arm and a two-finger gripper is a different embodiment from a humanoid with two 7-DOF arms and five-fingered hands, even doing the "same" task. | The physical form (kinematics, sensors, actuators) that grounds a policy's action space. A recurring theme in §6: whether data/policies collected on one embodiment transfer to another. |
| **Bimanual manipulation** | Using two arms together, the way you'd use both hands to open a jar. | Manipulation tasks/systems involving coordinated control of two robot arms simultaneously. |

---

## 2. The starting point: behavioral cloning

The most direct way to turn "a pile of recorded (observation, action) pairs from an expert" into a controller is **[behavioral cloning (BC)](https://en.wikipedia.org/wiki/Imitation_learning#Behavior_Cloning)**: treat it as an ordinary supervised-learning problem. Collect a dataset `{(o_1, a_1), (o_2, a_2), ..., (o_N, a_N)}` of observations paired with the action the expert took at that moment, and train a network `π_θ(o) → a` to predict the expert's action from the observation, using a standard regression or classification loss (you already know how to do this part — it's ordinary supervised training).

That's genuinely all BC is: no environment interaction during training, no reward, no planning — just "fit a function to the expert's (state, action) pairs." Its appeal is exactly its simplicity, and it remains, per the lecture, "one of the simplest machine learning methods to acquire robotic skills in the real world." Its problems (starting in §3) are equally simple to state but surprisingly deep to fix.

### Paper Breakdown: ALVINN — the historical starting point

*Citation: Dean A. Pomerleau, "ALVINN: An Autonomous Land Vehicle in a Neural Network," Advances in Neural Information Processing Systems 1 (NeurIPS/NIPS 1989).*

**Problem/motivation.** In the late 1980s, autonomous road-following was built from hand-engineered image-processing pipelines (edge detectors, hand-tuned road-color segmentation, hand-designed steering logic). Pomerleau's complaint: a fixed, hand-tuned pipeline "remains fixed across various driving situations" — every new road type or lighting condition needs new manual engineering. He proposed instead letting a neural network *learn*, from data, which image features matter for steering.

**Key idea/innovation.** Train one small feedforward network, end-to-end via backpropagation, to map a raw camera (and laser range-finder) image directly to a steering command — this is BC, applied to driving, years before the term "behavioral cloning" was standard vocabulary. As the paper puts it, "the data, not the programmer, determines the salient image features."

**Method & training procedure.** The original network: a **30×32-pixel** camera "retina" (960 units) plus an 8×32 laser-range retina (256 units) plus 1 feedback unit (1,217 input units total) → a **29-unit hidden layer** → a **46-unit output layer**, of which 45 units encode steering curvature (a "hill" of activation centered on the correct curvature, not a single spike) and 1 predicts road brightness for the next frame's feedback input. Rather than risk collecting a large, diverse real-driving dataset (logistically hard at the time), Pomerleau trained on **1,200 simulated road images** with randomized position/orientation/lighting, for 40 epochs, using the "Warp" parallel backprop machine — reaching ~90% correct-curvature-within-tolerance on held-out simulated images, then driving the real NAVLAB test vehicle at 0.5 m/s along a 400 m wooded path.

**Results.** On that single test course, ALVINN performed comparably to CMU's best hand-engineered vision system of the time (which drove marginally faster, at 1 m/s, though on faster dedicated hardware). A companion range-finder ablation showed the laser input was largely dispensable for plain road-following.

**How it connects.** ALVINN is the field's founding demonstration that BC *works at all* for real robot control — but its own 1989 paper already flags, as future work, exactly the two failure modes §3 formalizes: insufficient training variability, and the need to teach recovery from mistakes, not just accurate driving. A later, expanded Pomerleau paper (a 1993 book chapter, "Knowledge-Based Training of Artificial Neural Networks for Autonomous Robot Driving") is where the field's two classic BC fixes were actually first demonstrated in detail: **synthetic image augmentation** (each real camera frame was digitally shifted and rotated 14 extra ways, simulating what the road would look like from slightly off-center or off-angle vehicle positions, with the correct steering angle for each synthetic view recomputed geometrically) to manufacture "how do I get back on track" examples the human driver never dangerously demonstrated on purpose, and a **fixed-size replay buffer** (200 patterns, refreshed each cycle to keep the buffer's average steering direction close to straight-ahead) to stop the network from **catastrophically forgetting** how to handle a curve after a long straight stretch of new training data. With both fixes, that later system trained to a usable policy from about **4 minutes** of live human driving. (Course notes should keep these two Pomerleau papers distinct — the 1989 NeurIPS paper this course cites is the simulated-training, dual-retina proof of concept; the shift/rotate-augmentation-plus-buffer numbers usually quoted for ALVINN come from the follow-up work.) These two problems — needing more varied training data than an expert naturally provides, and forgetting old skills as new data streams in — are exactly what §3 names formally as *covariate shift* and *catastrophic forgetting*.

---

## 3. Why imitation learning isn't just supervised learning

### 3.1 Two failure modes, informally

Go back to the driving-instructor analogy from §0. Two distinct things can go wrong:

1. **Covariate shift.** The learner's *own* actions determine which states it visits next. The moment its policy is even slightly different from the expert's, it starts drifting into observations the expert's demonstrations never covered — and because the training data never covered them, the policy has no real signal for what to do there, so it's likely to err *again*, drifting further still. "Covariate" here just means the input distribution (the "X" side of the input→output mapping); the term names the fact that this input distribution *shifts* between training (expert's states) and test (learner's own states) — quietly breaking the standard supervised-learning assumption that train and test data come from the same distribution.
2. **[Catastrophic forgetting](https://en.wikipedia.org/wiki/Catastrophic_interference).** If a network is trained incrementally (e.g. on a live stream of new data, the way ALVINN's later on-the-fly version was), a long stretch of one kind of data (e.g. straight highway) can overwrite what the network previously learned about a different kind of data (e.g. sharp curves) it isn't currently seeing — a general pathology of online/streaming neural-network training, not unique to imitation learning, but one BC runs into constantly because demonstrations are naturally collected in whatever order the demonstrator happens to encounter situations.

### 3.2 Formalizing covariate shift: why the error compounds *quadratically*

This is the single most important quantitative idea in Week 1, and it's worth building up carefully, because "the error compounds" undersells how bad the effect actually is.

Picture a robot policy that is trained to match an expert with **per-step error rate ε** — meaning, on any single state drawn from the expert's own demonstrated state distribution, the learned policy disagrees with the expert with probability ε (or, more generally, incurs expected loss ε under some bounded loss function). Naive intuition, borrowed from ordinary supervised learning, says: over a **horizon of T steps**, you'd expect roughly `T · ε` total mistakes — errors are independent per step, so they should just add up.

That intuition is wrong, and the reason is exactly covariate shift. The first time the policy makes a mistake, it lands in a state that likely wasn't in the expert's demonstrated distribution — and the ε guarantee **only ever applied to states the expert actually demonstrated**. In this new, off-distribution state, the policy's error probability isn't bounded by ε at all; it can be much higher, since the policy never trained on anything like it. That elevated error rate then compounds again at the *next* step, and so on.

Formally (Ross & Bagnell 2010's result, restated precisely in this week's DAgger and "Invitation to Imitation" papers): if a policy trained via ordinary BC has expected 0-1 loss ε under the expert's own state distribution, the best *general* guarantee on total task cost is

```
J(policy) ≤ J(expert) + T² · ε
```

— quadratic in the horizon T, not linear. And this bound is **tight**: there exist simple constructed examples (a short Markov chain is enough) where a BC policy really does rack up close to `T²·ε` extra cost, so this isn't a pessimistic worst case that never happens in practice — it's the honest rate.

> **Worked intuition for why T² appears, not just T.** Suppose a mistake, once made, typically takes roughly T more steps to fully "recover" from (imagine drifting off a marked path — getting back on track and finishing the remaining route both take time proportional to how much task remains). If a mistake can happen at any of the T steps, and each one costs (up to) T steps' worth of downstream disruption, the total cost scales like (number of steps) × (cost per mistake) ≈ T × T = T². This is a loose sketch, not the formal proof (which uses a careful induction over per-step distribution shift), but it captures why the horizon appears *twice*: once for "how many chances there are to go wrong," and once for "how long each wrong turn costs you."

Two constants sharpen this picture across the readings this week:

- **u (recovery cost / "how bad is one bad action"):** a bound on how much worse the expert's own future cost-to-go gets after one deviation from its own advice. If the task is very *forgiving* — a wrong turn is easy to correct — u is small (close to 1) and the effective penalty is milder. If a single wrong action can be catastrophic and unrecoverable (u close to T), the T² effect is essentially unavoidable.
- **μ ("recoverability"), the version DAgger's own theory formalizes precisely:** the largest possible gap, at any state, between the value of the expert's chosen action and the value of the worst alternative action, under the expert's own value function. Small μ means "even the expert's worst-looking alternative isn't disastrous" (highly recoverable); large μ means some mistakes are unrecoverable.

### Paper Breakdown: "An Invitation to Imitation" — the framing paper for the whole field

*Citation: J. Andrew Bagnell, "An Invitation to Imitation," CMU Robotics Institute Technical Report CMU-RI-TR-15-08, 2015.*

**Problem/motivation.** This is a survey/tutorial, not a paper reporting new experiments — Bagnell's own decade of lessons on why robotics needs behaviors (driving, legged locomotion, UAV flight) that are easy for a human to *demonstrate* but hard to hand-code, and why naively applying ordinary supervised learning to "programming by demonstration" fails in practice even with abundant, low-error training data. He identifies two specific ways imitation learning breaks the assumptions of classical supervised learning: (1) the learner's predictions change its own future inputs (violating the i.i.d. assumption underlying supervised-learning theory — exactly §3.2's covariate shift, formalized here as the source of the paper's own O(T²ε) result), and (2) real robots are usually built on top of *planners* that reason many steps ahead using a hand-designed cost function, and a purely reactive learned policy that ignores this planning structure performs poorly on long-horizon tasks.

**Key idea/innovation.** Bagnell's unifying reframing: imitation learning problems split into two largely independent families of fixes, each a principled reduction to well-understood machinery. (a) **Interactive reductions to supervised learning** (DAgger, covered in §4) fix covariate shift by letting the learner query the expert along states *it itself* visits, not only the expert's own demonstrated states. (b) **Inverse Optimal Control (IOC)** fixes the "reactive policies can't plan" problem by learning a *cost function* from demonstrations instead of a policy directly, so an existing planner can use it — because cost functions, Bagnell argues, generalize better across situations than directly-learned policies or value functions do.

**Method & training procedure.** The paper's most load-bearing formal content, for this week's purposes, is the precise error-compounding argument in §3.2 above: a BC policy with per-step error ε (measured under the *expert's own* state distribution) has worst-case total cost `J(π) ≤ J(π*) + T²ε`, provably tight. It also describes a conceptually simple (if impractical) exact fix: **forward training** — train a *separate* policy for each timestep t = 1, …, T, always training policy t on the distribution actually induced by running the previously-learned policies for steps 1, …, t−1. Because each timestep's policy is only ever evaluated on the exact distribution it was trained on, no covariate shift ever occurs, and the cost bound improves to `J(π) ≤ J(π*) + u·T·ε` (linear in T, using the "u" recovery-cost constant from §3.2) — but training T separate policies is impractical whenever the horizon is long or variable. On the IOC side, the paper walks through **LEARCH** (Learning to Search), a functional-gradient/boosting algorithm for learning a cost map from demonstrated paths: repeatedly (a) plan a route under the current cost estimate, (b) compare that route to the demonstrated one state-by-state (raising the learned cost anywhere the plan went that the demonstration avoided, lowering it anywhere the demonstration went that the plan avoided), (c) fit a regressor to these adjustments, and (d) add it to the running cost estimate — provably a form of gradient boosting that converges over several iterations, not in one shot.

**Results.** As a survey paper it reports no controlled experiments of its own, only illustrative case studies from Bagnell's and collaborators' prior work: a driving-simulator comparison where plain supervised BC "averages about 3–4 failures per lap" while an interactive (DAgger-style) policy "very quickly reaches nearly 0 falls per lap" no matter how much more data plain BC gets; a UAV reactive forest-flight controller performing "at nearly the same effectiveness as a human pilot," with failures traced to its limited field of view rather than the learning method; and a DARPA-funded rough-terrain ground vehicle ("Crusher") that, using a learned LEARCH-style cost function, traversed "thousands of kilometers of diverse, rough terrain with minimal human intervention over years of field testing."

**How it connects.** This paper is the conceptual umbrella over almost the entire lecture: its O(T²ε) result is the formal justification for why §4's DAgger exists at all, and its "learn a cost, not a policy" IOC framing is the direct ancestor of the inverse-reinforcement-learning unit in Week 6. Its forward-training construction is also the conceptual bridge to DAgger's own theory in §4 — DAgger can be understood as forward training's practical, single-policy approximation.

### 3.3 Catastrophic forgetting and the online-learning buffer fix

Separately from covariate shift, a policy trained incrementally on a live data stream (rather than one fixed batch) can simply **forget** — new data overwrites what old data taught it, especially if the environment or task distribution "drifts" over time (a phenomenon sometimes called **[concept drift](https://en.wikipedia.org/wiki/Concept_drift)**: the statistical relationship between inputs and correct outputs itself changes over time, e.g. lighting conditions, road types, or object appearances shifting as a demonstration session goes on). This is a general pathology of online learning and reinforcement learning, not unique to imitation learning — but BC hits it often because demonstrations naturally arrive in whatever order the demonstrator happens to encounter situations, not shuffled i.i.d. the way a static supervised-learning dataset would be.

The standard mitigation, already previewed in §2's ALVINN discussion, is an **experience replay buffer**: instead of training only on the newest incoming data, maintain a fixed-size buffer of past (observation, action) examples, and train on a mix of new and buffered-old examples every update — actively curated (in ALVINN's later version, by biasing the buffer's contents to keep the *average* recorded action close to some neutral baseline) so that rare-but-important situations (like sharp turns, which occur less often than straight driving) don't get diluted away by whatever's most recent.

---

## 4. DAgger: fixing covariate shift by querying the expert on your own mistakes

### 4.1 The core idea

If covariate shift happens because the training data only covers the *expert's* states, the direct fix is: let the *learner's own* policy generate states, and get expert labels for *those* states too. That's the entirety of DAgger's ("Dataset Aggregation") idea — an iterative loop:

1. Train an initial policy on whatever demonstration data you already have.
2. Roll that policy out in the real environment (let it drive, or manipulate, on its own).
3. At every state the policy visits during that rollout, ask the expert: "what would *you* have done here?" — the expert's answer is recorded as a label, but the expert's own answer is **not** what actually gets executed; the learner's own (possibly wrong) action keeps driving the rollout forward, precisely so that later states in the rollout are the ones the learner's mistakes would actually lead to.
4. Add all these newly-labeled (learner-visited-state, expert-action) pairs into the training set (which keeps growing — this is the "aggregation" in the name).
5. Retrain a fresh policy on the whole aggregated dataset so far.
6. Repeat.

Over enough iterations, the aggregated dataset increasingly covers the states the *learner itself* tends to visit, not just the states the expert originally happened to visit — directly attacking the covariate-shift mechanism from §3.2.

### 4.2 No-regret online learning — the theoretical lens DAgger is built on

DAgger's guarantees are proven by treating each iteration of the loop as one round of an **[online learning](https://en.wikipedia.org/wiki/Online_machine_learning)** problem, and requiring the supervised-learning step (step 5 above) to behave like a **no-regret** algorithm.

**[Regret](https://en.wikipedia.org/wiki/Regret_(decision_theory))**, informally: how much worse a sequence of decisions performs, in total, compared to the single best *fixed* decision you could have made in hindsight, knowing everything in advance. A learner has **no regret** if this gap, averaged over N rounds, shrinks to zero as N grows — i.e. it eventually performs about as well as the best fixed choice it could only have known in hindsight, even though it had to commit to each round's choice before seeing what was coming.

```
regret_N = (1/N) · Σ loss_i(policy_i) − min over π of (1/N) · Σ loss_i(π)
```

A no-regret guarantee needs `regret_N → 0` as N → ∞. Many practical algorithms (e.g. training via gradient descent when the loss is strongly convex) provably have this property.

### 4.3 Why this fixes the quadratic bound

DAgger's central theoretical payoff: **if the retraining step in the loop is (or behaves like) a no-regret online learner**, then somewhere among the sequence of policies it produces — or a policy chosen uniformly at random from that sequence — there is one whose total task cost is bounded by

```
J(policy) ≤ J(expert) + u · T · ε_N + O(1)
```

**linear** in the horizon T (using the same recovery-cost constant u from §3.2), where ε_N is the average loss of the *best policy in hindsight* over all the iterations' aggregated data. This is the direct improvement over plain BC's `T² · ε` bound from §3.2 — the same style of guarantee forward training achieved in §3's "Invitation to Imitation" discussion, but without needing to train a separate policy per timestep.

### Paper Breakdown: DAgger

*Citation: Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning," AISTATS 2011.*

**Problem/motivation.** Formalizes exactly the covariate-shift diagnosis from §3.2 and proves it's tight (via a sequence-prediction construction where mistakes provably compound at close to the T² rate), then asks: can an *iterative*, interactive training procedure recover a linear-in-T guarantee, the way forward training does, without forward training's impracticality (a separate policy per timestep)?

**Key idea/innovation.** The DAgger loop from §4.1, reframed as a **reduction of imitation learning to no-regret online learning**: treat each iteration's retraining step as one round of an online-learning problem, and show that *any* no-regret algorithm used this way — not some special-purpose imitation-learning trick — yields the linear-in-T guarantee.

**Method & training procedure.** The exact algorithm, as given in the paper (β_i is the probability of querying the expert, rather than the learner's own current policy, at iteration i's rollout — used only to collect data, never to decide what the aggregated dataset's *labels* are):

```
DAgger(π̂_1, expert, N):
    D ← ∅
    for i = 1 to N:
        π_i ← β_i · expert + (1 − β_i) · π̂_i      # mixture policy used only to ROLL OUT
        roll out π_i for T steps, collecting visited states
        D_i ← {(state, expert(state)) : state visited above}   # expert LABELS every visited state
        D ← D ∪ D_i                                              # aggregate
        π̂_{i+1} ← train a new policy on all of D
    return best π̂_i on a held-out validation set (or a uniformly random one — both provably work)
```

A common, "parameter-free" special case sets β_1 = 1 (first iteration is pure expert demonstration, since there's no learned policy yet) and β_i = 0 for i > 1 (every later iteration rolls out the learner's own policy alone) — the paper notes this variant "often performs best in practice." The only formal requirement on β_i for the theory to hold is that its running average shrinks to 0 as iterations increase. The paper proves this reduction works for *any* no-regret learner (Theorem 3.1/3.2), and separately proves finite-sample versions (Theorems 3.3/3.4) bounding how many total rollout trajectories (`N`) are needed.

**Results.** Three experiments, each comparing plain supervised BC, the earlier SMILe/SEARN-style methods, and DAgger:
- **SuperTuxKart** (steering a kart via a linear ridge-regression controller on image features): plain BC's performance never improves no matter how much more expert-only data it gets (the falls happen off-distribution); SMILe still falls roughly twice per lap after 20 iterations; **DAgger reaches essentially zero falls per lap after 15 iterations**, and is already close after just 5.
- **Super Mario Bros.** (4 independent linear SVMs for left/right/jump/speed, scored by distance traveled before dying): plain BC stagnates because it never learns to "get unstuck" from obstacles (the expert always avoided getting near one). DAgger **outperforms every tested SMILe and SEARN configuration**; best variant reaches a score of 3030 out of a max around 4300 on the test stage.
- **Handwriting recognition**, reframed as a degenerate imitation-learning problem (predicting each character left-to-right, conditioned on the *previously predicted* — not ground-truth — character, exactly mirroring how a policy's own past actions shape its future inputs): a non-sequential baseline gets 82% character accuracy; standard "supervised" training (always conditioning on the *true* previous character, never the model's own guess) gets 83.6%; **DAgger reaches 85.5%**, the best of all methods tested.

**How it connects.** DAgger is the direct, practical instantiation of the "Invitation to Imitation" paper's forward-training idea, replacing "one policy per timestep" with "one continuously-retrained policy, trained on an ever-growing, learner-informed dataset." Its core limitation — needing to query a live expert repeatedly, which is often impossible outside a simulator or a very patient human — is precisely the gap the next section's paper addresses from a completely different angle.

---

## 5. A second fix: changing the loss function instead of querying the expert live

DAgger fixes covariate shift, but at a real cost: it needs the expert available *during training*, repeatedly, to label states the (partially-trained) policy visits. For a human demonstrator teleoperating a physical robot, or for many real deployments, this is expensive or simply infeasible — you usually only get to collect demonstrations once, offline, and then train.

This raises a natural question, and one that unsettled some of the field's long-standing "folk wisdom": is the T² vs. T gap between offline BC and online/interactive methods like DAgger a fundamental property of *offline learning itself* — or is it partly an artifact of exactly *how* BC's classical analysis measures error?

### Paper Breakdown: "Is Behavior Cloning All You Need? Understanding Horizon in Imitation Learning"

*Citation: Dylan J. Foster, Adam Block, Dipendra Misra, arXiv:2407.15007, 2024 (accepted to NeurIPS 2024).*

**Problem/motivation.** The classical result from §3–§4 (BC: `T²·ε`; DAgger-style online IL: `T·ε` under a recoverability condition) is the textbook justification for preferring interactive methods over plain offline BC. But interactive querying is often infeasible in practice, and offline BC remains the dominant method used empirically despite its supposedly worse theoretical guarantee. The authors ask directly: is offline BC truly stuck with quadratic sample complexity, or can a better *analysis* — not a different algorithm — close the gap?

**Key idea/innovation.** The classical `T²·ε` bound is derived using **0-1 (indicator) loss** — did the policy's action exactly match the expert's, yes or no — and measures the resulting distribution mismatch in **total variation distance**. The paper shows that if you instead train BC with the **log loss** (maximum-likelihood / cross-entropy over the predicted action distribution — the same objective used ubiquitously in practice, e.g. next-token prediction in language models) and measure the resulting error in **squared [Hellinger distance](https://en.wikipedia.org/wiki/Hellinger_distance)** between whole predicted-vs-expert *trajectory* distributions (rather than distance at a single state), the extra factor of T that plain BC's analysis picks up when translating "per-state prediction error" into "total rollout cost" simply disappears from the bound.

**Method & training procedure.** Precisely: with `LogLossBC` defined as maximum-likelihood training of the policy against the expert's action distribution,

```
LogLossBC:  π̂ = argmin over π in Π of  Σ_i Σ_h  −log( π_h(a_h^i | x_h^i) )
```

the paper proves a regret bound of the shape (their Theorem 2.1, for deterministic experts):

```
J(expert) − J(π̂) ≤ 4·R · D²_Hellinger(trajectory distribution of π̂, trajectory distribution of expert)
```

where R bounds the *total* cumulative reward range across an episode (R ≤ H for ordinary per-step-bounded rewards, but R = O(1) for many sparse-reward tasks) — critically, this bound scales with R, **not with R·H**, i.e. no extra horizon factor is multiplying the supervised-learning error term at all. Combined with a standard maximum-likelihood generalization bound, this yields **horizon-independent** sample complexity whenever two conditions hold: (1) the cumulative reward range R is controlled (doesn't grow with H — true for sparse-reward or otherwise bounded-payoff tasks), and (2) the policy class has controlled *supervised-learning* complexity, which in practice essentially means **parameter sharing across timesteps** — a single network (a transformer, an MLP) applied identically at every decision step, as is standard in essentially all modern IL, rather than a distinct policy per timestep. Under **dense** per-step rewards (R = O(H)) the bound degrades gracefully to *linear*-in-H — still an improvement over the classical quadratic rate, and matching what DAgger achieves, but without ever querying an expert live.

The paper is explicit about why this doesn't contradict DAgger's own theory: DAgger's advantage over plain BC, in the *classical* analysis, only shows up for policy classes **without** parameter sharing (a distinct policy per timestep, the setting the older tabular analyses implicitly assumed) — and "virtually all empirical [imitation learning] work uses parameter sharing." For a shared-parameter policy class, their reanalysis shows plain offline log-loss BC already matches what online DAgger-style methods buy you, in the regimes that matter in practice.

**Results.** Across four testbeds — a MuJoCo continuous-control task (Walker2d), an Atari game (Beam Rider), a custom sparse-reward top-down navigation task, and training a small GPT-2-style transformer to generate valid bracket sequences — the empirically measured regret of LogLossBC is essentially **flat as the horizon H grows** on the tasks where their theory predicts flatness (sparse reward, parameter sharing), confirming the theoretical prediction directly; on the one task (bracket-sequence generation) where regret *does* grow with H, they show this tracks a genuine growth in how complex a policy is needed to represent the expert at longer horizons, not a violation of the theory. A companion finding: log-loss BC is also markedly more *robust* to imperfect optimization than 0-1-loss BC — under a fixed amount of optimization error, indicator-loss BC's bound degrades by a full extra factor of H, while log-loss BC's degrades by no horizon factor at all.

**How it connects.** This paper doesn't overturn DAgger's guarantee so much as clarify exactly *when* it actually buys you something over well-analyzed offline BC — for the parameter-shared neural network policies used throughout the rest of this course (including nearly every architecture in §6–§7 below), the honest headline is closer to "the loss function and reward structure you choose matter as much as whether you query the expert live," rather than "offline BC is fundamentally doomed to quadratic error."

---

## 6. Teleoperation: how demonstrations for manipulation actually get collected

Sections 3–5 treated "getting expert-labeled data" as an abstraction. For robot *manipulation* specifically (rather than driving, where a human sitting in the vehicle already naturally provides steering/pedal data), that data almost always comes from **teleoperation** — a human, in real time, driving a robot's actual arm(s) and hand(s) through the task, while the robot logs its own sensor readings and the resulting commands. The five papers in this section are all, at heart, about *better teleoperation hardware/interfaces* — because the quality, diversity, and cost of collectible demonstrations bottlenecks everything built on top of them.

### 6.1 A shared building block: action chunking

Before the individual systems, one recurring design idea is worth introducing on its own, since four of this week's papers use some version of it. A plain BC policy predicts **one action at a time**: observe, act, observe again, act again. **Action chunking** instead has the policy predict a short **sequence** ("chunk") of several future actions in a single forward pass, of which typically only the first few are actually executed before the robot re-observes and re-plans (a pattern control theory calls **[receding-horizon control](https://en.wikipedia.org/wiki/Model_predictive_control)** — plan further ahead than you commit to, and keep re-planning as new information arrives).

Why this helps, connecting back to §3.2: querying the policy less often (because each query commits to several steps at once) shortens the *effective* decision horizon, which directly shrinks the T²-type compounding-error penalty. It also helps with a second, more practical problem: real human teleoperation demonstrations are noisy and **non-stationary** — pauses, hesitations, small corrections — and a policy predicting only the very next action can get confused about whether "pause" or "continue" is correct at a given instant, in a way that predicting a whole short future sequence is more robust to.

### 6.2 Leader-follower teleoperation and Action Chunking Transformers

The most direct way to teleoperate a bimanual robot is a **leader-follower** rig: a scaled-down, cheap, *passive* (unpowered) mechanical duplicate of the robot's own arm that a human physically holds and moves, with its joint sensors' readings sent, joint-by-joint, as position commands to the real ("follower") robot arm.

### Paper Breakdown: ALOHA

*Citation: Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware," RSS 2023.*

**Problem/motivation.** Fine-grained bimanual manipulation (threading a cable tie, slotting a battery into a holder) traditionally needed expensive, precisely-calibrated industrial hardware — putting this kind of research out of reach for most labs. Separately, BC in these high-precision settings suffers from both compounding error (§3.2) and the non-stationarity problem noted in §6.1 (human demonstrations include pauses/hesitations a naive single-step policy struggles to fit).

**Key idea/innovation.** Pair a **cheap ($20k total), open-source leader-follower bimanual teleoperation rig** with a new policy architecture, the **Action Chunking Transformer (ACT)**, that predicts chunks of future actions via a generative model — together letting inexpensive, imprecise hardware achieve 80–90%+ success on fine manipulation from only 10–20 minutes of demonstration per task.

**Method & training procedure.** Hardware: two off-the-shelf **WidowX-250** arms (leader, human-held) and two **ViperX-300** arms (follower), all built on cheap, easily-replaceable Dynamixel servo motors. Crucially, the leader→follower mapping is done in **joint space**, not task space (§1) — each leader joint angle is sent directly as a position command to the matching follower joint, avoiding inverse-kinematics singularities and keeping control latency low; the system runs at 50 Hz (a user study found dropping to 5 Hz caused a 62% task-completion slowdown). Four wrist- and scene-mounted RGB cameras provide vision. **ACT** is trained as the decoder of a **[conditional variational autoencoder (CVAE)](https://en.wikipedia.org/wiki/Variational_autoencoder)**: during training, a transformer encoder compresses the true future action chunk plus current state into a latent "style" variable `z` (capturing demonstration variability like hesitations); a transformer decoder, conditioned on `z`, the camera images, and the current joint state, outputs a chunk of the next **k = 100** actions in one pass. At test time the encoder is discarded (`z` set to zero) and the decoder runs open-loop over its predicted chunk, with **temporal ensembling**: overlapping chunks predicted at different past timesteps are combined via an exponentially-weighted average (`weight ∝ exp(−m · how_far_in_the_past)`) so more recent predictions dominate while jitter/noise gets smoothed out.

**Results.** Six real tasks, 50 demonstrations each (100 for the hardest, "Thread Velcro"): success rates from 20% (Thread Velcro) up to 96% (Slot Battery), with the two showcase tasks (opening a translucent cup, slotting a battery) headlined in the abstract at "80–90%" (the paper's own detailed per-task table reports 84% for the cup and 96% for the battery — the abstract's range is an approximate summary, not each task's literal figure). Ablations isolate the two design choices cleanly: sweeping the chunk size from k=1 to k=100 raised success from 1% to 44% on one task in isolation, and removing the CVAE/generative component specifically hurt performance on *noisy, human-collected* data (35.3% → 2%) while making almost no difference on noise-free scripted data — direct evidence the CVAE is earning its keep on real human demonstration noise specifically, not just generically helpful.

**How it connects.** ACT's action-chunking idea is reused (with a different generative backbone) by Diffusion Policy in §7.4, and the leader-follower rig itself is directly critiqued — for being robot-specific and pushing all safety/collision-avoidance onto the human — by the exoskeleton-based systems in §6.5–§6.6.

**A brief note on Mobile ALOHA (not itself a required reading).** A follow-up system (Fu, Zhao, Finn, CoRL 2024) mounts the same leader-follower arms on a wheeled mobile base, with the human's own waist tethered to the low-friction, back-drivable base so that walking drags it along while both hands stay free for the arms — enabling whole-body tasks like cooking or opening cabinets. Its main empirical finding directly reinforces a theme of this whole lecture: **co-training** new mobile-task demonstrations together with the original static-ALOHA dataset dramatically improved data efficiency, raising success rates by up to 90 percentage points from only 50 new demonstrations per task — reusing old data to make new data go further, rather than needing enormous fresh datasets per task.

### 6.3 Decoupling data collection from any particular robot

Leader-follower rigs, precisely because they're built to match one specific robot's kinematics, only ever collect data *for that robot*, in a lab, with that robot physically present. The next system asks: what if you didn't need a robot at all to collect manipulation data?

### Paper Breakdown: UMI (Universal Manipulation Interface)

*Citation: Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, Shuran Song, "Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots," RSS 2024.*

**Problem/motivation.** Two compounding bottlenecks: standard demonstration collection (teleoperation, kinesthetic teaching) needs a *physical robot on site*, making it expensive and slow to scale data collection across the diversity of real-world environments (homes, restaurants, outdoors) a general-purpose policy would need to see; and policies trained on one robot's action representation (its specific joint encoders/kinematics) typically don't transfer to a differently-shaped robot without retraining.

**Key idea/innovation.** A **hand-held, GoPro-instrumented gripper** lets a human collect demonstrations anywhere, with no robot present at all; an **embodiment-agnostic action representation** — relative end-effector motion rather than absolute joint angles — then lets the same collected data train policies that deploy zero-shot on different physical robot arms.

**Method & training procedure.** The device: a 3D-printed gripper (~$73 in parts) with a wrist-mounted GoPro (wide fisheye lens) as its *only* sensor — no external motion-capture. Side mirrors placed in the camera's peripheral view create an implicit stereo (depth) cue from a single lens. The gripper's own 6-DOF pose over time is recovered purely from the recorded video via an inertial-monocular **[SLAM](https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping)** system (a modified ORB-SLAM3, fusing the GoPro's built-in accelerometer/gyroscope with visual tracking) — average tracking error only ~6.1 mm / 3.5° versus motion-capture ground truth. Actions and observations are represented as **relative SE(3) poses** (this gripper's pose *relative to* a recent reference pose) plus a gripper-width value, rather than any robot's absolute joint angles — a representation any robot's own inverse-kinematics controller can consume regardless of how many joints or what kinematic chain it has, which is precisely what makes the same collected data usable across different arms. A **Diffusion Policy** (§7.4) trained on this representation, conditioned on the wrist camera's video, predicts chunks of future relative poses.

**Results.** Tested zero-shot on both a UR5 arm (its "home" collection platform) and a completely different Franka arm: cup-arrangement success stayed at 100% on the UR5 and 90% on the untrained-for Franka. An "in-the-wild" cup-arrangement dataset (1,400 demonstrations collected across 30 different real-world locations — homes, offices, restaurants — by 3 people in just 12 person-hours) reached 71.7% success overall, including 75% on cup styles never seen during data collection. Other tasks (dynamic tossing, bimanual cloth folding, dish washing) ranged 70–87.5% success.

**How it connects.** UMI's relative-pose, embodiment-agnostic representation is the direct opposite design choice from ALOHA's joint-space leader-follower mapping in §6.2 — a genuine design axis worth noticing across this section: mirror the target robot exactly (ALOHA) versus abstract away from any particular robot entirely (UMI). Its use of Diffusion Policy as the downstream learner is explained in full in §7.4.

### 6.4 VR-based dexterous hand tracking

A different approach to teleoperation skips *any* physical controller rig and instead tracks the human operator's own body directly with a consumer VR/AR headset.

### Paper Breakdown: Bunny-VisionPro

*Citation: Runyu Ding, Yuzhe Qin, Jiyue Zhu, Chengzhe Jia, Shiqi Yang, Ruihan Yang, Xiaojuan Qi, Xiaolong Wang, "Bunny-VisionPro: Real-Time Bimanual Dexterous Teleoperation for Imitation Learning," arXiv:2407.03162, 2024.*

**Problem/motivation.** Existing teleoperation approaches for **dexterous** (multi-fingered, humanlike-hand) bimanual manipulation each have a gap: leader-follower rigs (§6.2) are robot-specific and push all singularity/collision avoidance onto the human operator; earlier vision-based hand-tracking systems needed heavy GPU processing and were mostly single-arm only; and almost none of these systems give the operator any haptic (touch) feedback, or handle the noisy, self-occluded finger tracking that consumer headsets like Apple Vision Pro actually produce.

**Key idea/innovation.** Use a consumer VR headset purely as a markerless hand-pose *sensor*, and solve the human-to-robot motion mapping as a fast, real-time constrained optimization that simultaneously matches the robot's hand shape to the human's, stays smooth frame-to-frame, and automatically respects joint limits/avoids collisions and singularities — while feeding back cheap vibration-motor haptic cues to the operator's fingertips.

**Method & training procedure.** An Apple Vision Pro tracks the operator's two hands (no gloves or markers). Each frame, a constrained least-squares optimization (solved via **[Sequential Quadratic Programming](https://en.wikipedia.org/wiki/Sequential_quadratic_programming)**) finds robot joint angles that (a) place the robot hand's fingertips near the human's tracked fingertip positions, (b) penalize large frame-to-frame joint changes for smoothness, and (c) respect joint limits — with a reformulation that solves only over the robot hand's actively-driven joints (not its full, larger set of mechanically-coupled joints), reported as a 10× speedup. Hardware: two 7-DOF xArm arms, each fitted with a 6-DOF "Ability Hand." Haptic feedback: one $1.20 vibration motor per fingertip (10 total), worn via fabric fingertip sleeves, driven by a microcontroller reading contact-force sensors on the robot hand. Downstream policies (ACT, Diffusion Policy, and 3D Diffusion Policy) were trained on collected demonstrations for comparison.

**Results.** On a 10-task single-arm benchmark, Bunny-VisionPro matched or beat two prior teleoperation baselines on every task (e.g. 10/10 vs. 9/10 and 10/10 on one task; 9/10 vs. 3/10 and 7/10 on a harder two-cup-stacking task). On custom bimanual tasks it reported an 11% higher success rate with 45% less completion time than a strong baseline. Averaged across the three downstream policy architectures, training on Bunny-VisionPro-collected data gave a 22% success-rate improvement, with the haptic feedback specifically improving or maintaining success in 9 of 10 tested comparisons. The whole teleoperation loop ran above 60 Hz.

**How it connects.** Like ACE (§6.6), this system explicitly targets what it sees as leader-follower teleoperation's weak point (robot-specificity, no built-in safety) using a very different mechanism (markerless vision instead of a passive mechanical rig) — comparing the two is a useful exercise in seeing that "better teleoperation" has more than one defensible engineering answer.

### 6.5 Active, steerable viewpoints

A separate design axis from *how the operator's arms/hands are tracked* is *what the operator sees* while teleoperating — and a fixed camera view turns out to be a real bottleneck of its own.

### Paper Breakdown: Open-TeleVision ("Robot-TV")

*Citation: Xuxin Cheng, Jialong Li, Shiqi Yang, Ge Yang, Xiaolong Wang, "Open-TeleVision: Teleoperation with Immersive Active Visual Feedback," CoRL 2024.*

**Problem/motivation.** A standard fixed third-person camera bolted somewhere in the workspace inevitably gets occluded by the robot's own arms or body at exactly the moments precision matters most, so the operator can't be sure the demonstration actually captured what the downstream policy will need to see — silently degrading collected training data, not just making teleoperation feel harder.

**Key idea/innovation.** Mount a **stereo (two-eyed) camera on a small, actively-steerable robot "head,"** driven in real time by the operator's own VR headset head-tracking, so the operator can naturally look around and lean in for depth perception exactly as if their own eyes were physically at the robot's location, while their arm/hand motion is separately mirrored onto the robot's arms and hands.

**Method & training procedure.** The operator wears an Apple Vision Pro; a ZED Mini stereo camera sits on a 2–3-DOF actuated neck/gimbal mounted on the robot (tested on both a Unitree H1 and a Fourier GR-1 humanoid), whose motors are driven directly by the operator's tracked head pose, streaming stereo video back to the headset at 60 Hz. Wrist poses map to end-effector targets via closed-loop inverse kinematics; finger motions retarget to the robot's hand via an optimization-based retargeting library. Downstream policies used ACT with a pretrained DINOv2 vision-transformer backbone in place of the usual ResNet.

**Results.** Policy success rates of 87–100% across four long-horizon manipulation tasks (sorting, insertion, folding, unloading). Ablations make the paper's central claim directly testable: removing stereo vision (switching to a single monocular camera) dropped one task's pick success from 92% to 46% — roughly halving it. A **human** teleoperation user study (not just the trained policy) found 100% teleoperation success across all four tasks with stereo vision, versus dropping to 50–71% success on two tasks with monocular vision only — evidence that active stereo viewing helps the *human operator's* own performance, not only the downstream learned policy.

**How it connects.** This paper's core lesson generalizes past its specific hardware: what the demonstrator can *see* while teleoperating is itself part of the data-collection pipeline's design space, alongside how their motion is tracked (§6.4, §6.6) and what representation their motion is stored in (§6.2, §6.3).

### 6.6 Passive, cross-platform exoskeletons

A final teleoperation design point, sitting between the fully mechanical leader-follower rig (§6.2) and pure camera-based tracking (§6.4): a wearable mechanical linkage that is *not* motorized (it doesn't push back against the operator or hold their arm's weight) but is instrumented to measure the operator's own joint angles as they move naturally.

### Paper Breakdown: ACE (Cross-Platform Visual-Exoskeleton)

*Citation: Shiqi Yang, Minghuan Liu, Yuzhe Qin, Runyu Ding, Jialong Li, Xuxin Cheng, Ruihan Yang, Sha Yi, Xiaolong Wang, "ACE: A Cross-Platform and Visual-Exoskeletons System for Low-Cost Dexterous Teleoperation," CoRL 2024.*

**Problem/motivation.** Existing teleoperation interfaces generally have to trade off cost, ease of setup, and cross-robot reusability against each other — a system tuned to be cheap and easy to use is usually built or calibrated for one specific robot embodiment (§1) and needs hardware redesign to control a different arm, hand, or robot body plan.

**Key idea/innovation.** A single, cheap ($600), passive wearable mechanical arm — instrumented only with position-sensing servos (no motors pushing back) plus two ordinary wrist-mounted webcams for finger tracking — measures the operator's wrist pose via forward kinematics (§1) and finger pose via camera-based hand tracking, then maps that measured pose, through a general optimization-based retargeting layer, into the action space of essentially *any* downstream robot — the same physical rig reused across five very different robot configurations with no hardware changes, only a different software mapping per robot.

**Method & training procedure.** A 3D-printed, two-arm exoskeleton (7 links, 6 DOF per arm), each joint fitted with a Dynamixel servo used purely as a high-resolution angle *encoder* (not a motor) — the human's own arm moves the passive linkage, and the servos just report the resulting angles, from which wrist pose is computed via forward kinematics. Two low-cost webcams near the wrist, processed with the lightweight MediaPipe hand-tracking library, give 21 finger keypoints per hand for grip/finger control. The whole rig dons in under 30 seconds via magnetic connectors and adjusts to different arm lengths in under 2 minutes. Retargeting uses a linear scale-and-recenter formula for wrist position (rescaling the human's reachable workspace onto the robot's own, possibly differently-sized, workspace) and, for full dexterous hands, an optimization minimizing the gap between human and robot fingertip positions subject to the robot hand's joint limits. Validated across five robot platforms spanning a fixed arm with a simple gripper, two different humanoid-plus-dexterous-hand combinations, and a quadruped-mounted arm.

**Results.** Average forward-kinematics tracking error of about 1 mm, and 3 mm error in a controlled precision test. Head-to-head against GELLO (a prior low-cost, similarly passive rig): ACE was generally faster to reach targets and more successful (e.g. 97.1% vs. 47.6% success on one small-workspace condition), though not universally — GELLO edged out ACE on one medium-workspace condition, a genuine nuance rather than a clean sweep. Downstream imitation-learning tasks (3D Diffusion Policy or ACT, depending on platform) reached roughly 79–98% success across six tested tasks. Cost comparison the paper reports directly: ACE and GELLO both around $600, versus $20k for ALOHA (§6.2), $32k for Mobile ALOHA, and $4k for a comparable glove-based system (DexCap).

**How it connects.** ACE is the clearest illustration in this section of the "cost vs. cross-platform generality" trade-off named in its own motivation — it and Bunny-VisionPro (§6.4) both explicitly react against leader-follower teleoperation's robot-specificity, arriving at two different technical answers (passive mechanical sensing vs. markerless vision) to the same design question.

---

## 7. Multi-modal generative models for behavioral cloning

### 7.1 Why an ordinary regression policy can fail even with perfect data

Go back to §2's plain BC formulation: a network `π_θ(o) → a` trained with an ordinary regression loss (mean-squared error) to predict the expert's action. This works fine when, for a given observation, there's essentially one reasonable action to take. It breaks down when there are **multiple, genuinely different valid actions** for the same observation — for instance, an expert who sometimes goes left around an obstacle and sometimes goes right, both equally fine, depending on nothing visible in the current observation (maybe it depended on which way the demonstrator happened to be leaning at the time).

A regression loss like MSE is minimized by the **conditional mean** of the target distribution. If the training data contains both "go left" and "go right" for visually identical observations, MSE training pushes the network toward the *average* of those two actions — which, for steering around an obstacle, is "go straight into it": worse than either individually-valid choice. This is called **mode collapse/averaging**, and it's a structural limitation of any policy architecture built as a single continuous regression function — no amount of extra training data fixes it, because the problem is the *shape* of function the architecture can represent, not a lack of data.

### 7.2 Mixture Density Networks: predicting a distribution instead of one action

§7.1's problem is that a regression head can only emit one action. The most direct architectural fix — predating both of this section's other two approaches — is to change what the output layer predicts: instead of an action vector, predict the parameters of a **[Mixture Density Network (MDN)](https://en.wikipedia.org/wiki/Mixture_model)**, a Gaussian mixture distribution *over* actions. This idea (Bishop, 1994, for regression problems generally) is used here as a drop-in replacement for the final layer of an otherwise-ordinary policy network.

**Structure.** Keep the same trunk (a CNN or MLP encoder of the observation) as any other policy in this section, but replace the final action-prediction layer with three small output heads that together parameterize a Gaussian mixture with K components: (1) **K mixture weights** `π_1, ..., π_K`, produced through a softmax so they sum to 1 and read as "how likely this component is the right mode"; (2) **K mean vectors** `μ_1, ..., μ_K`, one candidate action per mode; (3) **K (co)variances** `Σ_1, ..., Σ_K`, one spread per mode (commonly diagonal, parameterized through something like an exponential or softplus so the output stays positive).

**Input, output, label, and loss.** Input is the same observation `o` any policy in this section takes. Output is the full set of mixture parameters `{π_k, μ_k, Σ_k}` for k = 1..K — not an action. The training label is still just the single expert action `a_expert` actually demonstrated: there is no "which mode" label anywhere in the data, since which mode a given expert action belongs to is entirely latent, inferred implicitly as a side effect of training. The loss is the negative log-likelihood of the expert action under the predicted mixture density, not MSE:

```
loss = −log( Σ_k π_k · N(a_expert; μ_k, Σ_k) )
```

Minimizing this pushes probability mass onto wherever expert actions actually land — including, if the data genuinely contains two separated clusters of valid actions, two different mixture components — rather than collapsing them to their average the way an MSE regression loss does.

**Inference.** At test time, either sample an action from the predicted mixture, or take the mean of whichever component has the largest weight `π_k` — a quick way to read off "the most likely mode" without sampling noise.

**Why it's innovative.** An MDN lets a single feedforward network express a genuinely multimodal output distribution — several distinct, separately-plausible actions for the same observation — instead of committing to one point estimate. This directly attacks §7.1's failure mode architecturally: rather than changing the loss's target or searching at inference time, it simply gives the output layer enough structure to describe more than one peak.

**Why it wasn't enough in practice.** A small, fixed K turns out to be brittle, in a way distinct from §7.1's failure mode. §7.1's problem is architectural: a single continuous regression function *cannot* represent two separated valid actions, no matter how it's trained. An MDN's output layer *can* represent them — but gradient descent on the NLL loss above doesn't reliably learn to spread responsibility evenly across all K components. Early in training, whichever component happens to land closest to a cluster of expert actions dominates the sum inside the log (its exponential term is largest), so it keeps absorbing most of the gradient signal and improving further, while the other components — contributing almost nothing to that sum — receive vanishingly little gradient and drift toward small, uninformative weights. The net effect is one or two "winning" modes capture the dominant behavior while the rest collapse toward near-zero weight, rather than the mixture partitioning the action space among genuinely different, comparably-weighted modes. This is a *training-dynamics* flavor of mode collapse — components starving each other of gradient signal during optimization — distinct from §7.1's *architectural* flavor, where the function class itself cannot represent multiple modes regardless of how it's trained.

This is exactly the failure mode behind the "LSTM-GMM" baseline named only by score in §7.4's Results paragraph below — "GMM" there is precisely this Gaussian-mixture output head, wrapped around an LSTM trunk instead of a feedforward one, and its weak results on multimodal tasks are a direct symptom of the mode-starvation problem just described. It's exactly this brittleness — a small, fixed number of modes that collapses under ordinary gradient training — that motivated the two alternatives covered next: §7.3's energy-based policies, which sidestep committing to any fixed number of modes by scoring the entire action space with one function evaluated many times at inference, and §7.4's diffusion policies, which represent multimodality implicitly through an iterative denoising process rather than any explicit, mode-counted parameterization.

### 7.3 Implicit, energy-based policies

### Paper Breakdown: Implicit Behavioral Cloning (IBC)

*Citation: Pete Florence, Corey Lynch, Andy Zeng, Oscar Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, Jonathan Tompson, "Implicit Behavioral Cloning," CoRL 2021.*

**Problem/motivation.** Exactly §7.1's mode-averaging failure: standard "explicit" policies, `â = F_θ(o)`, are continuous functions of the observation, and a continuous function *cannot* jump cleanly between two separated, both-valid actions or represent a genuine discontinuity — it necessarily interpolates through the invalid region in between.

**Key idea/innovation.** Instead of a network that directly *outputs* an action, train a network `E_θ(o, a)` — an **[energy-based model](https://en.wikipedia.org/wiki/Energy-based_model)** — that takes *both* the observation and a *candidate* action as input and outputs a single scalar "energy": low energy means "this action is a good match for this observation." At inference time, the chosen action is whichever candidate action **minimizes** this energy: `â = argmin over a of E_θ(o, a)`. Because this argmin operation is not itself required to be a continuous function of `o` (the paper proves this formally — any set-valued, discontinuous target mapping can be exactly represented as the argmin of some continuous scalar energy function, even though it cannot be represented by a single continuous direct-regression function), implicit policies can represent exactly the discontinuous, multi-valued mappings explicit regression cannot.

**Method & training procedure.** Architecturally, `E_θ(o, a)` is an entirely ordinary feedforward network: an embedding of the observation (a CNN trunk's feature vector for image observations, or the raw/encoded state vector for low-dimensional observations) is concatenated with the candidate action vector, and the combined vector is passed through a small MLP (or further convolutional layers, for image inputs) ending in a single scalar output unit — the energy. What's unusual about IBC is not this architecture (any standard encoder-plus-MLP stack works here) but the training objective and inference procedure built around it, described next. Training uses an **[InfoNCE](https://en.wikipedia.org/wiki/Contrastive_learning)**-style contrastive loss: for each (observation, expert-action) training pair, sample a batch of "negative" candidate actions (uniformly at random from the action space) and train the energy network so the true expert action gets *lower* energy than the negatives, via a softmax classification loss over (true action) vs. (negatives):

```
loss = −log[  exp(−E(o, a_expert))
             ─────────────────────────────────────────
             exp(−E(o, a_expert)) + Σ over negatives of exp(−E(o, a_negative))  ]
```

Since there's no closed-form way to directly compute `argmin_a E_θ(o,a)` for an arbitrary neural network, inference uses **[derivative-free optimization](https://en.wikipedia.org/wiki/Derivative-free_optimization)**: start with many randomly-sampled candidate actions, repeatedly re-weight and re-sample them toward the lowest-energy candidates (shrinking the sampling noise each round, much like the **[cross-entropy method](https://en.wikipedia.org/wiki/Cross-entropy_method)**), and return the best candidate after a few rounds — or, for higher-dimensional action spaces, a gradient-based **[Langevin-dynamics](https://en.wikipedia.org/wiki/Langevin_dynamics)** sampler that does use the energy function's gradient.

**Results.** On simulated block-pushing tasks, the implicit (energy-based) policy reached ~99–100% success across single-target, multi-target/multimodal, and pixel-observation variants, while an explicit MSE-trained policy dropped to 87–90% on the harder variants and a Gaussian-mixture baseline collapsed to 10% on pixel observations. On real-robot precision tasks the gap was much larger: on a tight 1mm-tolerance insertion task, implicit BC reached 83.3% success versus only 6.7% for explicit BC — roughly an order of magnitude better. A separate experiment varying action-space dimensionality with a fixed dataset size found implicit policies stayed at 95%+ success up to 16 action dimensions, versus only 8 for explicit policies — evidence implicit models also generalize better from limited data, not only handle multimodality.

**How it connects.** IBC's own training recipe turned out to have a subtle statistical flaw, fixed by the Ranking-NCE paper in §7.5; its "predict via argmin of a learned function" framing is also a conceptual cousin of the score/energy-gradient view of diffusion sampling introduced next.

### 7.4 Diffusion policies

### Paper Breakdown: Diffusion Policy

*Citation: Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song, "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," RSS 2023.*

**Problem/motivation.** IBC (§7.3) fixes mode-averaging but is finicky to train — its contrastive loss needs a working negative-sampling scheme to approximate an otherwise-intractable normalization term, and this negative sampling is a known source of instability, especially as action dimensionality grows. Simpler multimodal alternatives (Gaussian-mixture output heads) tend to collapse toward a single mode in practice rather than faithfully representing genuine multimodality.

**Key idea/innovation.** Formulate action prediction as a **[diffusion model](https://en.wikipedia.org/wiki/Diffusion_model)**: rather than an energy function evaluated once, train a network to iteratively **denoise** a sequence of future actions, starting from pure random noise, conditioned on recent observations — sidestepping IBC's negative-sampling instability (denoising is trained with an ordinary, stable regression loss) while still representing arbitrarily multimodal action distributions, because the iterative noisy refinement process can settle into different valid modes on different runs, the same way image-generating diffusion models produce varied outputs from the same text prompt.

**Method & training procedure.** Borrowing standard denoising-diffusion machinery: starting from Gaussian noise, a trained network `ε_θ` is applied repeatedly to progressively remove noise, converging on a clean sample. Here, "the sample" is a whole **chunk of future actions** (§6.1) rather than a single action, and the network is conditioned on a short history of recent observations (default: the last 2 steps) — critically, the observation conditioning is never itself noised; only the action chunk is. The chunking scheme uses three horizon parameters: an **observation horizon** (how much recent context the network sees, default 2 steps), a **prediction horizon** (how many future actions it predicts at once, default 16), and a shorter **execution horizon** (how many of those predicted actions actually get run before replanning, default 8) — this receding-horizon setup (§6.1) balances temporal smoothness against staying responsive to new observations; predicting too few actions ahead loses the smoothness benefit, predicting/committing too many makes the robot slow to react to surprises. The training loss is the standard diffusion objective, adapted to condition on observations:

```
loss = MSE( true_noise,  ε_θ(observation_history, noisy_action_chunk, diffusion_step) )
```

Two network variants were tested: a convolutional **[U-Net](https://en.wikipedia.org/wiki/U-Net)** (robust, works well "out of the box," but biased toward smooth/slow-changing action sequences) and a transformer (conditioned via cross-attention over the observation embeddings, better on fast-changing/complex tasks but more sensitive to tuning). At inference, **[DDIM](https://en.wikipedia.org/wiki/Diffusion_model#Denoising_Diffusion_Implicit_Model_%28DDIM%29)** sampling lets training use many denoising steps (100) while inference uses far fewer (as few as 10), keeping real-time latency low (~0.1s on a consumer GPU in their real-robot tests).

**Results.** Across 15 tasks spanning 4 benchmark suites (Robomimic, a 2D pushing task, a multimodal block-pushing task, and a long-horizon kitchen task), the paper reports an average 46.9% improvement over prior state-of-the-art baselines (an LSTM-GMM policy, IBC, and a Behavior Transformer). On the multimodal block-pushing stress test specifically, the transformer variant of Diffusion Policy reached 0.99/0.94 success on reaching the first/second target, dramatically ahead of LSTM-GMM (0.03/0.01) and IBC (0.01/0.00) — direct evidence the multimodality fix is doing real work, not just improving average performance. On a real-world pushing task, Diffusion Policy reached 95% success versus 20% for IBC and 0% for LSTM-GMM in the same setup. Ablations confirmed the action-horizon trade-off described above (too-short or too-long execution horizons both hurt) and found **position control** (predicting target end-effector positions) tolerates replanning delay much better than **velocity control**, tying back to §3.2's compounding-error framing.

**How it connects.** Diffusion Policy is the downstream policy architecture UMI (§6.3) actually trains on top of its embodiment-agnostic data, and its chunked, receding-horizon prediction scheme is a direct generalization of ACT's chunking idea (§6.2) to a different (diffusion-based, rather than CVAE-based) generative backbone.

### 7.5 Fixing implicit BC's hidden bias

### Paper Breakdown: Revisiting Energy-Based Models as Policies (Ranking NCE / Interpolating EBMs)

*Citation: Sumeet Singh, Stephen Tu, Vikas Sindhwani, arXiv:2309.05803, 2023 (published in Transactions on Machine Learning Research, 2024).*

**Problem/motivation.** Follow-up work using IBC (§7.3) repeatedly found its energy-based policies unstable and underperforming, without a clear mathematical explanation. This paper proves the reason is **not** merely "negative sampling is hard to tune" — it's that **IBC's training objective is statistically biased even with infinite data**, whenever the negative-action samples aren't drawn uniformly at random.

**Key idea/innovation.** IBC's contrastive loss is mathematically only unbiased for a *uniform* negative-sampling distribution — but uniform sampling is hopelessly inefficient in higher-dimensional action spaces (the vast majority of random samples are nowhere near anything useful), so practical implementations quietly need a smarter, non-uniform proposal distribution, which then makes the learned energy function converge to the wrong quantity: not the true action distribution `p(a|o)`, but the *ratio* `p(a|o) / (proposal distribution)`. The fix ("Ranking NCE," adapting an idea from Ma & Collins 2018): add an explicit correction term for the known density of the negative-sampling proposal directly into the same softmax-style loss, which restores an unbiased estimate for **any** proposal distribution, uniform or not — enabling a second, *learned* generative model to serve as an efficient proposal/negative-sampler, trained jointly but non-adversarially (via its own separate maximum-likelihood objective, not a GAN-style minimax game against the energy model).

**Method & training procedure.** The corrected loss (contrast this against §7.3's IBC loss — the only difference is the `− log(proposal density)` correction term subtracted inside each exponential):

```
loss = −log[  exp(E(o,a_true) − log p_proposal(a_true|o))
             ─────────────────────────────────────────────────────────────────
             Σ over {a_true, negatives} of exp(E(o,a) − log p_proposal(a|o))  ]
```

The paper proves this is unbiased regardless of which proposal distribution is used (whereas plain IBC is only unbiased for a uniform one), and separately proves it approaches the statistical efficiency of full maximum-likelihood estimation as the number of negative samples grows. Training alternates: fit the proposal (a **[normalizing-flow](https://en.wikipedia.org/wiki/Flow-based_generative_model)**-style generative model) via its own likelihood objective on the data, then fit the energy model via the corrected Ranking-NCE loss using fresh negatives drawn from that proposal — no adversarial min-max game, unlike some prior attempts to jointly train an EBM with a learned sampler. A second contribution, "**Interpolating Energy Models**," borrows the *multi-noise-scale* training idea that makes diffusion models (§7.4) work well: rather than one energy function, train a single energy function indexed by a continuous noise-scale variable (mirroring a diffusion process's interpolation between pure noise and the true action distribution), letting inference start cheaply from the easy, near-Gaussian noise-scale end and only invoke the more expensive learned energy landscape as it approaches the true, clean-data end.

**Results.** On synthetic 2D distributions, plain IBC's sample quality visibly lagged a normalizing flow, ordinary diffusion, and Ranking NCE (**[Bhattacharyya-coefficient](https://en.wikipedia.org/wiki/Bhattacharyya_distance)** scores of 0.54–0.87 for IBC vs. 0.99+ for the others) — direct confirmation of the predicted bias. On a simulated obstacle-avoidance path-planning benchmark, Ranking NCE achieved both the lowest collision rate and lowest path cost of every method tested, including diffusion — notably, an ablation ranking multiple candidate action samples at inference time and keeping the best (something the energy/likelihood-based methods can do naturally, but a plain diffusion or CVAE policy structurally cannot do as reliably) showed sizeable gains, underscoring a genuine practical advantage of the energy/likelihood framing. On the same contact-rich Push-T manipulation benchmark used to evaluate Diffusion Policy (§7.4), the **Interpolating** variant of Ranking NCE outperformed plain (non-interpolating) Ranking NCE, ordinary diffusion, and a normalizing flow alike — the one task in the paper's results where the multi-noise-scale machinery was shown to be necessary, not just helpful.

**How it connects.** This paper closes the loop opened in §7.3: it doesn't propose a new *kind* of policy so much as repair the training procedure originally proposed for implicit/energy-based policies, and shows that once repaired, energy-based methods are competitive with (and on some metrics, ahead of) the diffusion-based approach from §7.4 — while additionally being able to *rank* multiple candidate actions at inference time in a way diffusion and CVAE-based policies (§6.2, §7.4) cannot as naturally.

---

## 8. Appendix: uncertainty, and querying an expert only when necessary

Two shorter ideas from the lecture, included here at the depth the lecture itself gave them (neither is a required-reading paper this week, so no Paper Breakdown).

**Two kinds of uncertainty.** When a learned policy is unsure what to do, it's worth distinguishing *why*:
- **[Epistemic uncertainty](https://en.wikipedia.org/wiki/Uncertainty_quantification)** ("model uncertainty") comes from not having seen enough data — e.g. flipping an unfamiliar biased coin a few times and being unsure of its true bias. This kind of uncertainty **shrinks toward zero as more data arrives**, in principle vanishing entirely with infinite data.
- **[Aleatoric uncertainty](https://en.wikipedia.org/wiki/Uncertainty_quantification)** ("observation noise") is irreducible: even knowing a coin's *exact* bias precisely, you still can't predict any single flip's outcome. No amount of additional data removes this kind of uncertainty — it's intrinsic to the process being modeled, not a gap in what's been learned.

Practically, estimating a neural network's epistemic uncertainty usually means approximating a distribution over the network's *weights* (rather than a single fixed weight setting) — via an ensemble of independently-trained networks, an approximate Bayesian technique like Monte Carlo dropout, or full Markov-chain sampling over weights — and asking how much the network's predictions vary across that distribution.

**Querying an expert only when necessary.** DAgger (§4) assumes the expert is always available to label every visited state, every iteration — expensive in practice. A family of methods instead tries to query the expert only when the learner's own uncertainty is high (e.g. an ensemble of learned policies disagreeing sharply — "DropoutDAgger" — or the current state sitting suspiciously close to a decision boundary — "SHIV," "SafeDAgger"), aiming to get most of DAgger's covariate-shift fix while asking the human for far fewer labels overall.

---

# Part B — Glossary

Moved to the project's running, cumulative glossary so terminology stays in one place across all weeks: see [`glossary.md`](./glossary.md).

---

## Quick self-check

If you can explain, in your own words, **why plain behavioral cloning's error grows quadratically with task horizon while DAgger's grows only linearly**, using only the terms "covariate shift," "no-regret online learning," and "recoverability" — and separately explain **why an ordinary regression policy fails on multimodal demonstrations, and name two different fixes** (energy-based/implicit policies, and diffusion policies) **and what specifically distinguishes them** — you've understood the core of Week 1.
