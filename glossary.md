# CSC2626 — Running Glossary

A single, cumulative glossary that grows week by week as the course progresses. Each term: a
one-sentence, beginner-level definition plus which week it belongs to, so this file doubles as a
map of "what I've learned so far" and "what's coming."

**How this file is maintained:** every time a new week's study notes are written, add that week's
*actually-covered* terms to a `## Week N` section here (promoted out of the "forward-looking"
preview if they were already listed below), and trim the forward-looking preview under later weeks
once they're no longer speculative. Don't delete a term once it's here — later weeks may deepen a
definition, but the original stays as a beginner anchor.

---

## Week 1 — Imitation Learning vs. Supervised Learning
*(full explanations in [`week1-study-notes.md`](./week1-study-notes.md); this is the lookup-speed version)*

**Core imitation-learning vocabulary:**
- **[Robot policy](https://en.wikipedia.org/wiki/Robot_control)** — the function that maps a robot's current observation to the action it should take next; "training a policy" means fitting this function's parameters.
- **Expert / demonstrator** — whoever or whatever generated the training data a policy is trained to imitate (a human teleoperator, a planning algorithm, etc.).
- **Demonstration / trajectory** — one recorded run of (observation, action) pairs from start to finish of a task.
- **[Markov Decision Process (MDP)](https://en.wikipedia.org/wiki/Markov_decision_process)** — the standard state/action/transition/reward model this course's control and RL material sits on.
- **Rollout** — one full run of a policy through an MDP, start to finish.
- **Horizon (H or T)** — the number of decision steps in one rollout/episode.
- **[Behavioral cloning (BC)](https://en.wikipedia.org/wiki/Imitation_learning)** — training a policy to imitate expert (observation, action) pairs via ordinary supervised learning, with no environment interaction or reward signal.
- **[Covariate shift](https://en.wikipedia.org/wiki/Imitation_learning)** — the mismatch between the state distribution a policy trains on (the expert's) and the state distribution it actually visits at test time (its own, shaped by its own errors); the root cause of imitation learning's compounding-error problem.
- **[Catastrophic forgetting](https://en.wikipedia.org/wiki/Catastrophic_interference)** — a network trained incrementally on a live data stream overwriting what it previously learned as new data arrives; mitigated with a replay buffer.
- **Recoverability (μ) / recovery cost (u)** — constants bounding how bad a single wrong action can be (how hard it is to recover from); small values mean mistakes are forgivable, large values mean a single error can be catastrophic — these constants set how severely covariate shift's error-compounding actually bites in a given task.
- **[No-regret online learning](https://en.wikipedia.org/wiki/Online_machine_learning) / regret** — an online algorithm has no regret if its average performance gap versus the single best fixed choice in hindsight shrinks to zero over time; the theoretical property DAgger's guarantee is built on.
- **DAgger (Dataset Aggregation)** — the iterative imitation-learning algorithm that fixes covariate shift by rolling out the current policy, having the expert label the states it visits, and retraining on the ever-growing aggregated dataset.
- **Action chunking** — predicting a short sequence of several future actions in one forward pass (instead of one action at a time), typically executing only the first few before re-planning; shortens the effective decision horizon and smooths over noisy demonstrations.
- **[Receding-horizon control](https://en.wikipedia.org/wiki/Model_predictive_control)** — planning further ahead than you commit to, and re-planning repeatedly as new observations arrive; the control-theory pattern underlying action chunking.
- **[Mixture Density Network (MDN)](https://en.wikipedia.org/wiki/Mixture_model)** — a policy whose output layer predicts the parameters of a Gaussian mixture (mixture weights, means, covariances) over actions instead of a single action vector; lets one network express several distinct modes, but a small, fixed number of components often collapses during training onto just one or two of them.
- **[Energy-based model (EBM)](https://en.wikipedia.org/wiki/Energy-based_model)** — a model that scores (observation, action) pairs with a scalar "energy" rather than directly outputting an action; the best action is found by searching for the lowest-energy candidate.
- **Negative sampling / contrastive ([InfoNCE](https://en.wikipedia.org/wiki/Contrastive_learning)) loss** — a way to train an energy-based model without ever computing a true probability distribution: contrast the real (positive) action against several randomly-sampled decoy (negative) actions, and train the network to score the real one lower-energy than every decoy, via ordinary softmax/cross-entropy over the K+1 candidates.
- **[Diffusion model](https://en.wikipedia.org/wiki/Diffusion_model)** — a generative model that produces samples (here, action chunks) by iteratively denoising from random noise, conditioned on context (here, recent observations).
- **[Autoencoder](https://en.wikipedia.org/wiki/Autoencoder) / [variational autoencoder (VAE)](https://en.wikipedia.org/wiki/Variational_autoencoder) / conditional VAE (CVAE)** — a network pair (encoder compresses input to a small "bottleneck" vector, decoder reconstructs it) trained by reconstruction loss; variational makes the bottleneck a sampled distribution instead of one fixed vector, so new, valid outputs can be sampled; conditional also feeds the decoder side information (here, the observation), so it reconstructs the right output for that specific situation. ACT's policy is trained as a CVAE decoder.
- **Mode averaging / mode collapse** — the failure of an ordinary regression policy to represent multiple, equally-valid actions for the same observation; it instead outputs their average, which may be invalid.
- **[Epistemic uncertainty](https://en.wikipedia.org/wiki/Uncertainty_quantification)** — uncertainty from insufficient data, which shrinks toward zero as more data is collected.
- **[Aleatoric uncertainty](https://en.wikipedia.org/wiki/Uncertainty_quantification)** — irreducible uncertainty from the process itself, unaffected by collecting more data.

**Basic robotics/hardware vocabulary:**
- **[Manipulator / robot arm](https://en.wikipedia.org/wiki/Robotic_arm)** — a chain of rigid links connected by motorized joints, ending in a tool.
- **[End-effector](https://en.wikipedia.org/wiki/Robot_end_effector)** — whatever is mounted at the tip of a manipulator to interact with the world (a hand, gripper, or tool).
- **[Gripper](https://en.wikipedia.org/wiki/Robotic_gripper)** — a simple end-effector, usually two opposing fingers that open and close.
- **[Degrees of freedom (DOF)](https://en.wikipedia.org/wiki/Degrees_of_freedom_(mechanics))** — the number of independent ways a mechanism can move.
- **[Forward / inverse kinematics](https://en.wikipedia.org/wiki/Robot_kinematics)** — forward: computing end-effector position from joint angles; inverse: solving for the joint angles that reach a desired end-effector position.
- **Joint space vs. task space** — describing a robot's configuration by its own joint angles (joint space) versus by where its end-effector sits in the world (task space).
- **[Workspace](https://en.wikipedia.org/wiki/Robot_kinematics)** — the full set of positions an end-effector can physically reach.
- **[Proprioception](https://en.wikipedia.org/wiki/Proprioception)** — a robot's internal sense of its own joint angles/velocities, as opposed to external sensing like vision.
- **[RGB-D camera](https://en.wikipedia.org/wiki/RGB-D_camera)** — a camera producing an aligned color image plus a per-pixel depth map.
- **[Telerobotics / teleoperation](https://en.wikipedia.org/wiki/Telerobotics)** — real-time human control of a robot, the standard way manipulation demonstrations are collected.
- **Kinesthetic demonstration** — physically guiding a robot's arm by hand while it records its own joint sensors, as opposed to teleoperating it through a separate controller device.
- **Embodiment** — the specific physical robot body (kinematics, sensors, actuators) a policy controls; a recurring question is whether data/policies transfer across different embodiments.
- **Bimanual manipulation** — manipulation tasks or systems using two coordinated robot arms.
- **Leader-follower teleoperation** — a passive, scaled-down mechanical duplicate of a robot arm that a human moves, with its joint readings sent as commands to the real ("follower") arm — used by ALOHA.

---

## Weeks 2–13 — Forward-looking preview

*(seeded from the course schedule at https://csc2626.github.io/2026F_website/#schedule; terms get
promoted into their own dated `## Week N` section, with full beginner definitions, once that
week's notes are actually written)*

- **Week 2 — Intro to optimal control & RL**: [Linear Quadratic Regulator (LQR)](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator), Iterative LQR (iLQR), [Model Predictive Control (MPC)](https://en.wikipedia.org/wiki/Model_predictive_control).
- **Week 3 — Offline/batch reinforcement learning**: Conservative Q-Learning (CQL), IQ-Learn, [D4RL](https://en.wikipedia.org/wiki/Reinforcement_learning) benchmark suite.
- **Week 4 — Imitation learning combined with RL & planning**: guided policy search, planning with diffusion, expert iteration, [dynamic movement primitives](https://en.wikipedia.org/wiki/Dynamic_movement_primitives).
- **Week 5 — Imitation as program induction**: neural programmer-interpreters, Neural Task Programming, TACO, hierarchical task learning.
- **Week 6 — Inverse reinforcement learning**: maximum entropy [inverse reinforcement learning](https://en.wikipedia.org/wiki/Inverse_reinforcement_learning), guided cost learning, Bayesian IRL, preference learning, value alignment.
- **Week 7 — Adversarial imitation learning**: [GAIL](https://en.wikipedia.org/wiki/Generative_adversarial_network) (Generative Adversarial Imitation Learning), InfoGAIL, the divergence-minimization view of imitation learning.
- **Week 8 — Fall Reading Week**: no lecture.
- **Week 9 — Shared autonomy**: RelaxedIK, shared autonomy via deep RL, hindsight optimization, human-in-the-loop learning.
- **Week 10 — Imitation learning from videos**: K-VIL, Track2Act, VideoDex, causal confusion in imitation learning.
- **Week 11 — Representation learning for imitation**: generalization guarantees, contrastive Fourier features, TRAIL, self-supervised correspondence.
- **Week 12 — TBA**.
- **Week 13 — Final project presentations**.
