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
- **[Teach-and-repeat navigation](https://en.wikipedia.org/wiki/Robot_navigation)** — a robot is driven once along a route, building a map tied to that specific route, then repeats it by matching its current sensor view against the stored map; unlike BC, there is no trained function approximator and no generalization beyond the one taught route.
- **[Covariate shift](https://en.wikipedia.org/wiki/Imitation_learning)** — the mismatch between the state distribution a policy trains on (the expert's) and the state distribution it actually visits at test time (its own, shaped by its own errors); the root cause of imitation learning's compounding-error problem.
- **[Catastrophic forgetting](https://en.wikipedia.org/wiki/Catastrophic_interference)** — a network trained incrementally on a live data stream overwriting what it previously learned as new data arrives; mitigated with a replay buffer.
- **Recoverability (μ) / recovery cost (u)** — constants bounding how bad a single wrong action can be (how hard it is to recover from); small values mean mistakes are forgivable, large values mean a single error can be catastrophic — these constants set how severely covariate shift's error-compounding actually bites in a given task.
- **[No-regret online learning](https://en.wikipedia.org/wiki/Online_machine_learning) / regret** — an online algorithm has no regret if its average performance gap versus the single best fixed choice in hindsight shrinks to zero over time; the theoretical property DAgger's guarantee is built on.
- **DAgger (Dataset Aggregation)** — the iterative imitation-learning algorithm that fixes covariate shift by rolling out the current policy, having the expert label the states it visits, and retraining on the ever-growing aggregated dataset.
- **[Maximum Mean Discrepancy (MMD)](https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions)** — a way to measure how different two collections of samples are without needing either one's probability density function, by comparing sample averages of a rich family of functions; used by MMD-IL to decide when the learner's own visited states have drifted too far from the expert's demonstrated ones to trust the current policy.
- **Query-when-necessary DAgger variants (SHIV, DropoutDAgger, SafeDAgger)** — a family of DAgger modifications that query the expert only on states flagged "risky" instead of on every visited state: SHIV via a one-class SVM novelty/misclassification detector, DropoutDAgger via Monte Carlo dropout uncertainty, SafeDAgger via a separately-trained classifier predicting disagreement with the expert.
- **[One-class SVM](https://en.wikipedia.org/wiki/Support_vector_machine)** — a support vector machine trained on only "normal" examples (no negative class) that draws a boundary around them and flags anything falling well outside it as an outlier; used by SHIV to detect states unlike anything reliably handled so far.
- **[Monte Carlo dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks))** — running the same trained network several times on the same input with a different random dropout mask each time, and treating how much the outputs disagree as an approximate estimate of the network's own epistemic uncertainty.
- **Consistent surrogate loss / learning to defer** — a loss function that is easier to optimize than the true decision objective (e.g. "should I predict or defer to an expert?") but is proven to have the same optimal solution, so optimizing it is not a compromise; used to jointly train a classifier and a "rejector" deciding when to defer.
- **Eluder dimension** — a complexity measure for a space of possible models: the longest sequence of inputs on which some model in the class can keep looking correct on all earlier inputs yet be surprisingly wrong on the next one; bounds how many queries a noisy-expert imitation-learning algorithm needs before its uncertainty genuinely runs out.
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

## Week 2 — Optimal Control and Model-Based Reinforcement Learning
*(full explanations in [`week2-study-notes.md`](./week2-study-notes.md); this is the lookup-speed version)*

**Bellman-equation / control-vs-RL vocabulary:**
- **Dynamics / transition function** — the function (or, if stochastic, distribution) describing how a system's state evolves in response to an action, `x_{t+1} = f(x_t, u_t)`; control theory's word for what Week 1's MDP called the transition.
- **Cost vs. reward** — control theory's cost `c(x,u)` (to be minimized) and RL's reward `r(x,u)` (to be maximized) are literal negatives of each other, `c = -r`, not two different ideas.
- **Control law vs. policy-as-distribution** — disambiguating "policy": control theory's control law `u_t = π_t(x_t)` is a deterministic function of state (what LQR/iLQR compute); RL's policy `π_θ(a_t|x_t)` is a full probability distribution an action is sampled from.
- **[Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation)** — named after Richard Bellman, the recursive equation expressing the optimal cost-to-go/value at a state as the immediate cost/reward plus the optimal cost-to-go/value at the next state; built on Bellman's "principle of optimality" — that the remaining decisions of an optimal plan must themselves be optimal for the sub-problem that remains, letting a multi-step optimization be solved backward one step at a time.
- **Cost-to-go / value function** — two names (control theory's and RL's, respectively) for the same recursively-defined quantity: the minimum possible total future cost (or maximum future reward) achievable from a given state onward; "cost-to-go" is literal — cost still to be paid going forward, not cost already sunk.
- **[Q-function / state-action value function](https://en.wikipedia.org/wiki/Q-learning)** — the cost-to-go/value of committing to one specific action right now, then acting optimally (or per a given policy) from then on.

**LQR, iLQR, and MPC vocabulary:**
- **[Linear Quadratic Regulator (LQR)](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator)** — the exactly-solvable special case of optimal control with linear dynamics and quadratic cost; named literally (linear dynamics, quadratic cost, a "regulator" — classical control's term for a controller that drives/holds a system at a setpoint), solved via a backward Riccati recursion with no training data or iterative numerical optimization needed.
- **[Riccati recursion](https://en.wikipedia.org/wiki/Algebraic_Riccati_equation)** — the backward recursive computation (named after 18th-century mathematician Jacopo Riccati) producing LQR's optimal linear controller gains, exploiting the fact that a quadratic cost-to-go stays quadratic after every backward step.
- **Linear Quadratic Gaussian (LQG)** — LQR with added Gaussian process noise in the dynamics; the optimal control law keeps the same linear form as long as the true state is observed at each step.
- **[Iterative LQR (iLQR)](https://en.wikipedia.org/wiki/Iterative_LQR)** — extends LQR to nonlinear dynamics by repeatedly linearizing around a nominal (reference) trajectory, solving the resulting local LQR problem, and updating the trajectory — similar in spirit to Newton's method.
- **[Trust region](https://en.wikipedia.org/wiki/Trust_region)** — a penalty or constraint limiting how far an optimization step can move from the point a local approximation (e.g. iLQR's linearization) was built at, since that approximation is only trustworthy nearby.
- **Open-loop vs. closed-loop control** — open-loop executes a pre-computed action sequence with no feedback; closed-loop re-observes the true state and adjusts, correcting for model errors or disturbances.
- **[Model Predictive Control (MPC) / receding-horizon control](https://en.wikipedia.org/wiki/Model_predictive_control)** — closed-loop LQR/iLQR: solve for a full action sequence, execute only the first action, observe the new state, and repeat; the planning window's endpoint keeps "receding" into the future as you go (already previewed informally in Week 1's action-chunking entry — Week 2 derives it explicitly).
- **Warm start** — initializing an optimization (e.g. MPC's next replan) from the previous solution rather than from scratch, since consecutive replanning problems are nearly identical.

**Model learning and model-based RL vocabulary:**
- **Dynamics model / world model** — a function `f_θ` fit, via ordinary supervised learning exactly like Week 1's behavioral cloning but regressing next-state instead of action, to approximate how a system evolves; "world model" is the deep-RL/generative-modeling community's name for the same object, emphasizing that it captures how the external world evolves rather than mapping observations directly to actions.
- **Model-based reinforcement learning** — the loop of collecting data, fitting a dynamics model, planning through that model (via MPC/iLQR/MPPI/CEM), executing the result, and appending the new data before repeating.
- **Differentiable simulator** — a physics simulator engineered so that gradients of its outputs with respect to its inputs (states, actions, parameters) can be computed exactly, letting gradient-based planners like iLQR run through it even for contact-rich dynamics.
- **Model Predictive Path Integral control (MPPI)** — a derivative-free planner that samples many randomly-perturbed action sequences, rolls each through the model, and averages them weighted by a softmin (a smooth, temperature-controlled version of "keep only the best," structurally identical to a softmax over negative cost); named for a statistical-mechanics reformulation expressing the optimal control law as an expectation over trajectories (a "path integral").
- **[Cross-Entropy Method (CEM)](https://en.wikipedia.org/wiki/Cross-entropy_method)** — a derivative-free planner that samples action sequences from a Gaussian, keeps the best-scoring "elite" fraction, and refits the Gaussian to them; refitting to the elites is a maximum-likelihood fit, mathematically equivalent to minimizing cross-entropy between the elites' empirical distribution and the fitted Gaussian — where the method's name comes from.
- **Elites** — the top-performing fraction of sampled candidates kept at each iteration; a term borrowed from evolutionary computation/genetic algorithms, not CEM-specific jargon.
- **Monotonic policy improvement** — a guarantee that each iteration of model-fitting-and-planning provably improves (or at least does not worsen) a computable bound on true performance, even though the underlying model is imperfect.
- **Value-aware model learning** — fitting a dynamics model to be accurate specifically where it matters for choosing good actions, rather than uniformly accurate everywhere, since ordinary maximum-likelihood model fitting can be misaligned with what control actually needs.

---

## Weeks 3–13 — Forward-looking preview

*(seeded from the course schedule at https://csc2626.github.io/2026F_website/#schedule; terms get
promoted into their own dated `## Week N` section, with full beginner definitions, once that
week's notes are actually written)*

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
