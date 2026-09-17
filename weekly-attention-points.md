# Weekly Attention Points (Professor's Checklist)

This file is the verbatim "Things to pay attention to" checklist the professor circulated for
CSC2626, one entry per lecture week. It is a required source alongside the lecture slides and
readings — see `CLAUDE.md`'s "Professor's weekly attention-points checklist" section for how it's
used when writing or auditing a week's notes.

The original is a Google Doc that could not be reliably re-fetched by tooling (it requires Google
sign-in, and automated fetching only returns a paraphrased view rather than the exact text), so
this file is the durable, checked-in copy of what the user pasted directly. If the professor
updates the source doc during the term, this file is updated by re-pasting the new text — it is
not auto-synced.

## Week 1: Imitation Learning vs Behavior Cloning

Things to pay attention to:
- The definition of behavior cloning (supervised learning)
- Common problems in behavior cloning: dataset shift, need for data augmentation, cautious
  demonstrators covering small parts of the state space
- Training networks that map pixels -> steering vs. networks that map pixels -> intermediate
  representation -> steering
- The difference between DAgger and behavior cloning
- DAgger's guarantees on compounding error as a linear function of the decision/planning horizon
- The difficulty of applying DAgger annotations to real robotic scenarios
- Tighter guarantees on behavioral cloning than DAgger
- Implicit behavioral cloning
- Diffusion policy

## Week 2: Introduction to Optimal Control & Model-Based RL

Things to pay attention to:
- The definition of value function V(s), state-action value function Q(s,a), cost-to-go J(x)
- The requirements of LQR optimization problems on the dynamics and cost function
- The LQR policy is linear in the state (or error of the state) that we want to drive to zero
- The LQR policy is time-dependent
- We can optimally solve LQR in closed form using dynamic programming (backward pass to compute
  control gains and forward pass to compute controls)
- Iterative LQR needs special care to use the Taylor expansion of the nonlinear dynamics and cost
  function, so that it satisfies LQR requirements
- Open-loop control vs control with state feedback
- To fit dynamics models we can minimize one-step prediction errors, but also multi-step
  prediction errors

## Week 3: Offline/Batch RL

Things to pay attention to:
- Terminology of RL methods
- The difference between offline RL and behavior cloning
- QT-Opt is roughly a continuous version of Q-Learning with function approximation
- The main problem with offline RL: querying the Q-value or evaluating the policy outside of the
  training distribution
- Addressing this issue with policy constraints or value conservatism
- The CQL objectives

## Week 4: Imitation Learners Guided by Optimal Control Experts and Physics

Things to pay attention to:
- Learning a policy from locally optimal controllers (Guided Policy Search)
- The objective of discounted reward + entropy(trajectory distribution) in Guided Policy Search
- Learn a policy from another policy or optimal controller that has access to privileged
  information
- Dynamic movement primitives
- DeepMimic and combining BC and RL
- Expert iteration and the Dual Policy Iteration paper (and how it subsumes all expert iteration
  methods as special cases)

## Week 5: Inverse Reinforcement Learning

Things to pay attention to:
- The maximum entropy formulation of linear reward learning
- Is computing the partition function necessary?
- Learning from preferences
- Actively selecting which scenarios to show during preference learning
- Boltzman-type distributions and how the gradient of the likelihood requires computing an
  expectation
- Importance sampling and its drawbacks
- Guided Policy Search produces a distribution of trajectories that is good for estimating the
  partition function in Guided Cost Learning through importance sampling. "Good" in the sense that
  it minimizes the KL divergence w.r.t. the Boltzman distribution that is hard to sample from.

## Week 6: Imitation as Program Induction and Modular Decomposition of Demonstrations

Things to pay attention to:
- The motivation behind compositionality: reusing/re-combining sub-behaviors in ways that didn't
  exist in the original training set
- NPI requires full program execution traces (known sub-task boundaries, ids, and arguments), not
  just input and output sequences
- NTP for few-shot imitation learning from video and its relationship to NPI
- Task sketches as weak supervision, but unknown task boundaries
- Learning task boundaries as well as the library of motion primitives that generates each
  hypothesized segment in the trajectory

## Week 7: Adversarial Imitation Learning

Things to pay attention to:
- Trajectory matching as an objective function in GAIL
- How to infer different demonstrated behaviors in imitation learning
- Model-based optimization as a way to enable end-to-end training of adversarial imitation using
  backprop

## Week 8: Reading Week

Reading Week - No lectures or office hours.

## Week 9: Shared Autonomy and Human-in-the-Loop Learning

Things to pay attention to:
- The difference between an MDP, POMDP, and QMDP
- How to incorporate known and unknown reward components
- The difference between forward and backward kinematics
- How to encourage smooth motion
- The difference between collocation and shooting methods in optimal control

## Week 10: Imitation learning from videos

Things to pay attention to:
- Causal confusion in IL
- Retargeting
- Keypoint tracking

## Week 11: Representation learning for imitation and safe IL

Things to pay attention to: (none listed in the source doc as of this writing)

## Week 12: TBA

No checklist available yet.

## Week 13: Project presentations

No checklist listed (project presentations).
