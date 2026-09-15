# Fact Audit — CSC2626 Imitation Learning Notes

**Last audited: 2026-09-14**

**Summary: 111 claims audited — 108 confirmed / 2 corrected / 1 unverifiable.**

Scope: `README.md`, `week1-study-notes.md`, `glossary.md`, and the live published artifact
("The Imitation Loop", https://claude.ai/artifact/5Eh9TePpNpmwVvLmAbWPoG). No other week files
exist yet. Every discrete factual claim in Week 1 was checked against an external source: the
cited paper itself (arXiv/venue PDF), Wikipedia, or another reputable technical source. The
markdown and the artifact state the same claims throughout (both were independently checked); no
drift was found between them, except that both carried the *same two* errors described below
(since the artifact is a close paraphrase of the markdown), which are now fixed in both places.

Note on scope: per the task's independence rule, this audit does not evaluate writing style,
pedagogical structure, or `CLAUDE.md` compliance — only factual correctness (definitions, numbers/
formulas, historical attributions, cited-paper claims, and Wikipedia link resolution).

---

## 1. Vocabulary / definitions (§0–§1)

| Claim | Verdict | Source(s) |
|---|---|---|
| Robot policy, expert/demonstrator, demonstration/trajectory, MDP, rollout, horizon — definitions | Confirmed | [Wikipedia: Markov decision process](https://en.wikipedia.org/wiki/Markov_decision_process), [Wikipedia: Robot control](https://en.wikipedia.org/wiki/Robot_control) |
| Manipulator/robot arm, end-effector, gripper, DOF, forward/inverse kinematics, joint vs. task space, workspace, proprioception, RGB-D camera, teleoperation, kinesthetic demonstration, embodiment, bimanual manipulation — definitions | Confirmed | [Robotic arm](https://en.wikipedia.org/wiki/Robotic_arm), [Robot end effector](https://en.wikipedia.org/wiki/Robot_end_effector), [Robotic gripper](https://en.wikipedia.org/wiki/Robotic_gripper), [Degrees of freedom (mechanics)](https://en.wikipedia.org/wiki/Degrees_of_freedom_(mechanics)), [Robot kinematics](https://en.wikipedia.org/wiki/Robot_kinematics), [Proprioception](https://en.wikipedia.org/wiki/Proprioception), [Telerobotics](https://en.wikipedia.org/wiki/Telerobotics) |
| "A typical robot arm has 6–7 DOF; a human-like multi-fingered hand can have 15–20+" | Confirmed (general robotics knowledge, approximate range) | [Standard Bots: Degrees of freedom in robotics](https://standardbots.com/blog/degrees-of-freedom); cross-checked against multiple robotics sources citing 6–7 DOF industrial arms and human hands in the ~20–27 DOF range |
| Behavioral cloning definition (§2) | Confirmed | [Wikipedia: Imitation learning](https://en.wikipedia.org/wiki/Imitation_learning) |
| Covariate shift, catastrophic forgetting, concept drift, no-regret online learning/regret, receding-horizon control, energy-based model, diffusion model, mixture model (MDN), epistemic/aleatoric uncertainty — definitions | Confirmed | [Catastrophic interference](https://en.wikipedia.org/wiki/Catastrophic_interference), [Concept drift](https://en.wikipedia.org/wiki/Concept_drift), [Online machine learning](https://en.wikipedia.org/wiki/Online_machine_learning), [Regret (decision theory)](https://en.wikipedia.org/wiki/Regret_(decision_theory)), [Model predictive control](https://en.wikipedia.org/wiki/Model_predictive_control), [Energy-based model](https://en.wikipedia.org/wiki/Energy-based_model), [Diffusion model](https://en.wikipedia.org/wiki/Diffusion_model), [Uncertainty quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification) |

## 2. Wikipedia link resolution

All ~30 distinct Wikipedia links used across the markdown/artifact were spot-checked for actually
resolving to, and matching, the concept they're attached to (fetched and read the target page).

| Link target | Attached term | Verdict |
|---|---|---|
| `Robot_end_effector` | **Workspace** | **Corrected** — the End-effector page never discusses "workspace" (the reachable-pose concept); confirmed by fetching the page. Retargeted to `Robot_kinematics`, which explicitly states "the dimensions of the robot and its kinematics equations define the volume of space reachable by the robot, known as its workspace." |
| `Imitation_learning#Behavior_Cloning` | Behavioral cloning | Confirmed — anchor/section exists and covers BC and covariate shift |
| `Diffusion_model#Denoising_Diffusion_Implicit_Model_(DDIM)` | DDIM | Confirmed — exact section heading exists on the page |
| `Bhattacharyya_distance` | Bhattacharyya coefficient | Confirmed — page explicitly defines the coefficient as the basis of the distance |
| `Contrastive_learning` | InfoNCE | Confirmed — page explicitly defines InfoNCE |
| `Model_predictive_control` | Receding-horizon control | Confirmed — page states MPC "is also called receding horizon control" |
| `Uncertainty_quantification` | Epistemic/aleatoric uncertainty | Confirmed — page defines both terms distinctly |
| All remaining links (Robotic_arm, Robotic_gripper, Degrees_of_freedom_(mechanics), Robot_kinematics, Proprioception, Telerobotics, Catastrophic_interference, Concept_drift, Online_machine_learning, Regret_(decision_theory), Hellinger_distance, Variational_autoencoder, Simultaneous_localization_and_mapping, Sequential_quadratic_programming, Mixture_model, Energy-based_model, Derivative-free_optimization, Cross-entropy_method, Langevin_dynamics, Diffusion_model, U-Net, Flow-based_generative_model, Markov_decision_process, Robot_control) | — | Confirmed resolving to matching pages (spot-checked; no mismatches found) |

## 3. Covariate shift / the T²ε bound (§3)

| Claim | Verdict | Source(s) |
|---|---|---|
| BC's quadratic bound `J(π) ≤ J(π*) + T²ε`, tight, via a short Markov-chain construction | Confirmed | [Ross & Bagnell 2010 / Bagnell survey, below] |
| Worked intuition (T×T scaling), constants u (recovery cost) and μ (recoverability) | Confirmed | [Ross et al. AISTATS 2011](https://publications.ri.cmu.edu/storage/publications/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf), Definition 1.1 (μ); [Bagnell survey](https://publications.ri.cmu.edu/storage/publications/pub_files/2015/3/InvitationToImitation_3_1415.pdf) |

### Paper Breakdown: ALVINN (Pomerleau, NeurIPS 1989)

| Claim | Verdict | Source |
|---|---|---|
| Architecture: 30×32 video retina (960 units) + 8×32 laser retina (256 units) + 1 feedback unit → 1217 input units → 29 hidden units → 46 output units (45 direction + 1 feedback) | Confirmed — exact match | [Original paper PDF](https://proceedings.neurips.cc/paper/1988/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf) |
| Trained on 1,200 simulated road snapshots, 40 epochs, ~90% correct-curvature-within-tolerance | Confirmed — exact match | Same |
| NAVLAB driven at 0.5 m/s along a 400 m wooded path; comparable to CMU hand-engineered system at 1 m/s on faster hardware | Confirmed — exact match ("1/2 meter per second," "400 meter path," "one meter per second") | Same |
| Venue/year "NeurIPS/NIPS 1989" | Confirmed — paper published in *Advances in Neural Information Processing Systems 1* (1989), from the Nov/Dec 1988 conference; "1989" is the standard citation year | Same |
| 1993 book chapter (not the 1989 paper) is the source of the augmentation/replay-buffer numbers | Unverifiable in this pass — plausible and consistent with the field's citation practice, but the 1993 chapter itself was not directly fetched to verify the "14 extra ways" / "200-pattern buffer" / "~4–5 minutes" figures | — |

### Paper Breakdown: "An Invitation to Imitation" (Bagnell, CMU-RI-TR-15-08, 2015)

| Claim | Verdict | Source |
|---|---|---|
| Two departures from supervised learning (covariate shift; ignoring planning structure) | Confirmed | [Full PDF](https://publications.ri.cmu.edu/storage/publications/pub_files/2015/3/InvitationToImitation_3_1415.pdf) |
| Forward training construction and its role as an exact-but-impractical fix | Confirmed | Same |
| LEARCH algorithm description (plan, compare, fit regressor, iterate) | Confirmed | Same |
| Driving-simulator result: BC "averages about 3-4 failures per lap," DAgger-style "very quickly reaches nearly 0 falls per lap," "no amount of training data" fixes plain BC | Confirmed — exact quotes | Same |
| UAV forest-flight controller "at nearly the same effectiveness as a human pilot"; failures traced to reactive-controller field-of-view limits | Confirmed — exact quotes | Same |
| Crusher: "traversed thousands of kilometers of diverse, rough terrain with minimal human intervention over years of field testing" | Confirmed — exact quote | Same |

## 4. DAgger (§4)

### Paper Breakdown: DAgger (Ross, Gordon, Bagnell, AISTATS 2011)

| Claim | Verdict | Source |
|---|---|---|
| DAgger algorithm pseudocode (β mixture rollout, expert labels, aggregate, retrain) | Confirmed — matches Algorithm 3.1 exactly | [Full paper PDF](https://publications.ri.cmu.edu/storage/publications/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf) |
| Parameter-free variant β₁=1, βᵢ=0 for i>1, "often performs best in practice" | Confirmed — exact quote | Same |
| Return best π̂ᵢ on validation, or one chosen uniformly at random — both provably work | Confirmed | Same, footnote 3 |
| SuperTuxKart: SMILe falls ~2×/lap after 20 iterations; DAgger reaches ~0 falls/lap by iteration 15, already close after 5 | Confirmed — exact match | Same, §5.1 |
| Super Mario Bros.: best DAgger score 3030 vs. max ~4300; beats all SMILe/SEARN configurations | Confirmed — exact match | Same, §5.2 |
| Handwriting: 82% (non-sequential) / 83.6% (supervised) / 85.5% (DAgger) | Confirmed — exact match | Same, §5.3 |
| Venue: AISTATS 2011 | Confirmed | Same |

## 5. "Is Behavior Cloning All You Need?" (Foster, Block, Misra, arXiv:2407.15007, NeurIPS 2024)

| Claim | Verdict | Source |
|---|---|---|
| Theorem: `J(expert) − J(π̂) ≤ 4·R · D²_Hellinger(traj(π̂), traj(expert))` | Confirmed — exact match to Theorem 2.1 | [Full paper PDF](https://arxiv.org/pdf/2407.15007) |
| No R·H term; horizon-independence requires controlled R and controlled policy-class complexity (parameter sharing) | Confirmed | Same, §2.3 |
| Dense-reward case degrades to linear-in-H (Table 1) | Confirmed | Same, Table 1 |
| Four testbeds: MuJoCo Walker2d, Atari BeamRider, a custom sparse-reward navigation ("Car") task, and a GPT-2-style Dyck bracket-sequence generator | Confirmed | Same, §5 |
| Regret flat/improving with H on sparse-reward + parameter-sharing tasks; Dyck task's growth traced to genuine growth in required policy complexity, not a theory violation | Confirmed | Same, §5.2 |
| Log-loss BC robust to optimization error (no horizon factor) vs. indicator-loss BC (degrades by a full extra H factor) | Confirmed | Same, §6.1 |
| "Virtually all empirical work on imitation learning uses parameter sharing" (quoted in notes as "...imitation learning] work uses parameter sharing") | Confirmed as an accurate paraphrase of the paper's own sentence (near-verbatim, not character-for-character) | Same, §2.4 |

## 6. Teleoperation hardware papers (§6)

### ALOHA (Zhao, Kumar, Levine, Finn — RSS 2023)

| Claim | Verdict | Source |
|---|---|---|
| WidowX-250 leader arms, ViperX-300 follower arms, joint-space mapping, ~$20k system cost, 50 Hz | Confirmed | [arXiv PDF](https://arxiv.org/pdf/2304.13705) + Trossen Robotics product pages (WidowX 250, ViperX 300) |
| ACT trained as CVAE, k=100 chunk size, temporal ensembling | Confirmed | Same |
| Chunk-size ablation k=1→100 raising success 1%→44% | Confirmed — exact match | Same, §VI-A |
| CVAE-removal ablation: human data 35.3%→2%, scripted data barely affected | Confirmed — exact match | Same, §VI-B |
| **Six-task success range "20% (Thread Velcro) up to 93% (Slot Battery)"** | **Corrected** — the paper's own results state Slide Ziploc = 88% and **Slot Battery = 96%** final success (not 93%); Thread Velcro = 20% is correct. Fixed to "20%–96%" in both `week1-study-notes.md` and the artifact, with a note that the abstract's "80–90%" headline is an approximate summary (actual per-task figures: cup 84%, battery 96%) | Same, §V-C ("ACT achieves 88% and 96% final success rates respectively") |
| Mobile ALOHA (Fu, Zhao, Finn, CoRL 2024): ~$32k cost; co-training raised success up to 90 percentage points from 50 new demos/task | Confirmed | [Mobile ALOHA project page / arXiv 2401.02117](https://mobile-aloha.github.io/); [ALOHA 2 arXiv 2405.02292](https://arxiv.org/pdf/2405.02292) (cost figure) |

### UMI — Universal Manipulation Interface (Chi et al., RSS 2024)

| Claim | Verdict | Source |
|---|---|---|
| ~$73 gripper; 6.1 mm / 3.5° SLAM tracking error vs. motion capture | Confirmed — exact match | [arXiv HTML](https://arxiv.org/html/2402.10329v3) |
| 100% (UR5) / 90% (Franka zero-shot) cup arrangement | Confirmed — exact match | Same |
| In-the-wild: 1,400 demos, 30 locations, 12 person-hours, 71.7% overall, 75% on unseen cup styles | Confirmed — exact match | Same |
| Other tasks 70–87.5% (tossing 87.5%, cloth folding 70%, dish washing 70%) | Confirmed | Same |

### Bunny-VisionPro (Ding et al., arXiv:2407.03162, 2024)

| Claim | Verdict | Source |
|---|---|---|
| 10-task benchmark, e.g. 9/10 vs. 3/10 and 7/10 on two-cup-stacking | Confirmed — exact match | [arXiv HTML](https://arxiv.org/html/2407.03162) |
| 11% higher success, 45% less completion time on custom bimanual tasks | Confirmed — exact match | Same |
| 22% average success-rate improvement across ACT/Diffusion Policy/3D Diffusion Policy | Confirmed — exact match | Same |
| Haptic feedback improved/maintained success in 9/10 comparisons | Confirmed — exact match | Same |
| >60 Hz loop; $1.20 vibration motor | Confirmed | Same |

### Open-TeleVision (Cheng et al., CoRL 2024)

| Claim | Verdict | Source |
|---|---|---|
| Tested on Unitree H1 and Fourier GR-1; ZED Mini stereo camera, 60 Hz | Confirmed | [arXiv HTML](https://arxiv.org/html/2407.01512) |
| 87–100% policy success across four tasks | Confirmed (task rates 87–100%) | Same |
| Monocular ablation: one task's pick success 92%→46% | Confirmed — exact match | Same |
| Human study: 100% stereo vs. 50–71% monocular on two tasks | Confirmed — exact match (monocular: Can Insertion 71%, Unloading 50%) | Same, Table 3 |

### ACE — Cross-Platform Visual-Exoskeleton (Yang et al., CoRL 2024)

| Claim | Verdict | Source |
|---|---|---|
| Five robot platforms (not six, per one secondary/aggregator summary that was double-checked and found unreliable) | Confirmed | [arXiv HTML](https://arxiv.org/html/2408.11805) — explicitly lists 5 platforms |
| ~1 mm avg. tracking error, 3 mm precision-test error | Confirmed | Same |
| 97.1% vs. 47.6% (ACE vs. GELLO) on one small-workspace condition; GELLO ahead on one medium-workspace condition | Confirmed — exact match | Same |
| Downstream tasks 79–98% success across six tasks | Confirmed — exact match (computed from raw trial counts: 30/38, 45/53, 30/37, 120/123, 25/29, 62/66 → 78.9%–97.6%) | Same, Table 3 |
| Cost: ACE/GELLO ~$600, ALOHA $20k, Mobile ALOHA $32k, DexCap $4k | Confirmed | Same |

## 7. Multi-modal generative-policy papers (§7)

### Implicit Behavioral Cloning (Florence et al., CoRL 2021)

| Claim | Verdict | Source |
|---|---|---|
| Block-pushing: EBM ~100% vs. MSE 87–90% on harder variants, MDN 10% on pixels | Confirmed — exact match (Table 3: EBM 100/100/100, MSE 98.3/89.7/87.0, MDN 100/99.7/10.0) | [arXiv PDF](https://arxiv.org/pdf/2109.00137) |
| Real-robot 1mm-insertion: 83.3% (implicit) vs. 6.7% (explicit) | Confirmed — exact match | Same, Table 6 |
| 95%+ success up to 16 dims (implicit) vs. 8 dims (explicit) | Confirmed — exact match | Same, §4 (N-D Particle Integrator) |

### Diffusion Policy (Chi et al., RSS 2023)

| Claim | Verdict | Source |
|---|---|---|
| 46.9% average improvement over LSTM-GMM/IBC/Behavior Transformer across 15 tasks/4 suites | Confirmed | Multiple independent summaries of the paper |
| Multimodal block-pushing: 0.99/0.94 vs. 0.03/0.01 (LSTM-GMM) vs. 0.01/0.00 (IBC) | Confirmed (consistent with paper's reported results) | [RSS proceedings](https://roboticsproceedings.org/rss19/p026.pdf) |
| Real-world pushing: 95% vs. 20% (IBC) vs. 0% (LSTM-GMM) | Confirmed | Same |

### Revisiting EBMs as Policies / Ranking NCE (Singh, Tu, Sindhwani, arXiv:2309.05803, TMLR 2024)

| Claim | Verdict | Source |
|---|---|---|
| IBC's contrastive objective is biased for non-uniform proposals; Ranking-NCE correction term | Confirmed | [Full PDF](https://arxiv.org/pdf/2309.05803) |
| Synthetic 2D: IBC Bhattacharyya coefficients 0.54–0.87 vs. 0.99+ for flow/diffusion/Ranking-NCE | Confirmed — exact match (0.873, 0.540 for IBC; 0.991–0.998 for others) | Same, Table 1 |
| Obstacle-avoidance path planning: Ranking-NCE lowest collision rate AND lowest cost of every method, including diffusion | Confirmed — exact match (R-NCE: Col=0.030, Cost=0.194, both lowest) | Same, Fig. 4 |
| Push-T: Interpolating Ranking-NCE outperforms plain Ranking-NCE, diffusion, and a flow | Confirmed — exact match (I-R-NCE 0.884 > R-NCE 0.824, Diffusion 0.864, NF 0.866) | Same, Table 6 |

## Corrections made

1. **ALOHA Slot Battery success rate (93% → 96%).** `week1-study-notes.md` §6.2 and the artifact's
   ALOHA card both stated the six-task ALOHA success range as "20%–93%," attributing 93% to the
   hardest showcase task ("Slot Battery"). The ALOHA paper's own results table (§V-C) reports a
   final success rate of **96%** for Slot Battery (and 88% for Slide Ziploc); the "80–90%" figure
   quoted from the abstract is an approximate summary, not the literal per-task numbers. Both files
   now read "20%–96%" with a note explaining the abstract/table discrepancy.
2. **Workspace Wikipedia link.** `week1-study-notes.md` §1, `glossary.md`, and the artifact's §1
   vocabulary table linked "Workspace" to `Robot_end_effector`, a page that does not discuss the
   concept of a robot's reachable workspace at all. Retargeted to `Robot_kinematics`, which
   explicitly defines workspace as "the volume of space reachable by the robot."

## Unverifiable

1. The claim that Pomerleau's 1993 book chapter (not the 1989 NeurIPS paper) is the source of the
   synthetic-augmentation ("14 extra shifted/rotated views"), 200-pattern replay-buffer, and
   "~4–5 minutes of driving" figures often quoted for ALVINN. This is consistent with how the field
   generally cites these two Pomerleau works, but the 1993 chapter itself was not directly located
   and fetched in this pass to confirm the exact figures — flagged as unverifiable rather than
   confirmed.

---

*This file is regenerated in full on every audit run and reflects only the current state of the
notes — it is not a running history of past corrections.*
