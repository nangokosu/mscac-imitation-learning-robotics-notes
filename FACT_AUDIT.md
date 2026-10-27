# Fact Audit

This file holds one section per audited week. Each run replaces only the section for the week it audited; other weeks' sections are left intact.

---

## Week 4: Diffusion Planning & World Models

**Last audited:** 2026-10-27 (Week 4)
**Scope of this run:** the `id="week-4"` `.week-block` of the published artifact (https://claude.ai/artifact/5Eh9TePpNpmwVvLmAbWPoG) plus its `#glossary` card "§C Diffusion Models for Planning (Week 4)". `week4-study-notes.md` does not exist in the repo, so there was no markdown to audit or to drift from (no drift check possible). The "Weeks 4 & 11: World models and JEPA" section of `glossary.md` (Week 4 entries only) was checked and needed no change. Most effort went to §4.1–§4.6. §4.7–§4.11 (world models) were spot-checked only, since the caller had already checked them against the papers.

**Summary: 33 claims audited: 22 confirmed, 8 corrected, 3 unverifiable (left as-is).**

**Artifact republished:** version 24 (version id `1793111285-8e43`), same URL. Only the `week-4` block and the §C glossary card changed. Figure numbers, class names, ids and `data-*` attributes are unchanged.

### Corrected claims

#### 1. §4.6 "Diffusion Policy on MetaWorld benchmarks" table (fabricated; replaced)
- **Before:** a 5-row table (Box-Close 92/71/68, Coffee-Button 89/45/52, Drawer-Close 85/61/58, Hammer 76/22/28, Pen-Insert 72/18/15) comparing "Diffusion Policy" with an "LSTM Baseline" and a "Transformer Baseline" on MetaWorld.
- **Finding:** Diffusion Policy (Chi et al., RSS 2023, arXiv:2303.04137) never evaluates on MetaWorld. Its simulation benchmarks are Robomimic (Lift, Can, Square, Transport, ToolHang), Push-T, Block Pushing and Franka Kitchen. Its baselines are LSTM-GMM (Robomimic's BC-RNN), IBC and BET. There is no "Transformer baseline"; the transformer is one of Diffusion Policy's own two variants. None of the old numbers appear in the paper.
- **After:** a table taken from the paper's visual-policy benchmark table (max / average of the last 10 checkpoints), with DP-CNN, DP-Transformer, LSTM-GMM and IBC on Lift (ph), Square (mh), Transport (mh), ToolHang (ph) and Push-T. Example: Transport (mh) is 0.89/0.69 for DP-CNN, 0.73/0.50 for DP-Transformer, 0.44/0.24 for LSTM-GMM and 0.00/0.00 for IBC. All 20 cells were extracted twice from the paper in independent fetches and matched both times.
- **Sources:** [arXiv:2303.04137 abstract](https://arxiv.org/abs/2303.04137), [arXiv HTML v5](https://arxiv.org/html/2303.04137v5) (benchmark tables, baselines list, task list).

#### 2. §4.6 interpretation: "2-5×" improvement and "especially on multi-modal tasks (Hammer, Pen-Insert)" (removed)
- **Before:** "diffusion outperforms LSTM/Transformer baselines substantially, especially on multi-modal tasks (Hammer, Pen-Insert). Improvement is 2-5× on complex skills."
- **After:** easy tasks (Lift) are nearly saturated for every method except IBC. The gap opens on long-horizon and mixed-quality tasks (Transport mh: 0.69 vs. LSTM-GMM's 0.24 average), and IBC scores 0 on most Robomimic tasks. The paper reports "an average improvement of 46.9%" over prior state-of-the-art methods. The "2-5×" figure and the Hammer/Pen-Insert tasks do not appear in the paper.
- **Source:** [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) (the abstract, all versions, states 46.9%).

#### 3. §4.5 takeaway: "best empirical results on long-horizon manipulation benchmarks (MetaWorld, bridge, ALOHA)"
- **Before:** "...has the best empirical results on long-horizon manipulation benchmarks (MetaWorld, bridge, ALOHA)."
- **After:** "...has strong empirical results on manipulation benchmarks: Chi et al.'s Diffusion Policy reports an average improvement of 46.9% over prior state-of-the-art methods across Robomimic, Push-T, Block Pushing and Franka Kitchen (§4.6)."
- **Finding:** the paper mentions none of MetaWorld, Bridge or ALOHA. ALOHA is the hardware platform of ACT (Zhao et al.), a different method. No source in the section supports "best".
- **Source:** [arXiv HTML v5](https://arxiv.org/html/2303.04137v5).

#### 4. Fig. 9, reverse-process step 3: wrong coefficient on the predicted noise
- **Before:** `x_{t-1} = (x_t - √(1-α_t) * predict_noise) / √α_t + small noise`
- **After:** `x_{t-1} = (x_t − β_t/√(1−ᾱ_t) · predict_noise) / √α_t + σ_t · z`
- **Finding:** in DDPM Algorithm 2 the coefficient is (1−α_t)/√(1−ᾱ_t) = β_t/√(1−ᾱ_t), not √(1−α_t) = √β_t. The `<pre>` formula in §4.2 was already correct; only the figure text was wrong, so the two now agree.
- **Source:** Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models," [arXiv:2006.11239](https://arxiv.org/abs/2006.11239), Algorithm 2 and Eq. 11 (read from the PDF, p. 4).

#### 5. Fig. 9, network structure labelled "Diffusion Policy Architecture"
- **Before:** observation "from CNN/ViT", a "concat" box joining observation, noisy action and timestep, and a "Diffusion Network: Dense 512→256→512 (with residual skip)".
- **After:** observation "ResNet-18 enc."; the joining box reads "FiLM"; network "1D temporal-conv U-Net (or causal transformer)". The caption now explains that in the CNN variant the observation and timestep enter every conv layer through FiLM, and the transformer variant takes them by cross-attention.
- **Finding:** the paper's noise-prediction network is a 1D temporal CNN (adapted from Janner et al.) with FiLM conditioning on observation features and diffusion step k, or a minGPT-based transformer. In the transformer, observation features enter via cross-attention and the k-embedding is prepended as the first token. The visual encoder is "a standard ResNet-18 (without pretraining)" with spatial softmax and GroupNorm. There is no dense 512-256-512 MLP and no concatenation.
- **Source:** [arXiv HTML v5](https://arxiv.org/html/2303.04137v5), §III (network architecture, visual encoder).

#### 6. Fig. 9 caption: "denoises over T steps (typically 100)"
- **Before:** "...iteratively denoises over T steps (typically 100)..."
- **After:** "...over T steps (Diffusion Policy: 100 in simulation; 10–16 DDIM steps on the real robot)..."
- **Finding:** simulation uses "the same 100 denoising diffusion iterations for both training and inference." For the real robot, the main text quotes "DDIM with 100 training iterations and 10 inference iterations," while Appendix A.4 and hyperparameter Table 7 give 16 evaluation iterations. Both figures are reported, so the range 10–16 is stated rather than one number.
- **Source:** [arXiv HTML v5](https://arxiv.org/html/2303.04137v5), main text and Appendix A.4 / Tables 7–8.

#### 7. Fig. 10 step labels "(75% noise)", "(50% noise)", "(25% noise)" at t = 75 / 50 / 25 of 100
- **Before:** noise fractions implied to fall linearly with t.
- **After:** "(≈85% noise)", "(≈50% noise)", "(≈15% noise)", plus a caption sentence saying these are the noise variance 1 − ᾱ_t under the cosine schedule Diffusion Policy uses (T = 100), and that noise is not removed linearly in t.
- **Derivation:** Diffusion Policy uses iDDPM's square-cosine schedule, ᾱ(t) = f(t)/f(0) with f(t) = cos²(((t/T)+s)/(1+s)·π/2), s = 0.008. Evaluated by hand: 1 − ᾱ ≈ 0.856 at t = 75, ≈ 0.506 at t = 50, ≈ 0.153 at t = 25. Diffusers' implementation offsets the index by one step, giving ≈ 0.866 / 0.522 / 0.165, so the labels are written as approximate ("≈"). These are computed values, not numbers quoted from a paper.
- **Source:** [arXiv HTML v5](https://arxiv.org/html/2303.04137v5) ("the Square Cosine Schedule proposed in iDDPM ... works best for our tasks"; T = 100).

#### 8. Glossary card §C (Week 4): "Denoising Steps: typically 100-1000 reverse steps for inference"
- **Before:** "typically 100-1000 reverse steps for inference."
- **After:** "DDPM image models use T = 1000; Diffusion Policy trains with 100 and samples with 100 in simulation, or 10–16 DDIM steps on the real robot."
- **Sources:** [DDPM, arXiv:2006.11239](https://arxiv.org/abs/2006.11239) §4 ("We set T = 1000 for all experiments"); [Diffusion Policy HTML v5](https://arxiv.org/html/2303.04137v5).

### Unverifiable claims (left as-is)

- **§4.4 planning loop (sample N trajectories from the diffusion model, score each with a value function, execute the argmax's first action).** This is presented as a generic pattern and cites no paper. Diffusion Policy itself does no value-based reranking: it executes the sampled action chunk with receding-horizon control. Diffuser (Janner et al., ICML 2022, [arXiv:2205.09991](https://arxiv.org/abs/2205.09991)) plans by classifier-guided sampling and inpainting, not sample-then-rerank. The nearest published match found is IDQL ([arXiv:2304.10573](https://arxiv.org/abs/2304.10573)), which samples from a diffusion behavior policy and selects or reweights candidates with a critic. Its abstract describes critic-weighted importance sampling rather than a hard argmax. Nothing here contradicts the loop as a generic design, but no source was found that states it exactly as written, so it is not marked Confirmed.
- **Qualitative comparison tables in §4.5 (and the "VAE ✓ (limited)", "Very High" quality ratings).** These are judgement calls, not quantitative claims tied to a source. They were not changed.
- **"A self-driving company has neither [a reward signal nor online data collection] for a real fleet" (§4.9 lead-in).** This is the author's framing of MILE's motivation, not a claim from the MILE paper. No source confirms or refutes it, so it was left unchanged.

**Waabi internal-use check:** the Week 4 block makes no claim about what Waabi uses internally. Every Waabi mention is authorship (Copilot4D, UnO, UniSim), and each was confirmed below.

### Confirmed claims

**Diffusion fundamentals (§4.1–§4.3), against DDPM ([arXiv:2006.11239](https://arxiv.org/abs/2006.11239), read from the PDF, pp. 3–5):**
- Linear noise schedule β_t from 10⁻⁴ to 0.02: "increasing linearly from β1 = 10−4 to βT = 0.02" (§4).
- ᾱ_t = ∏ α_i, α_t = 1 − β_t, and closed-form forward sample x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε: matches Eq. 4 / Eq. 9.
- Training loss ‖ε − ε_θ(x_t, t)‖² (conditioned on observation in the policy setting): matches L_simple, Eq. 14 / Algorithm 1.
- The §4.2 reverse update x_{t−1} = (1/√α_t)(x_t − β_t/√(1−ᾱ_t)·ε_θ) + σ_t·z: matches Eq. 11 and Algorithm 2 exactly.
- "σ_t shrinks toward 0 as t → 0": consistent with σ_t² = β_t under the increasing linear schedule, and z = 0 at the final step (Algorithm 2, line 3).
- §4.1 "100 times with refinements": consistent with Diffusion Policy's 100 denoising iterations in simulation ([arXiv HTML v5](https://arxiv.org/html/2303.04137v5)).
- Diffusion handles multimodal action distributions, and policies generate action sequences rather than single actions (§4.3–§4.4). The paper states both: it predicts an action sequence of horizon T_p (typically 16) and executes T_a = 8.
- Diffusion Policy authors (Chi, Xu, Feng, Cousineau, Du, Burchfiel, Tedrake, Song) and venue (RSS 2023; arXiv v5 is the journal extension).

**Spot-checks, §4.7–§4.11:**
- PlaNet (Hafner et al., ICML 2019, [arXiv:1811.04551](https://arxiv.org/abs/1811.04551), read from the PDF, pp. 6–7): Cheetah Run 662 vs. D4PG 524 (D4PG at 100,000 episodes) and Walker Walk 951 vs. 968, from Table 1. CEM with H = 12, J = 1000 candidates, top K = 100, I = 10 iterations. Training takes 10–20 hours on one V100. The stochastic component is "even more important – the agent does not learn without it." All match.
- Copilot4D ([arXiv:2311.01017](https://arxiv.org/abs/2311.01017)): title, authors (Zhang, Xiong, Yang, Casas, Hu, Urtasun), ICLR 2024, and Chamfer reduction of >65% at 1 s and >50% at 3 s across NuScenes, KITTI Odometry and Argoverse2. All match.
- MILE ([arXiv:2210.07729](https://arxiv.org/abs/2210.07729)): Hu et al. (Wayve), NeurIPS 2022, 31% driving-score improvement in a new town and new weather. Matches.
- Ha & Schmidhuber, World Models ([worldmodels.github.io](https://worldmodels.github.io/)): at τ = 0.10, 2086 ± 140 in the dream vs. 193 ± 58 in reality. At τ = 1.15, 1092 ± 556 in reality. The required score is 750. CarRacing 906 ± 21 used a controller trained in the actual environment. All match.
- Related Waabi work: UnO is CVPR 2024 (Agro, Sykora, Casas, Gilles, Urtasun; [CVF open access](https://openaccess.thecvf.com/content/CVPR2024/html/Agro_UnO_Unsupervised_Occupancy_Fields_for_Perception_and_Forecasting_CVPR_2024_paper.html)) and self-supervises a continuous 4D occupancy field from LiDAR. UniSim is CVPR 2023 (Yang et al., Waabi; [arXiv:2308.01898](https://arxiv.org/pdf/2308.01898), [CVPR 2023 supplement](https://www.cs.toronto.edu/~zeyang/publication/CVPR2023UniSim/supp.pdf)) and is a neural closed-loop sensor simulator. Both match.

**Wikipedia links in the Week 4 block:** [Bird's-eye view](https://en.wikipedia.org/wiki/Bird%27s-eye_view), [Evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound), [Vector quantization](https://en.wikipedia.org/wiki/Vector_quantization) and [Gated recurrent unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit) were fetched this run; each resolves and matches its attached concept. Model predictive control, Kullback–Leibler divergence and Cross-entropy method were confirmed in the Week 3 run. Recurrent neural network and Lidar were not re-fetched this run.

**Not re-verified this run (the caller had already checked them against the papers):** Dreamer's 823 / PlaNet 332 / D4PG 786; DreamerV3 details (KL-balancing weights, symlog, Minecraft diamonds); MILE's Table 1 cells (61.1 ± 3.2, LAV 46.5, TransFuser 31.0, expert 88.4), 2.9M frames / 32 h, 43.0 Hz, ablation 7621 → 4564; the individual Copilot4D Chamfer table cells and parameter counts; GAIA-1 parameter counts, 4,700 hours, 64 A100s × 15 days. None of these looked inconsistent on reading, but they were not independently re-fetched.

### Notes on process
- README.md was read only to find the artifact URL.
- No `week4-study-notes.md` exists. It was not created, per the caller's instruction.
- The artifact's full HTML was paged through end to end because the publish tool requires it. Only the `week-4` block and the §C glossary card were audited or edited. Weeks 1, 3, 10 and 11 were not fact-checked.
- `glossary.md` Week 4 entries (world model, RSSM, prior/posterior, imagination training, MILE, VQ tokenizer, discrete diffusion, Copilot4D, model exploitation) are consistent with the confirmed claims above. No change was needed.
- No git commit or push was made.

---

## Week 3: Offline/Batch RL

**Last audited:** 2026-09-17
**Scope of this run:** Week 3 only (`week3-study-notes.md`, the "## Week 3" section of `glossary.md`, and the `id="week-3"` `.week-block` plus its `#glossary` entries in the published artifact). Weeks 1 and 2 were **not** audited in this run and are intentionally excluded from this report — see prior audit history for those, if any. This file is regenerated in full each run and reflects only the current run's scope, not an accumulating history.

**Summary: 47 claims audited — 42 confirmed, 4 corrected, 1 claim group flagged unverifiable (see below).**

The markdown (`week3-study-notes.md`) and the artifact's `week-3` section said the same things everywhere checked, with one exception: the artifact's NFQ citation never included a page range in the first place (so it could not itself be "wrong" the way the markdown's `55–73` was), meaning the two sources had not drifted apart from each other so much as the markdown carried an extra, incorrect detail the artifact never had. Everywhere else, wording differed cosmetically (the artifact is compressed for space) but every checked number, name, and claim matched between the two.

### Corrected claims

#### 1. BEAR paper's author list (missing an author)

- **Before:** "Aviral Kumar, Justin Fu, George Tucker, Sergey Levine — NeurIPS 2019" (in both `week3-study-notes.md`'s BEAR Paper Breakdown citation line and the artifact's BEAR paper-card citation).
- **After:** "Aviral Kumar, Justin Fu, Matthew Soh, George Tucker, Sergey Levine — NeurIPS 2019."
- **Source:** NeurIPS proceedings abstract page, [Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction](https://papers.nips.cc/paper/2019/hash/c2073ffa77b5357a498057413bb09d3a-Abstract.html), and [PDF](https://proceedings.neurips.cc/paper_files/paper/2019/file/c2073ffa77b5357a498057413bb09d3a-Paper.pdf), both confirming the five-author list including Matthew Soh.
- **Files changed:** `week3-study-notes.md`, published artifact (BEAR paper-card).

#### 2. NFQ paper's page range (off by one page)

- **Before:** "Autonomous Robots 27(1):55–73, 2009" in `week3-study-notes.md`'s NFQ citation. (The artifact's own citation line never included a page range, so nothing needed changing there.)
- **After:** "Autonomous Robots 27(1):55–74, 2009."
- **Source:** Three independent citation-metadata lookups (Springer listing via web search snippets, a second independent search, and a third cross-check) consistently report pages 55–74; one earlier search snippet said 55–73 and could not be corroborated by any other source.
- **Files changed:** `week3-study-notes.md` only (artifact had no page range to correct).
- Note: the paper's specific *results* numbers quoted in the notes (83.8%/52.8% defense success, ~198 seconds of motor-control interaction, 132 real-robot dribbling trials, "80% of decisions," five world championships) could **not** be independently re-verified against the paper's full text, which is paywalled and not accessible via the tools available during this audit. These are flagged as Unverifiable below rather than left silently as Confirmed.

#### 3. IRIS paper — task/dataset mischaracterized (numbers were right, description was wrong)

- **Before:** "On a real-robot-derived crowdsourced pick-and-place task (RoboTurk-collected, crowdsourced, intentionally suboptimal demonstrations), IRIS reached 81.3% success versus BC's 13.7%, BC-RNN's 16.7%, and BCQ's 18.0%..." (`week3-study-notes.md`); artifact said "On a real-robot-derived crowdsourced pick-and-place task: 81.3% success vs. BC's 13.7%, BC-RNN's 16.7%, BCQ's 18.0%."
- **What was actually verified:** The IRIS paper (arXiv:1911.05321) reports these exact numbers (81.3% / 13.7% / 16.7% / 18.0%) in Table I, but for the **"Robosuite Lift"** task — a simulated lifting task using demonstrations collected via the RoboTurk teleoperation interface from a **single human**, deliberately made suboptimal — not the paper's separate, genuinely crowdsourced, real-robot-derived "RoboTurk Cans" pick-and-place dataset (multiple humans), where every method scored far lower (IRIS 28.3%, BC/BCQ ≈0%).
- **After:** Notes and artifact now describe the 81.3%/13.7%/16.7%/18.0% result as coming from "a simulated Robosuite object-lifting task with suboptimal demonstrations from a single human (collected via the RoboTurk teleoperation interface)," and add the separate, much-lower crowdsourced-dataset numbers for contrast.
- **Source:** IRIS paper full text (arXiv:1911.05321), Table I and its surrounding "Robosuite Lift — Suboptimal Demonstrations from a Human" / "RoboTurk Can Pick and Place — Crowdsourced Demonstrations" sections, fetched and read directly.
- **Files changed:** `week3-study-notes.md`, published artifact (IRIS paper-card).

#### 4. COG paper — baseline comparison overstated as "near 0%" for plain BC

- **Before:** "...COG reached 68–78% success while BC, plain offline RL trained only on the task-specific set, and SAC all scored at or near 0%..." (`week3-study-notes.md`); artifact said "...BC, task-only offline RL, and SAC all scored near 0%."
- **What was actually verified:** COG's Table 1 (arXiv:2010.14500) shows COG at 68–78% on the three novel-initial-condition drawer settings, matching the notes. But the paper's plain "BC" baseline (trained on *all* prior + task data) scored 22–34% on those same settings — not "near 0%." Only the "no prior data" offline-RL ablation and SAC (which diverged) scored 0%.
- **After:** Notes and artifact now state that the offline-RL-without-prior-data ablation and SAC both scored 0% (SAC diverged), while the naive BC-on-all-data baseline topped out at 22–34% — still far below COG, but not "near 0%." The real-robot comparison baseline is also now correctly identified as "BC-oracle" (the paper's handpicked-successful-trajectories variant, which scored 0/8), matching the paper's own terminology.
- **Source:** COG paper full text (arXiv:2010.14500), Table 1 and the real-world-evaluation section, fetched and read directly.
- **Files changed:** `week3-study-notes.md`, published artifact (COG paper-card).

### Unverifiable claims

- **NFQ (Neural Fitted Q-Iteration for RoboCup Soccer) specific result figures** — "Defense success rose from 52.8% (hand-coded) to 83.8% (learned)... An effective motor controller emerged from only ~198 seconds of real interaction... 132 real-robot trials (~30 minutes)... up to 80% of in-game decisions... five RoboCup world championships." The paper (Riedmiller, Gabel, Hafner, Lange, *Autonomous Robots* 27(1):55–74, 2009) is paywalled, and none of the tools available during this audit (WebSearch, WebFetch, direct PDF fetch attempts) could retrieve its full text. A closely related 2007 conference paper by an overlapping author set was retrieved and independently confirms the *general* claims (Brainstormers were "always among the best three teams... during the last 7 years," won the RoboCup 2005 World Championship, and were "a five-time winner of the RoboCup World Championship" per a secondary source) but does not contain the specific NFQ case-study percentages quoted in the notes. These specific figures are therefore Unverifiable — not confirmed, not contradicted — and are left as-is in the notes/artifact per the audit's instruction not to "correct" a claim that cannot actually be verified.

### Confirmed claims

All of the following were checked against the primary source (the paper's own abstract/PDF, or Wikipedia for definitions/links) and matched exactly.

**Terminology / definitions (§1–§3, §6):**
- MDP tuple (S, A, T, r, γ, d₀) — matches standard RL formalization (Wikipedia, *Markov decision process*).
- State-visitation distribution, behavior vs. target policy, on/off-policy, REINFORCE log-derivative-trick derivation, actor-critic advantage formula — standard RL theory, consistent with Wikipedia's *Policy gradient method*, *Q-learning* pages.
- Q-learning Bellman optimality backup and Fitted Q-Iteration description — standard, matches Wikipedia *Q-learning*.
- Maximum Mean Discrepancy definition/formula, used for BEAR — confirmed against Wikipedia's *Kernel embedding of distributions* article, which explicitly defines and discusses MMD.
- Mass-covering vs. mode-seeking KL divergence description — standard, correctly attributed direction (forward KL(π_β‖π_θ) mass-covers; reverse KL(π_θ‖π_β) mode-seeks).

**Wikipedia link resolution (spot-checked):**
- `Model-free_(reinforcement_learning)` (linked from "TD3" in the TD3+BC breakdown) — page exists, covers model-free RL, and explicitly lists/describes TD3 in its algorithm table. Appropriate fallback since no dedicated Wikipedia page for TD3 exists.
- `Kernel_embedding_of_distributions` (linked from "Maximum Mean Discrepancy") — page exists and defines MMD directly.
- `RoboCup`, `Reinforcement_learning`, `Markov_decision_process`, `Q-learning`, `Policy_gradient_method`, `Exploration-exploitation_dilemma`, `Kullback–Leibler_divergence`, `Cross-entropy_method` — all resolve and match their attached concepts.

**QT-Opt (Kalashnikov et al., CoRL 2018, arXiv:1806.10293)** — fully confirmed against the paper's own PDF (fetched and read directly, including architecture diagram in Appendix E):
- 472×472 image crop — confirmed (Fig. 2, Appendix D.1).
- 7 convolutional layers before the action is fused, 9 more after, two FC layers — confirmed verbatim ("processes it with 7 convolutional layers... modeled by 9 more convolution layers followed by two fully-connected layers," Appendix E).
- CEM: N=64 samples, M=6 elites, 2 iterations — confirmed verbatim (§4.2).
- 1,000 "Bellman updater" jobs, 10 training workers, up to 15M gradient steps — confirmed verbatim (§4.3).
- 7 KUKA LBR IIWA robots, ~4 months, ~800 robot-hours — confirmed verbatim (§5, Data collection).
- 580K off-policy + 28K on-policy grasps → 96% success; 78% for the Levine et al. baseline; 87% using only the 580K off-policy grasps — confirmed exactly against Table 1.

**BCQ (Fujimoto, Meger, Precup, ICML 2019)** — three named batch settings (final buffer / concurrent / imitation), three extrapolation-error sources (Absent Data, Model Bias, Training Mismatch), and the headline result (BCQ alone matches/beats the behavior policy in every setting; DDPG/DQN diverge even in "concurrent") — all confirmed against the paper's own text (fetched via arXiv HTML).

**TD3+BC (Fujimoto, Gu, NeurIPS 2021)** — objective formula (λ = α / mean|Q|, α = 2.5), total D4RL score 979.3 vs. CQL's 764.3 vs. Fisher-BRC's 974.6, training time 39 minutes vs. CQL's 4h11m — all confirmed exactly against the paper's own text.

**CQL (Kumar, Zhou, Tucker, Levine, NeurIPS 2020)** — CQL(ρ) and CQL(H) objective forms, the lower-bound guarantee claim, and the D4RL domains tested — confirmed against the paper's own text; general "2–5× higher return" framing in the abstract is consistent with (though more specific than) the notes' qualitative summary.

**REM (Agarwal, Schuurmans, Norouzi, ICML 2020)** — median normalized scores 123.8% (REM, 49/60 games beat online DQN), 118.9% (QR-DQN, 45/60), 111.0% (Ensemble-DQN, 39/60) on the Atari DQN Replay Dataset — confirmed exactly against the paper's Table 1.

**"When Should We Prefer Offline RL Over Behavioral Cloning?" (Kumar, Hong, Singh, Levine, ICLR 2022)** — concentrability coefficient C* definition, the three-regime analysis (C*=1 no-advantage theorem, critical-states argument, noisy-data Õ(√H) vs. Õ(H) bound), and the exact manipulation-task numbers (85.7%/90.3%/92.4% CQL-on-noisy-expert vs. 14.5%/17.4%/33.2% BC-on-expert) — all confirmed exactly against the paper's own text and Table 2 (fetched and read directly). Author list (Kumar, Hong, Singh, Levine) also confirmed correct.

**"Why Should I Trust You, Bellman?" (Fujimoto, Meger, Precup, Nachum, Gu, ICML 2022)** — the toy two-state MDP example (Q(s₀,a)=C, Q(s₁,a)=C/γ ⟹ zero Bellman error, value error C), the proven bound (for γ=0.99, ratio bounded in [0.503, 100], matching the notes' "~0.5 to 100"), and the FQE-vs-BRM off-policy HalfCheetah result (FQE's Bellman error over 1000× BRM's while having lower value error) — all confirmed exactly against the paper's own text and Table 1.

**"Instabilities of Offline RL with Pre-Trained Neural Representation" (Wang, Wu, Salakhutdinov, Kakade, ICML 2021)** — general framing (a good representation alone is insufficient; exponential error amplification without a policy-completeness/low-distribution-shift condition) confirmed against the paper's PMLR abstract page.

**D4RL (Fu, Kumar, Nachum, Tucker, Levine, arXiv:2004.07219)** — author list and general framing (benchmark suite designed around realistic data-collection properties: narrow/biased data, human demonstrations, mixed-quality data) confirmed against the paper's own introduction (fetched and read directly).

**IRIS (Mandlekar et al., arXiv:1911.05321)** — numeric results 81.3%/13.7%/16.7%/18.0% confirmed exactly against Table I (see Corrected #3 above for the task-description correction).

**robomimic (Mandlekar et al., CoRL 2021)** — six algorithms compared (BC, BC-RNN, HBC, BCQ, CQL, IRIS); Square(PH) task numbers BC-RNN 84.0%, BCQ 50.0%, CQL 5.3% confirmed exactly; direct quote "neither BCQ nor CQL performs particularly well on these human-generated datasets" confirmed verbatim; "10% to 100% decrease" from checkpoint-selection-by-validation-loss vs. best-checkpoint confirmed (notes' "10–100% swings" is a fair paraphrase).

**Reward sketching (Cabi et al., RSS 2020)** — 80% lifting / 60% stacking success under normal conditions confirmed exactly against Table II; ablation results (task-specific-only data and non-distributional RL both sharply degrading performance) confirmed against the paper's own text.

**COG (Singh et al., CoRL 2020)** — 68–78% success range and 7/8 real-robot trials confirmed exactly against Table 1 and the real-world-evaluation section (see Corrected #4 above for the baseline-characterization correction).

**Targeted Environment Design from Offline Data (Gur, Nachum, Faust)** — confirmed to exist at NeurIPS 2021 DeepRL Workshop with the described "offline targeted environment design" framing, via OpenReview and ML Anthology listings.

**Benchmarking Batch Deep RL Algorithms (Fujimoto, Conti, Ghavamzadeh, Pineau)** — confirmed to exist (arXiv:1910.01708, Deep RL Workshop, NeurIPS 2019); general result direction (many off-policy methods underperform both online DQN and the behavioral policy; a discrete BCQ adaptation is introduced as a strong baseline) confirmed against the paper's abstract.

**IQ-Learn (Garg, Chakraborty, Cundy, Song, Ermon, NeurIPS 2021)** — Humanoid result 5227.1 (IQ-Learn) vs. 5312.8 (expert) confirmed exactly against Table 3; "reached expert-level performance roughly 3× faster (in environment steps)" and "GAIL/ValueDICE near random" on Atari confirmed against the paper's own §7.3 text ("IQ obtains 3-7x normalized score and converges in ~300k steps, being 3x faster compared to Q-learning based RL methods").

**Cal-QL (Nakamoto et al., NeurIPS 2023)** — 11 tasks, average post-fine-tuning score 90 (Cal-QL) vs. 71 (CQL) vs. 69 (IQL), best on 9/11 tasks, AntMaze medium-diverse 62→98 (CQL) vs. 54→97 (Cal-QL), Adroit relocate-binary 6→69 (CQL) vs. 3→98 (Cal-QL) — all confirmed exactly against the paper's Table 1.

### Notes on process

- README.md was read only to locate the published-artifact URL, per the scoping instruction for this run.
- Only `week3-study-notes.md` was read among the `weekN-study-notes.md` files.
- Only the "## Week 3" section of `glossary.md` was checked; every term there was cross-referenced against the paper-level verification above and found consistent with what was actually confirmed (no glossary-specific errors found; no changes made to `glossary.md`).
- Week 1 and Week 2 content in the notes and the artifact was read only insofar as unavoidable while paging through the full artifact HTML file (required by the publish tool before an update is accepted), but was not audited, fact-checked, or edited, per this run's explicit scope restriction.
