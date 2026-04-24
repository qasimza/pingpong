# Beyond Manual Categorization: Benchmarking Generative Strategies for Scalable Physics-Based Animation in Table Tennis

Zaina Qasim
University of Illinois Urbana-Champaign
CS 537: Advanced Topics in Internet of Things- Multimedia Systems
March 6, 2026

## Introduction
Recent advancements in physics-based character animation leverage deep learning to generate agile and natural motion, enabling characters to execute complex movements such as backflips and tennis. However, reproducing the selection and use of diverse motor skills in dynamic environments to solve complex tasks, as humans do, remains a challenge. Wang, Hodgins, and Won (2024) addressed the issue of mode collapse by demonstrating a hierarchical control system for diversified skill learning and a strategy learning framework for effective decision-making [1]. While their method produces agents that play competitively in VR, it is restricted by a reliance on high-quality labeled datasets. This research proposes to overcome this limitation by developing a hybrid model that combines the existing hierarchical approach with a model learnable from unlabeled motions to achieve both high motion quality and scalability. The broader impact of this solution is the democratization of complex character animation, allowing agents to acquire professional-grade athletic skills from massive, unstructured motion databases without the need for manual labeling.


## Possible Approaches

To solve the scalability bottleneck of relying on manually categorized skills, I will benchmark the explicitly programmed "mixer policy" of the Supervised Baseline (Wang et al., 2024) against three distinct generative architectures designed to learn from unlabeled/unstructured motion:

**Unsupervised Hybrid Latent Prior (Bae et al., 2025)**. This architecture replaces the baseline's manual skill mixer with an encoder-decoder driven by a Hybrid Motion Prior. It utilizes a discrete latent model (hierarchical codebook) to act as a high-level strategy for capturing distinct skills, while simultaneously augmenting the sampled vector with continuous residuals from a prior network to ensure temporally smooth physical transitions [2].
**Iterative Physics-based Augmentation (Xu et al., 2025 - PARC)**. Rather than directly tackling a massive dataset, this architecture employs a self-consuming loop starting from a small data seed. It uses a kinematic motion generator (diffusion model) to synthesize new terrain-traversal skills. These outputs are then fed into a physics-based motion tracking controller (a DeepMimic-style RL policy) which corrects physically impossible artifacts in a simulator. The corrected motions are added back to the dataset to continually retrain the generator [3].
**Inference-Time Physics Steering (Yuan et al., 2026)**. This architecture relies on an unstructured generative prior but aligns it at inference time. It utilizes a self-supervised Latent World Model (VJEPA-2) to observe generated context frames and predict future representations. A "surprise score" (WMReward) is computed by calculating the cosine similarity between the world model's prediction and the actual generated motion, acting as a reward signal to steer the character [4].

## Experiment
To rigorously evaluate whether the three generative models (Unsupervised Hybrid Prior, Inference-Time Physics Steering, and Iterative Physics-Based Augmentation) can overcome the baseline's reliance on explicit data labels without sacrificing performance, the evaluation will strictly replicate the three experiments used by Wang et al. (2024). The methodology detailed below will use the dataset used by Wang et al (2024) [1].

### Skill Evaluation (Motion Quality and Task Performance)
The agents will be tested on their fundamental ability to execute table tennis skills by launching 10,000 random balls at them in simulation
**Motion Quality Metrics**: To test if the generative models can autonomously discover and execute natural skills without manual categorization, I will measure:
- Discriminator Score: Evaluates how similar the generatively synthesized strike motions are to the original human reference motions
- Skill Accuracy: Measures if the generative agents successfully select the correct fundamental skill (e.g., forehand vs. backhand) for the incoming ball without explicit rule-based mixers
- Diversity Score: Measures the visual distinctiveness of the generated motions to ensure the unsupervised models do not suffer from mode collapse
**Task Performance Metrics**: I will measure the physical precision of the generated strikes using Sustainability (the average number of successful continuous hits) and Accuracy (the average error distance in meters between the ball's actual contact point and the target location)

### Agent-Agent Interaction (Strategic Efficacy)
To evaluate high-level decision-making, the generative agents will play simulated 10,000-point table tennis matches against two fixed baseline opponents: a "Random Strategy" opponent and a "Video Strategy" opponent (derived from human broadcast videos)
**Metrics**: This experiment will evaluate both competitive and cooperative scenarios.
- Competitive Metric: Measured by the Winning Rate against the opponents. 
- Cooperative Metric: Measured by the Average Rounds (length of rallies) the agents can maintain without dropping the ball, demonstrating control and restraint

### Human-Agent Interaction in VR (Real-World Robustness)
- The ultimate test of the generated motions is their robustness against unpredictable human behavior. The generative models will control a physically simulated full-body agent in real-time to play against human users in Virtual Reality (VR)
- **Metrics**: Real-world applicability and responsiveness will be quantified using the human-agent Winning Rate (for competitive matches) and Average Hits (for cooperative rallies).

## Action Plan

Week 1-2: Infrastructure Setup. Complete literature review. Set up the Wang et al. (2024) code+dataset, as well as VR interface. Ensure that the results derived from this paper are achieved independently.
Week 3: Model Integration. Implement the three generative approaches as described above.
Week 4: Data Collection. Run the 10,000-ball skill evaluations for all agents. Log the resulting 3D kinematic data and record gameplay performance metrics during the Agent-Agent and Human-Agent VR matches.
Week 5: Final Analysis. Compare the generative models to the baseline using Discriminator Score, Diversity Score, Sustainability (average hits), Accuracy (landing error), and overall Winning Rates. Write the draft for the final report.
Week 6-9: Presentation and Final Report. Buffer for unforeseen issues. 

