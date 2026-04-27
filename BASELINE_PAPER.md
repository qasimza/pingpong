# Strategy and Skill Learning for Physics-based Table Tennis Animation

**Authors:** Jiashun Wang, Jessica Hodgins, Jungdam Won
**Affiliations:** Carnegie Mellon University, The AI Institute, Seoul National University

---

## Abstract
Recent advancements in physics-based character animation leverage deep learning to generate agile and natural motion, enabling characters to execute movements such as backflips, boxing, and tennis. However, reproducing the selection and use of diverse motor skills in dynamic environments to solve complex tasks, as humans do, still remains a challenge. We present a strategy and skill learning approach for physics-based table tennis animation. Our method addresses the issue of mode collapse, where the characters do not fully utilize the motor skills they need to perform to execute complex tasks. More specifically, we demonstrate a hierarchical control system for diversified skill learning and a strategy learning framework for effective decision-making. We showcase the efficacy of our method through comparative analysis with state-of-the-art methods, demonstrating its capabilities in executing various skills for table tennis. Our strategy learning framework is validated through both agent-agent interaction and human-agent interaction in Virtual Reality, handling both competitive and cooperative tasks.

---

## 1. Introduction
The integration of deep learning into physics-based character animation has led to significant advancements in generating agile and natural motion. To increase versatility, recent approaches have focused on learning reusable skill embeddings. Characters initially learn various skill embeddings by imitating reference motions and then apply these skills to accomplish diverse tasks. 

However, these approaches often suffer from mode collapse during the task training phase when differences between skills are subtle. Mode collapse restricts the agents' potential in scenarios that require a diverse set of skills and restricts exploration during RL training, resulting in sub-optimal task performance. Furthermore, agents have generally not been equipped with the capability to employ different strategies to adapt to complex and dynamic environments.

### Contributions
* A hierarchical skill controller that empowers physically simulated agents to explicitly perform various skills, enabling rapid skill transitions.
* An interaction learning framework designed to create a decision strategy allows agents to continually learn and adapt to competition or cooperation.
* Novel results demonstrating the framework's capacity to generate intelligent decisions in two scenarios: agent-agent interactions in simulation and human-agent interactions in VR.

---

## 2. Related Work

### 2.1 Physics-based Character Animation
Incorporating physical laws into character animation allows for the development of controllers that generate more realistic behaviors. Deep reinforcement learning (DRL) methods eliminate the need for designing complex objective functions while delivering outstanding results. Recently, much attention has been paid to reusable motor skills through latent models like conditional variational autoencoders (VAE) and vector-quantized VAEs. [cite_start]While previous systems have explored physically simulated boxing or kinematics-based tennis, our method learns not only agile motor control to strike the ball but also strategies to select skills and targets [cite: 93-95, 97].

### 2.2 Transition of Skills
Option-based methods represent skills as options, which are sequentially constructed. Behavior Trees are also a common method for planning the transition between different states. While these methods work well for tasks that are not time-sensitive, table tennis poses a challenge as players do not always hit the ball from a well-defined initial state.

### 2.3 Human-agent Interaction
Commercial games like Eleven Table Tennis allow humans to interact with an agent in VR, but often simulate only a floating head and paddle rather than full-body dynamics. Advances in GPU-accelerated simulation enable us to create a physically-simulated agent with full-body dynamics that can play in real-time with humans.

---

## 3. Method Overview
We propose a hierarchical approach that includes a strategy-level controller and a skill-level controller. 
* **Strategy-level controller:** Takes the states of the agent, opponent, and ball as inputs, and outputs a strategy action, which includes the skill to use and the target landing location.
* **Skill-level controller:** Takes the states of the agent and ball, along with the strategy action as inputs, and generates a skill action, which includes the target joint angles for PD controllers.

---

## 4. Skill-level Controller
[cite_start]Training the skill-level controller requires three stages: training imitation policies from motion capture data, learning a ball control policy for each skill, and learning a mixer policy for plausible transitions [cite: 134-136].

### 4.1 Imitation Policy
We categorize the motion capture dataset into five subsets corresponding to each skill and utilize all the data to train a universal imitation policy. The imitation policy is represented as $\pi^{i}(a^{i}|s,z^{i})$, where $i \in \{1,2,3,4,5,u\}$. Using a single universal imitation policy often leads to mode collapse; our mixture-of-experts inspired controller mitigates this. The policy is updated by minimizing the discriminator objective:
$$\min_{D^{i}}-\mathbb{E}_{d_{M^{i}}(s,s^{\prime})}\log(D^{i}(s,s^{\prime}))-\mathbb{E}_{d_{\pi^{i}}(s,s^{\prime})}\log(1-D^{i}(s,s^{\prime}))$$

### 4.2 Ball Control Policy
We train ball control policies $\omega^{i}(z^{i}|s,b,y)$ to enable the agent to hit and move a ball to the desired location. The task reward $r$ is a composite of three terms:
$$r(t) = w_{p}r_{p}(t) + w_{b}r_{b}(t) + w_{s}r_{s}(t)$$

### 4.3 Mixer Policy
To create plausible transitions among the different skills, we learn a mixer policy $\omega^{m}(z^{m}|s,b,\delta,y)$. The target joint angles for PD controllers are computed as:
$$a = \varphi \odot \pi^{u}(\cdot|s,z^{u}) + (1-\varphi) \odot \sum_{i=1}^{5}\delta_{i}\pi^{i}(\cdot|s,z^{i})$$

---

## 5. Strategy-level Controller
The strategy-level controller is developed by iterative behavior cloning. We collect interaction data by randomly sampling strategy actions during gameplay, then update the controller iteratively. We utilize a Conditional Variational Autoencoder (CVAE) to model the stochastic nature inherent in sports gameplay. The training loss is defined as:
$$\sum_{k=1}^{K}||c_{k}^{expert}-c_{k}^{\prime}||+\beta_{KL}D_{KL}(Q(u|\mu_{k},\sigma_{k}^{2})||\mathcal{N}(0,I))$$

> **Algorithm 1: Strategy Learning**
> * **Input:** Number of iterations N, interaction environment Env.
> * **Output:** Updated policy f.
> * **Steps:** Initialize f randomly, interact with Env to collect expert data, and apply stochastic gradient descent to update f.

---

## 6. Interaction Environment
* **Agent-agent interaction:** Two virtual agents play table tennis. The opponent uses fixed heuristic strategies (a random strategy or a strategy built from broadcast videos) while our agent iteratively updates.
* **Human-agent interaction:** A user interacts with an agent using a VR device and a physically simulated paddle.

---

## 7. Experiments

### 7.1 Skill Evaluation
We compare our method with ASE, CASE, and an explicit transition model (ET). 

**Table 1: Comparisons on Discriminator Score, Skill Accuracy, and Diversity Score.**

| Metric | ASE | CASE | ET | Ours |
| :--- | :--- | :--- | :--- | :--- |
| Discriminator Score | 1.62 | 2.28 | 4.95 | **5.72** |
| Skill Accuracy | 0.38 | 0.47 | 0.69 | **0.76** |
| Diversity Score | 6.13 | 6.05 | 7.32 | **8.01** |
*Data derived from.*

**Table 2: Task performance evaluation.**

| Metric | ASE | CASE | ET | Ours |
| :--- | :--- | :--- | :--- | :--- |
| Avg Hits | 9.54 (5.94) | 8.79 (5.28) | 6.55 (3.66) | **10.93 (6.28)** |
| Avg Error | 0.28 (0.33) | 0.35 (0.39) | **0.25 (0.28)** | 0.26 (0.31) |
[cite_start]*Data derived from [cite: 352-356].*

### 7.2 Agent-Agent Interaction
[cite_start]We evaluate the performance in competition (higher winning rate) and cooperation (longer rallies) settings against an RL baseline [cite: 405-407].

**Table 3: Strategy Evaluation (Winning Rates & Average Rounds).**

| Opponent | Competition (Ours vs RL) | Cooperation (Ours vs RL) |
| :--- | :--- | :--- |
| Random op | **0.641** vs 0.687 | **14.9** vs 16.4 |
| Video op | **0.637** vs 0.681 | **15.6** vs 18.2 |
*Data derived from. Note: Our method consistently outperformed the RL baseline when pitted directly against it (see Table 4).*

### 7.3 Human-Agent Interaction
We finetuned the skill-level controller using VR play data due to the domain gap between the simulated agent and human styles. 

**Table 5: Evaluation of Human-agent Interaction.**

| Metric | Initial policies | Competition | Cooperation |
| :--- | :--- | :--- | :--- |
| Winning rate | 0.64 | 0.78 | 0.58 |
| Avg hits | 4.04 | 3.75 | 5.34 |
*Data derived from.*

---

## 8. Discussion and Conclusion
We introduced a learning approach for physics-based table tennis animation that utilizes a hierarchical controller structure to overcome mode collapse. While effective, the method has limitations: it does not scale seamlessly to datasets with hundreds of skills, final motion quality heavily relies on the captured motion style (e.g., relying on large arm motions), and the simulation lacks aerodynamic models like the Magnus effect. Nonetheless, it learns effective decision strategies for both agent-agent and human-agent environments.