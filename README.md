# MADRL-readings


### Multi-agent actor-critic for mixed cooperative-competitive environments. (Lowe et al., 2017)

### Counterfactual multi-agent policy gradients. (Foerster et al., 2018)

### Deep decentralized multi-task multi-agent rl under partial observability. (Omidshafiei et al., 2017)

### Cooperative multi-agent control using deep reinforcement learning. (Gupta et al., 2017)

### Fully decentralized multiagent reinforcement learning with networked agents. (Zhang et al. 2018) 








### Stabilising experience replay for deep multi-agent reinforcement learning. (Foerster et al. 2017)

### Mean field multi-agent reinforcement learning. (Yang et al., 2018).



---
## Learning to communicate with deep multi-agent reinforcement learning.(Foerster et al., 2016).

Goal: learn communication protocols among agents in a fully cooperative, partially observable settings

Approach: they use a deep Q network

They developed 2 methods:
- Reinforced Inter-Agent Learning (RIAL):  
  - end-to-end trainable within an agent (no gradients are passed between agents).
  - they trains two deep Q networks Q<sup>a</sup><sub>u</sub>(o<sup>a</sup><sub>t</sub>, m<sup>a'</sup><sub>t-1</sub>, h<sup>a</sup><sub>t-1</sub>, u<sup>a</sup>) and Q<sub>m</sub>(.) to predict values for environment and communication actions respectively, where:
    - o<sup>a</sup><sub>t</sub> represents agent's local observation, 
    - m<sub>t-1</sub><sup>a'</sup> is the received message from the previou
  - the agent uses an action selector to picks uat and mat from Q<sub>u</sub> and Q<sub>m</sub>, using an \epsilon-greedy policy.

- Differentiable Inter-Agent Learning (DIAL):
  - gradients can be pushed through the communication channel, yielding a system that is end-to-end trainable even across agents.


They followed a centralized training decentralized execution paradigm. 

Centralized training is achieved via:
- parameter sharing (learning a single network) in RIAL.
- parameter sharing and pushing gradients across agents through the communication channel in DIAL.


---


### Learning multiagent communication with backpropagation. (Sukhbaatar et al., 2016).

### Multiagent bidirectionally-coordinated nets: Emergence of human-level coordination in learning to play starcraft combat games. (Peng et al., 2017)

### Revisiting the master-slave architecture in multi-agent deep reinforcement learning. (Kong et al., 2017),








### Learning attentional communication for multi-agent cooperation. (Jiang & Lu, 2018)

### Learning when to communicate at scale in multiagent cooperative and competitive tasks. (Singh et al., 2018)

