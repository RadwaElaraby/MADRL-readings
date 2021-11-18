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

Settings & Goal: 
- learn a discrete communication signal among agents in a fully cooperative, partially observable settings
- At each timestep, agents take an environment action **u<sub>t</sub>** and communication action **m<sub>t</sub>** a discrete communication signal to other agents via a communication channel. Agents should then use their local observation accompanied with the messages from other agents to select their actions.

Approach: they developed 2 methods:
- Reinforced Inter-Agent Learning (RIAL):  
  - agents use two deep Q networks **Q<sup>a</sup><sub>u</sub>(o<sup>a</sup><sub>t</sub>, m<sup>a'</sup><sub>t-1</sub>, h<sup>a</sup><sub>t-1</sub>, u<sup>a</sup>)** and **Q<sub>m</sub>(.)** to predict Q-values for environment and communication actions respectively, where:
    - **o<sup>a</sup><sub>t</sub>** is the agent's local observation, 
    - **m<sup>a'</sup><sub>t-1</sub>** are messages from other agents,
    - **h<sup>a</sup><sub>t-1</sub>** is the hidden state (recurrent neural network)
  - the Q values are then passed to an action selector unit to pick **u<sup>a</sup><sub>t</sub>** and **m<sup>a</sup><sub>t</sub>** (using an ε-greedy policy).
  - it's end-to-end trainable within an agent (no gradients are passed between agents)
  - gradient chains are based on the DQN loss

- Differentiable Inter-Agent Learning (DIAL):
  - communication actions are replaced with direction connections from the output of one agent's network to the input of another
  - gradients can be pushed through the communication channel, yielding a system that is end-to-end trainable even across agents
  - gradient chains are based on both the DQN loss and the backpropagated error from the recipient of the message to the sender

![](imgs/Foerster16_rial_dial.PNG)


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

