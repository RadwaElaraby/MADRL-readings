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

Goal: 
- learn a discrete communication signal among agents in a fully cooperative, partially observable settings.

Description:
- at each timestep t, each agent receives a local observation **o<sub>t</sub>** and discrete message signals from other agents **m<sub>t-1</sub>**
- then, it should take an environment action **u<sub>t</sub>** and a communication action **m<sub>t</sub>** 
- the communication action is sent to other agents via a communication channel. 

Method: 
- Reinforced Inter-Agent Learning (RIAL):  
  - agents use two deep Q networks **Q<sup>a</sup><sub>u</sub>(o<sup>a</sup><sub>t</sub>, m<sup>a'</sup><sub>t-1</sub>, h<sup>a</sup><sub>t-1</sub>, u<sup>a</sup>)** and **Q<sub>m</sub>(.)** to predict Q-values for environment and communication actions respectively, where:
    - **o<sup>a</sup><sub>t</sub>** is the agent's local observation, 
    - **m<sup>a'</sup><sub>t-1</sub>** are messages from other agents,
    - **h<sup>a</sup><sub>t-1</sub>** is the agent's hidden state (RNN to overcome partial observability)
  - the Q values are then passed to an action selector unit to pick **u<sup>a</sup><sub>t</sub>** and **m<sup>a</sup><sub>t</sub>** (using an ε-greedy policy).
  - it's end-to-end trainable within an agent (no gradients are passed between agents)
  - gradient chains are based on the DQN loss

- Differentiable Inter-Agent Learning (DIAL):
  - communication actions are replaced with direction connections from the output of one agent's network to the input of another
  - gradients can be pushed through the communication channel, yielding a system that is end-to-end trainable even across agents
  - gradient chains are based on both the DQN loss and the backpropagated error from the recipient of the message to the sender

![](imgs/foerster16_rial_dial.PNG)


They followed a centralized training decentralized execution paradigm. 

Centralized training is achieved via:
- parameter sharing (learning a single network) in RIAL.
- parameter sharing and pushing gradients across agents through the communication channel in DIAL.

---


### Learning multiagent communication with backpropagation. (Sukhbaatar et al., 2016).

Goal:
- coordination between agents in fully cooperative, partially observable settings by learning suitable communication between them

Description: 
- learns a single controller **Φ** that maps agents' states **s = {s<sub>1</sub>, ..., s<sub>J</sub>}** into their actions **a = {a<sub>1</sub>, ..., a<sub>J</sub>}** where **J** is the number of agents
- **Φ** consists of individual modules **f<sup>i</sup>** where i refers to the communication step
- each module **f** takes two input vectors for each agent **j**: the hidden state **h<sup>i</sup><sub>j</sub>** and the communication **c<sup>i</sup><sub>j</sub>** and outputs a vector **h<sup>i+1</sup><sub>j</sub>**
- the output vectors **h<sup>i+1</sup>** are then averaged according to the following formula to compute the communication for the next step **c<sup>i+1</sup><sub>j</sub>**

<p style="text-align:center;">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;c^{i+1}_j=\frac{1}{J-1}\sum\limits_{j%27\neq%20j}{h^{i+1}_{j%27}}" adjust="" />
</p>


![](imgs/sukhbaatar16_commNet.PNG)

---

### Multiagent bidirectionally-coordinated nets: Emergence of human-level coordination in learning to play starcraft combat games. (Peng et al., 2017)

### Revisiting the master-slave architecture in multi-agent deep reinforcement learning. (Kong et al., 2017),








### Learning attentional communication for multi-agent cooperation. (Jiang & Lu, 2018)

### Learning when to communicate at scale in multiagent cooperative and competitive tasks. (Singh et al., 2018)

