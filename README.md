# MADRL-readings


## Multi-agent actor-critic for mixed cooperative-competitive environments. (Lowe et al., 2017)

---
## Counterfactual multi-agent policy gradients. (Foerster et al., 2018)

---
## Deep decentralized multi-task multi-agent rl under partial observability. (Omidshafiei et al., 2017)

---
## Cooperative multi-agent control using deep reinforcement learning. (Gupta et al., 2017)

---
## Fully decentralized multiagent reinforcement learning with networked agents. (Zhang et al. 2018) 



---





## Stabilising experience replay for deep multi-agent reinforcement learning. (Foerster et al. 2017)

---
## Mean field multi-agent reinforcement learning. (Yang et al., 2018).



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


## Learning multiagent communication with backpropagation. (Sukhbaatar et al., 2016).

Goal:
- coordination between agents in fully cooperative, partially observable settings by learning suitable communication between them
- communication is learned rather than being pre-determined

Description: 
- learns a single controller **Φ** that maps agents' states **s = {s<sub>1</sub>, ..., s<sub>J</sub>}** into their actions **a = {a<sub>1</sub>, ..., a<sub>J</sub>}** where **J** is the number of agents
- **Φ** consists of individual modules **f<sup>i</sup>** where i refers to the communication step
- each module **f** takes two input vectors for each agent **j**: the hidden state **h<sup>i</sup><sub>j</sub>** and the communication **c<sup>i</sup><sub>j</sub>** and outputs a vector **h<sup>i+1</sup><sub>j</sub>** using the following formula (works as a signle linear layer followed by a non-linearity)

<img src="https://latex.codecogs.com/svg.latex?\Large&space;h^{i+1}_j=\sigma(H^i*h^i_j%20+%20C^i*c^i_j)" />


- the hidden vectors **h<sup>i+1</sup>** from agents are then averaged according to the following formula to compute the communication for the next step **c<sup>i+1</sup><sub>j</sub>**

<img src="https://latex.codecogs.com/svg.latex?\Large&space;c^{i+1}_j=\frac{1}{J-1}\sum\limits_{j%27\neq%20j}{h^{i+1}_{j%27}}"  />

- the hidden vectors **h<sup>i</sup>** are also used to generate a distribution over the space of actions at the output layer

![](imgs/sukhbaatar16_commNet.PNG)

- as an alternative to broadcasting to all other agents, a mask could be used to choose only a certain range of agents to communicate with 

[Implementation](https://github.com/0b01/CommNet/tree/b826f00e21c22f38bea288ca2ee8cc15e2dde1eb)

---

## Multiagent bidirectionally-coordinated nets: Emergence of human-level coordination in learning to play starcraft combat games. (Peng et al., 2017)

---
## Revisiting the master-slave architecture in multi-agent deep reinforcement learning. (Kong et al., 2017),







---

## Learning attentional communication for multi-agent cooperation. (Jiang & Lu, 2018)

---
## Learning when to communicate at scale in multiagent cooperative and competitive tasks. (Singh et al., 2018)



---

## TarMAC: Targeted Multi-Agent Communication (Das et al., 2020)

Goal: not only learn what to send, but also to whom to address (in cooperative partially-observable settings)

Description:
- allows agents to choose which agents to address messages to using a signature-based soft attention mechanism
- at each timestep, each agent receives a local observation vector **w<sup>t</sup><sub>i</sub>** and an aggregated message vector **c<sup>t</sup><sub>i</sub>** from all other agents, and update its hidden state **h<sup>t</sup><sub>i</sub>** accordingly 
- the agent uses its policy **π<sub>Θ<sub>i</sub></sub>(a<sup>t</sup><sub>i</sub>|h<sup>t</sup><sub>i</sub>)** to output a probability distribution over its actions, and to also output an outgoing message vector **m<sup>t</sup><sub>i</sub>**

![](imgs/das20_policy_network.PNG)

- each message consists of a signature **k<sup>t</sup><sub>i</sub>** (encoding the properties of the recipient) and a value **v<sup>t</sup><sub>i</sub>**
- at the receiving side, each agent **j** predicts a query vector **q<sup>t+1</sup><sub>j</sub>** from its hidden state **h<sup>t+1</sup><sub>j</sub>** 

![](imgs/das20_targeted_communication.PNG)

- the query vector **q<sup>t+1</sup><sub>j</sub>** is multiplied by the received signature vector **k<sup>t</sup><sub>i</sub>** to compute the attention weights **α<sub>ij</sub>** which is then multiplied by the received value vector **v<sup>t</sup><sub>i</sub>** to compute the input message **c<sup>t+1</sup><sub>j</sub>** for the agent at **t+1**

![](imgs/das20_weights.PNG)

![](imgs/das20_messages.PNG)

- 

Side notes: 
- centralized training and decentralized execution paradigm
- uses actor critic algorithm with a centralized critic learnt over the agents' joint action space
- policy parameters are shared across agents
- the applied targeting mechanism is implicit (encode properties of recipients without addressing them explicitly)
- supports multiple rounds of interactions at every timestep 
- can be extended into competitve settings by combining it with IC3Net 











