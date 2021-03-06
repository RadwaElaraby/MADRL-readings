# MADRL-readings
*Most of the text in this repo is copied from the mentioned papers with minimal adaptations*


General interests: <!--dealing with nonstationary when moving into multi-agent settings-->
- learn communication between agents
- centralized training decentralized execution (CTDE) approaches
- credit assignment problem



---

## Learn Communication 

#### Learning to communicate with deep multi-agent reinforcement learning.(Foerster et al., 2016). - RIAL & DIAL

- learn communication signal among agents in a fully cooperative, partially observable settings.

#### Learning multiagent communication with backpropagation. (Sukhbaatar et al., 2016). - CommNet

#### Learning when to communicate at scale in multiagent cooperative and competitive tasks. (Singh et al., 2018)

- learn what to communicate and when to communicate (allow agents to block their communication using a gating mechanism)

#### Learning attentional communication for multi-agent cooperation. (Jiang & Lu, 2018) - ATOC

- enables **dynamic** communication among agents only when necessary
- when communication is needed, a communication group is formed by selecting at most m (fixed bandwidth) agents from agent's observable field based on proximity
- then, the agents within that communication group are allowed to share information for number of timesteps

#### TarMAC: Targeted Multi-Agent Communication (Das et al., 2020)

- not only learn what to send, but also to whom to address
- uses a signature-based soft attention mechanism to allow agents choose which agents to address messages to 

---

## CTDE via Parameter Sharing 

#### Cooperative multi-agent control using deep reinforcement learning. (Gupta et al., 2017) 

- extends single deep learning algorithm into multi-agent settings and evaluate them
- centralizing training -> parameter sharing

## CTDE via Centralized Critic

#### Counterfactual multi-agent policy gradients. (Foerster et al., 2018) - COMA

- centralized critic conditions on the joint action-observation histories 
- learns a single centralized critic for all agents
- only cooperative settings

#### Multi-agent actor-critic for mixed cooperative-competitive environments. (Lowe et al., 2017)

- extends COMA to competitve and mixed settings by learning a centralized critic for each agent

#### Actor-Attention-Critic for Multi-Agent Reinforcement Learning (Iqbal et al., 2019)

- support the centralized critic with attention mechanism to dynamically select which agents to attend to 

---

## Credit Assignment via Value Decomposition (value-based)

#### Value-Decomposition Networks For Cooperative Multi-Agent Learning (Sunehag et al., 2017) - VDN

- assumption is that joint action-value function for the system can be additively decomposed into value functions across agents
- represents Q_tot as a sum of individual value functions Q_a that conditions only on individual observations and actions 

#### QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning (Rashidet al., 2018)

- generalize VDN
- only need to ensure that a global argmax performed on Q_tot yields the same result as a set of individual argmax operations performed on each Q_a (monotonic)

#### Randomized Entity-wise Factorization for Multi-Agent Reinforcement Learning (Iqbal al., 2021)

- incorporate knowledge of shared patterns to accelerate learning in a multi-task setting
