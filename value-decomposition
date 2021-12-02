# MADRL-readings





## Value-Decomposition Networks For CooperativeMulti-Agent Learning (Sunehag et al., 2017)

a system of several learning agents must jointly optimize a single reward signal. The centralised approach fails by learning inefficient policies with only one agent active and the other being “lazy”. In contrast, independent Q-learners cannot distinguish teammates’ exploration from stochasticity in the environment

The value decomposition network aims to learn an optimal linear value decomposition from the team reward signal, by back-propagating the total Q-gradient through deep neural networks representing the individual component value functions

The main assumption is that the joint action-value function for the system can be additively decomposed into value functions across agents

![](imgs/sunehag2017_VDN_vs_I.PNG)

![](imgs/sunehag2017_VDN.PNG)

Agent’s learning algorithm is based on DQN (LSTM for partial observability)

their approach can be nicely combined with weight sharing and information channels.

They have experimented with different approaches including low-level communication channel, high-level communication channel, centralized, individual. They found that the architectures based on value-decomposition perform much better

Notes
- The implicit value function learned by each agent depends only on local observations, 
- Learns in a centralised fashion at training time, while agents can be deployed individually



---
