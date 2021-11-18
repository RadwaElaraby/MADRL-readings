# MADRL-readings


### Multi-agent actor-critic for mixed cooperative-competitive environments. (Lowe et al., 2017)

### Counterfactual multi-agent policy gradients. (Foerster et al., 2018)

### Deep decentralized multi-task multi-agent rl under partial observability. (Omidshafiei et al., 2017)

### Cooperative multi-agent control using deep reinforcement learning. (Gupta et al., 2017)

### Fully decentralized multiagent reinforcement learning with networked agents. (Zhang et al. 2018) 








### Stabilising experience replay for deep multi-agent reinforcement learning. (Foerster et al. 2017)

### Mean field multi-agent reinforcement learning. (Yang et al., 2018).



----------
### Learning to communicate with deep multi-agent reinforcement learning.(Foerster et al., 2016).

Settings: fully cooperative, partially observable

Approach: they use a deep Q network and developed 2 methods
1. Reinforced Inter-AgentLearning (RIAL):  
end-to-end trainable withinan agent (no gradients are passed between agents)

2. Differentiable Inter-Agent Learning (DIAL) 
gradients can be pushed through thecommunication channel, yielding a system that is end-to-end trainable even across agents

They followed a centralized training decentralized execution paradigm. Centralized training is done via parameter sharing (learning only one network for all agents) in RIAL and via both parameter sharing and pushing gradients across agents through the communication channel.

----------

---

--

### Learning multiagent communication with backpropagation. (Sukhbaatar et al., 2016).

### Multiagent bidirectionally-coordinated nets: Emergence of human-level coordination in learning to play starcraft combat games. (Peng et al., 2017)

### Revisiting the master-slave architecture in multi-agent deep reinforcement learning. (Kong et al., 2017),








### Learning attentional communication for multi-agent cooperation. (Jiang & Lu, 2018)

### Learning when to communicate at scale in multiagent cooperative and competitive tasks. (Singh et al., 2018)

