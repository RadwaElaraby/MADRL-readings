

# ** Counterfactual multi-agent policy gradients. (Foerster et al., 2018) - COMA

![](imgs/foerster17_COMA_architecture.PNG)

COMA 
- centralisation of the critic
  - the critic is only used during learning, while only the actor is needed during execution
  - the critic conditions on the true global state s, if available, or the joint action-observation histories, while each agent’s policy/actor conditions only on each agent's own action-observation history

- use of a counterfactual base-line
  - naive approach: follow a gradient based on the TD error estimated from this critic (gradient for a particular agent does not explicitly reason about how that particular agent’s ac-tions contribute to that global reward)
![](imgs/foerster17_naive_approach.PNG)

  - instead, it uses a counterfactual baseline (inspired by difference rewards which compares the global reward to the reward received when that agent’s action is replaced with a default action)
  - the centralised critic can be used to implement difference rewards (without any further simulations or complications)
  - For each agent a, we can compute an agent-specific advantage function that compares the Q-value for the current joint action u to a counterfactual baseline that marginalises out a single agent’s action u<sup>a</sup>, while keeping the other agents’ actions u<sup>-a</sup> fixed
![](imgs/foerster17_counterfactual_baseline.PNG)
  
  - thus, it computes a separate baseline for each agent that relies on the centralised critic to reason about counterfactuals in which only that agent’s action changes

- use of a critic representation that allows efficient evaluation of the baseline
  - the actions of the other agents, u<sub>t</sub><sup>−a</sup>, are part of the input to the critic network, which then outputs a Q-value for each of agent a’s actions. In other words, the critic computes in a single forward the Q-values for all the different actions of a given agent, conditioned on the actions of all the other agents
  - counterfactual advantage can be calculated efficiently by a single forward pass of the actor and critic for each agent

They also introduced 2 variatns of independent actor-critic (IAC)
- each agent learn independently, with its own actor and critic conditions only on its own action-observation history
- speed learning by sharing parameters among the agents
- the critic in the 1st variant estimates V while it estimates Q in the 2nd variant



COMA can significantly improve performance over other multi-agent actor-critic methods

COMA’s best agents are competitive with state-of-the-art centralised controllers that are given access to full state information and macro-actions



---
# *** Multi-agent actor-critic for mixed cooperative-competitive environments. (Lowe et al., 2017)

adopt the framework of centralized training with decentralized execution

a general-purpose multi-agent learning algorithm, which can be applied not only to cooperative interaction but to competitive or mixed interaction involving both physical and communicative behavior

![](imgs/lowe17_architecture.PNG)

a simple extension of actor-critic policy gradient methods where the critic is augmented with extra information about the policies of other agents, while the actor only has access to local information. 

only the local actors are used at execution phase, acting in a decentralized manner

The gradient of the expected return for agent i can be written as 
![](imgs/lowe17_gradient.PNG)

where Q<sup>π</sup><sub>i</sub>(x,a<sub>1</sub>,...,a<sub>N</sub>) is a centralized action-value function. in the simplest case, x consists of the observations of all agents, but it could also include additional state information

To remove the assumption of knowing other agents’ policies, each agent can maintain an approximation to the true policy of each other agent and use them in their own policy learning procedure

One downside is that the input space of Q grows linearly with the number of agents N. This could be remedied by, for example, having a modular Q function that only considers agents in a certain neighborhood of agiven a gent


vs COMA:
- learn a centralized critic for each agent (allowing differing reward functions) whereas COMA learns a single centralized critic for all agents 
- learn continuous policies whereas COMA learns discrete policies 


---
# ** Cooperative multi-agent control using deep reinforcement learning. (Gupta et al., 2017)





















---
<!-- ## Stabilising experience replay for deep multi-agent reinforcement learning. (Foerster et al. 2017) -->
