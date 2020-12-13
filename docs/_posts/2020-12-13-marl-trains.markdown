---
layout: post
title:  "Multi-Agent Reinforcement Learning on Trains"
date:   2020-12-13 15:08:56 +0530
categories: rl
use_math: true
---



## Multi-Agent RL

Reinforcement learning has been successfully applied to solving tasks in various fields such as game playing and robotics. However, the most successful settings have predominantly been single-agent systems, where the behavior of other actors is not taken into account. Like our problem statement, many real-world environments are multi-agent settings that involve interaction between multiple agents. Multi-Agent RL is an active field of research with potentially significant impact on RL in real-world. In this section, we discuss aspects of multi-agent environments that theoretically differentiate between single-agent and multi-agent settings.

### Concepts

**Markov Games** are a multi-agent extension of Markov Decision Processes (MDPs) that are used to model single-agent systems. A Markov Game is tuple $$ \langle S, A_1, ..., A_N, O_1, ..., O_N \rangle $$ where:

$$N$$ = number of agents   
$$S$$ = set of states (all possible configurations of all agents)   
$$A_1, ..., A_N$$ = set of actions of the agents   
$$O_1, ..., O_N$$ = set of observations of the agents   

![MDP vs MG]({{ site.baseurl }}/images/rl/mdp_vs_mg.png)

Each agent follows a stochastic policy $$\pi_{\theta_i} : O_i \times A_i \rightarrow [0, 1]$$ while transitioning according to the state transition function $$T : S \times A_1 \times ... A_n \rightarrow S$$ and receives a reward defined by the reward function $$r_i : S \times A_i \rightarrow \R$$. The aim of all agents is to maximize their total expected return $$R_i = \sum_{t = 0}^{T} \gamma^{t} r_i^t$$ over time $$T$$, with a discount factor $$\gamma \in [0, 1]$$.

In contrast to single-agent RL, the value function for every agent is a function of the joint policy. Therefore, intuitively, we can see that the solution of a Markov Game would be different from that of a Markov Decision Process because the optimal policy of an agent may be dependent on the policies of all other agents. One such solution that characterizes the optimal joint policy is the **Nash Equilibrium**. The equilibrium has been proved to always exist in discounted Markov Games, but may not be unique. MARL algorithms are designed to converge to such an equilibrium.

MARL settings can be mainly characterized into the following:
- Cooperative : agents share a common reward
- Competitive : one agent's loss is other's gain; modeled as a zero-sum Markov Game
- Mixed


### Challenges

Simultaneous learning by all agents can make the environment **non-stationary** from the view of a single agent. This is so, since the action taken by one agent can influence the decisions made by others and the evolution of the state. Hence, it is necessary to take into account joint behavior of the agents. This is in contrast with the single-agent setting which violates the stationarity assumption (Markov property).

To account for non-stationarity, we may attempt to tackle MARL from a combinatorial perspective by modeling the joint action space of all agents. Such a modeling exponentially increases the dimension of the action space and is hence cursed by dimensionality. This casuses an issue of **scalibilty** on the computation side and also complicates theoretical convergence guarantees.


