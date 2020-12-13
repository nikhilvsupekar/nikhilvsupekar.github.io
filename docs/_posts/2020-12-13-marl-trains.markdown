---
layout: post
title:  "Multi-Agent Reinforcement Learning on Trains"
date:   2020-12-13 15:08:56 +0530
categories: rl
use_math: true
---



## **Problem Description**

This project’s main focus is to solve the problem of railway traffic control. One of the major bottlenecks faced by the transportation and logistic companies around the world right now is the punctuality of all their train operations. Punctuality not only involves regularity in time but also relies on several factors like ensuring safety and providing reliability of service. [1] Through this project, we aim to work on this domain by shifting solutions from traditional Operations Research techniques to a Reinforcement Learning approach!

We make use of the environment provided in the flatland challenge that is built to tackle this exact problem of scheduling and rescheduling trains. They provide a grid-world environment that will be described in detail over the next section. The major moving parts in this problem, for now, are the trains (which are generally referred to as agents throughout this blog), stations (to which specific trains have to arrive at), and tracks (that form a network that connects different stations).

The goal of this project is not only to make the trains reach their destination without suffering from deadlocks or traffic accidents, but it is also to minimize the time required to reach the destination. Since most of the approaches to solve this problem efficiently were previously based on Operations Research techniques, through this project, we test various single-agent and multi-agent reinforcement learning algorithms and compare their efficiency.




## **Environment**


The environment used for this project can be found here. The major features of this environment are:

### The Observation Space

There are two forms of observation spaces currently supported by the flatland environment. 

The first observation type is the Global Observation space, which is very similar to the raw-pixel type of observation spaces used in the Atari games, with a slight difference. The global observation space is h x w x c, dimensional where h is the height of the environment, w is the width and c represents the channels. Instead of raw pixel values, each channel (total 5) provides some information about the current state. For example, channel 0 contains the one-hot encoding representation of an agent’s current position and direction, channel 1 contains information about other agent’s position and direction, etc.

The second observation type is the Tree Observation space, which makes use of the network/graph-like structure of the environment. The tree is 4-ary tree with each node’s 4 child edges representing nodes in the Left, Forward, Right, Backward direction. Each node in the tree consists of 12 features.


### The Action Space

The action space for this problem is discrete having 5 possible options:

0: DO NOTHING   
1: DEVIATE LEFT   
2: GO FORWARD   
3: DEVIATE RIGHT   
4: STOP   

One thing to note in this environment is that not all actions are possible at any given moment because the agent is restricted by the structure of the network.

### The complexity control

This environment contains various properties that can be configured to control the complexity of the resulting network. Some of these properties are,

malfunction_rate: Poisson rate at which malfunctions happen for an agent   
max_num_cities: Maximum number of stations in on observation   
grid_mode: random vs uniform distribution of cities in a grid   
max_rails_between_cities: Maximum number of rails connecting cities   
max_rails_in_city: Maximum number of parallel tracks inside the city   
observation_tree_depth: The depth of the observation tree   

For this project, the complexity used was simple as we wanted to test the performance of different algorithms in a limited time and resources. Here are the parameters we used for reference:

malfunction_rate = 2%   
max_num_cities = 2   
grid_mode = False   
max_rails_between_cities = 2   
max_rails_in_city = 3   
observation_tree_depth = 2   





## **Multi-Agent RL**

Reinforcement learning has been successfully applied to solving tasks in various fields such as game playing and robotics. However, the most successful settings have predominantly been single-agent systems, where the behavior of other actors is not taken into account. Like our problem statement, many real-world environments are multi-agent settings that involve interaction between multiple agents. Multi-Agent RL is an active field of research with potentially significant impact on RL in real-world. In this section, we discuss aspects of multi-agent environments that theoretically differentiate between single-agent and multi-agent settings.

### Concepts

**Markov Games** are a multi-agent extension of Markov Decision Processes (MDPs) that are used to model single-agent systems. A Markov Game is tuple $$ \langle S, A_1, ..., A_N, O_1, ..., O_N \rangle $$ where:

$$N$$ = number of agents   
$$S$$ = set of states (all possible configurations of all agents)   
$$A_1, ..., A_N$$ = set of actions of the agents   
$$O_1, ..., O_N$$ = set of observations of the agents   

![MDP vs MG]({{ site.baseurl }}/images/rl/marl-trains/mdp_vs_mg.png)

Each agent follows a stochastic policy $$\pi_{\theta_i} : O_i \times A_i \rightarrow [0, 1]$$ while transitioning according to the state transition function $$T : S \times A_1 \times ... A_n \rightarrow S$$ and receives a reward defined by the reward function $$r_i : S \times A_i \rightarrow \R$$. The aim of all agents is to maximize their total expected return $$R_i = \sum_{t = 0}^{T} \gamma^{t} r_i^t$$ over time $$T$$, with a discount factor $$\gamma \in [0, 1]$$.

In contrast to single-agent RL, the value function for every agent is a function of the joint policy. Therefore, intuitively, we can see that the solution of a Markov Game would be different from that of a Markov Decision Process because the optimal policy of an agent may be dependent on the policies of all other agents. One such solution that characterizes the optimal joint policy is the **Nash Equilibrium**. The equilibrium has been proved to always exist in discounted Markov Games, but may not be unique. MARL algorithms are designed to converge to such an equilibrium.

MARL settings can be mainly characterized into the following:
- Cooperative : agents share a common reward
- Competitive : one agent's loss is other's gain; modeled as a zero-sum Markov Game
- Mixed


### Challenges

Simultaneous learning by all agents can make the environment **non-stationary** from the view of a single agent. This is so, since the action taken by one agent can influence the decisions made by others and the evolution of the state. Hence, it is necessary to take into account joint behavior of the agents. This is in contrast with the single-agent setting which violates the stationarity assumption (Markov property).

To account for non-stationarity, we may attempt to tackle MARL from a combinatorial perspective by modeling the joint action space of all agents. Such a modeling exponentially increases the dimension of the action space and is hence cursed by dimensionality. This casuses an issue of **scalibilty** on the computation side and also complicates theoretical convergence guarantees.



### Solutions

We explore two paradigms that MARL algorithms employ for overcoming these challenges. The first and a naive approach is to consider all agents independent and learn Q-values for each of them. This approach is known as **Independent Q-Learning** (IQL). From the view of one single agent, other agents are considered as part of the environment. As we can see intuitively, the algorithm is not learning any explicit co-operation amongst the agents and is not expected to converge. However, in practice, it ends up working quite well in some cases. This suggests the possible use of IQL as baselines for multi-agent systems.

A second and a recently popular approach is based on the principle of **Centralized Training and Decentralized Execution**. The key trick here is that communication between agents is not restricted during learning, however the individual policies for all agents are limited in communication during execution. This is a reasonable paradigm because it can also be applied to some real-world settings.




## **Results**

We present the results from D3QN, Rainbow DQN and MAAC. We report the metric 'normalized score' as is used in the Flatland challenge.

<div align="center" markdown="1">
![Results]({{ site.baseurl }}/images/rl/marl-trains/results.jpeg)
</div>


We observe that we get the best performance from the IQL-based DDDQN model. This might be because of the simplicity of the environment, but in general this approach may not scale well with increasing number of agents and increasing complexity of the train network. Note that the Rainbow DQN has not yet reached convergence and do so with more number of episodes. Also, the MAAC policies do not give good results compared to the DQNs. This might might attributable to the number of episodes required for convergence. The original MAAC paper trains agents for over 50k episodes. Since we were limited by resources provided by Google Colab, we could not train our policies for a very long time. From the visualization below, we can also see that the MAAC policy hasn't converged completely as even the best performing test case isn't optimal.

<br/>
<br/>


<div align="center" markdown="1">
![Results_DDDQN]({{ site.baseurl }}/images/rl/marl-trains/viz_dddqn.gif){:height="50%" width="50%"}   
*DDDQN*
</div>

<div align="center" markdown="1">
![Results_Rainbow]({{ site.baseurl }}/images/rl/marl-trains/viz_rainbow.gif){:height="50%" width="50%"}   
*Rainbow DQN*
</div>

<div align="center" markdown="1">
![Results_MAAC]({{ site.baseurl }}/images/rl/marl-trains/viz_maac.gif){:height="50%" width="50%"}   
*MAAC*
</div>

