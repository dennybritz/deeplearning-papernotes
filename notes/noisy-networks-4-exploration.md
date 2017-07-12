## [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)

TLDR; The authors introduce NoisyNet, a deep reinforcement learning agent with parametric noise(learned with gradient descent) added to its weights, and show that the induced stochasticity of the agent’s policy can be used to aid efficient exploration. The authors combined this new agent with 2 well known DRL algorithms(DQN and A3C) achieving significant performance improvements across many Atari games and in some cases advancing the agent
from sub to super-human performance.

### Key Points
- Random perturbations of the agent’s policy, such as ϵ-greedy or entropy regularisation are unlikely to lead to the large-scale behavioural patterns needed for efficient exploration in many environments;
- Exploration trough augmentation of the environment’s reward signal with an additional intrinsic motivation term, such as learning progress, compression progress etc... separate the mechanism of generalisation from exploration;
- Evolutionary or black box algorithms are usually quite good at exploring the policy space but aren't data efficient and require a simulator and prolonged interactions with the environment;
- The perturbations of the network weights are used to drive exploration: a single change to the weight vector can induce a consistent, and potentially very complex, state-dependent change in policy over multiple timesteps;
- The perturbations are sampled from a noise distribution;
- The variance of the perturbation is a parameter that can be considered as the energy
of the injected noise;
- __NoisyNet exploration strategy can be applied to any deep RL algorithms that can be trained with gradient descent__.
