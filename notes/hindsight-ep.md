## [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)

TLDR; The authors present a novel way to deal with sparse rewards in Reinforcement Learning. The key idea (called HER, or Hindsight Experience Replay) is that when an agent does not achieve the desired goal during an episode, it still has learned to achieve *some other* goal, which it can learn about and generalize from. This is done by framing the RL problem in a multi-goal setting, and adding transitions with different goals (and rewards) to the experience buffer. When updating the policy, the additional goals with positive rewards lead to faster learning. Note that this requires an off-policy RL algorithm (such as Q-Learning).

#### Key Points

- Proper reward shaping can be difficult. Thus, it is important to develop algorithms that can learn from sparse binary reward signals.
- HER requires an off-policy Reinforcement Learning algorithm. For example, DQN, etc.
- Multi-Goal RL vs. "Standard RL"
    - Policy depends on the goal
    - Reward function depends on the goal
    - Goal is sampled at the start of each episode
- HER
    - Assume that the goal is some *state* that the agent can achieve
    - Needs a way to sample/generate a set of additional goals for an episode (hyperparameter)
        - For example: The goal is the last state visited in the episode
    - Store transitions with newly sampled goals (in addition to the original goal) in the replay buffer
    - Induces a form of implicit curriculum as goals become more difficult
        - Because the agent becomes better over time, the states it visits become "more difficult"
- Experiments: Robot Arm simulation
    - Clearly outperforms DDPG and DDPG with count-based exploration on binary rewards
    - Works whether we care about a single or multiple goals
    - Shows that shaped rewards may hinder exploration

#### Notes/Questions

- The idea that shaped rewards can hinder exploration is a good one, I really enjoyed that
- How does this approach relate to model-based learning. While there is no direct relationship you learn to generalize across goals - Learning about the environment can have a similar effect.
- Not really sold/convinced on the implicit curriculum learning. I see how it applies to some problems, but not to all. Just because an agent becomes better at achieving G, the states it visits are not necessarily more "difficult" to achieve. Maybe I'm missing something.
