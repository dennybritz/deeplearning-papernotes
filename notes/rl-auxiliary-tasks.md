## [Reinforcement Learning with Unsupervised Auxiliary Tasks](http://openreview.net/forum?id=SJ6yPD5xg)

TLDR; The authors augment the A3C (Asynchronous Actor Critic) algorithm with auxiliary tasks. These tasks share some of the network parameters but value functions for them are learned off-policy using n-step Q-Learning. The auxiliary tasks only used to learn a better representation and don't directly influence the main policy control. The technique, called UNREAL (Unsupervised Reinforcement and Auxiliary Learning), outperforms A3C on both the Atari and Labyrinth domains in terms of performance and training efficiency.


#### Key Points

- Environments contain a wide variety of possible training signals, not just cumulative reward
- Base A3C agent uses CNN + RNN
- Auxiliary Control and Prediction tasks share the convolutional and LSTM network for the "base agent". This forces the agent to balance improvement and base and aux. tasks.
- Auxiliary Tasks
  - Use off-policy RL algorithms (e.g. n-step Q-Learning) so that the same stream of experience from the base agent can be used for maximizing all tasks. Experience is sampled from a replay buffer.
  - Pixel Changes (Auxiliary Control): Learn a policy for maximally changing the pixels in a grid of cells overlaid over the images
  - Network Features (Auxiliary Control): Learn a policy for maximally activating units in a specific hidden layer
  - Reward Prediction (Auxiliary Reward): Predict the next reward given some historical context. Crucially, because rewards tend to be sparse, histories are sampled in a skewed manner from the replay buffer so that P(r!=0) = 0.5. Convolutional features are shared with the base agent.
  - Value Function Replay: Value function regression for the base agent with varying window for n-step returns.
- UNREAL
  - Base agent is optimized on-policy (A3C) and aux. tasks are optimized off-policy.
- Experiments
  - Agent is trained with 20-step returns and aux. tasks are performed every 20 steps.
  - Replay buffer stores the most recent 2k observations, actions and rewards
  - UNREAL tends to be more robust to hyperparameter settings than A3C
  - Labyrinth
    - 38% -> 83% human-normalized score. Each aux. tasks independently adds to the performance.
    - Significantly faster learning, 11x across all levels
    - Compared to input reconstruction technique: Input reconstruction hurts final performance b/c it puts too much focus on reconstructing relevant parts.
  - Atari
    - Not all experiments are completed yet, but UNREAL already surpasses state of the art agents and is more robust.

#### Thoughts

- I want an algorithm box please :)