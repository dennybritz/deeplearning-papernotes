## [Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732)

TLDR; The authors combine the DQN algorithm with human demonstration data, called DQfD. They pre-train the agent with a combination of four losses, supervised and td-losses, from human demonstration data. Once the agent starts interacting with the environment, both the human demonstration data and the transitions taken by the agent are kept in the same replay buffer. Transitions are sampled with prioritzed experience replay. The algorithms learns much faster than most other DQN variants, does not need large amounts of demonstration data, and achieves new high scores on some of the games.


#### Key Points

- Most real-world problems don't have good (or any) simulators. But we often have some sample plays from human controllers.
- Four losses are used when learning from transitions:
    - 1-step Q-Learning loss
    - n-step Q-Learning loss
    - supervised large-margin classification loss (this loss is only added for the demonstration transitions)
    - L2 regularization loss
- Difference from Imitation Learning
    - Imitation Learning uses a pure supervised loss. It can never exceed the performance of the human demonstrator
    - DQfD continues learning on-line and can learn to become better than the human policy
- Replay Buffer
    - Both agent and demonstration data is mixed in the same buffer
    - Demonstration data is fixed in the buffer, never replaced by agent transitions. Extra probability is added to the demonstration data to encourage sampling it more often.
- Human demonstration data ranges for 5k to 75k transitions depending on the game
- Experiments show that the combination of the four losses is crucial, taking out n-step returns or supervised loss significantly degrades agent performance.
- Very good performance especially on games that require longer-term planning (human demonstrations very useful here)

#### Thoughts

Very nice paper and good results. This is a relatively simple technique to bootstrap agents and speed up the learning process. I'm not really sure if the experimental results are fair with the hyperparameter tuning and extra data, and also no comparison to "better" techniques like Rainbow, A3C, Reactor, etc. The authors give good arguments for why they don't compare, I still would've liked to see the difference in scores.

