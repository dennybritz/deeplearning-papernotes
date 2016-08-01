## [An Actor-Critic Algorithm for Sequence Prediction](http://arxiv.org/abs/1607.07086)

TLDR; The authors propose to use the Actor Critic framework from Reinforcement Learning for Sequence prediction. They train an actor (policy) network to generate a sequence together with a critic (value) network that estimates the q-value function. Crucially, the actor network does not see the ground-truth output, but the critic does. This is different from LL (log likelihood) where errors are likely to cascade. The authors evaluate their framework on an artificial spelling correction and a real-world German-English Machine Translation tasks, beating baselines and competing approaches in both cases.

#### Key Points

- In LL training, the model is conditioned on its own guesses during search, leading to error compounding.
- The critic is allowed to see the ground truth, but the actor isn't
- The reward is a task-specific score, e.g. BLEU
- Use bidirectional RNN for both actor and critic. Actor uses a soft attention mechanism.
- The reward is partially receives at each intermediate step, not just at the end
- Framework is analogous to TD-Learning in RL
- Trick: Use additional target network to compute q_t (see Deep-Q paper) for stability
- Trick: Use delayed actor (as in Deep Q paper) for stability
- Trick: Put constraint on critic to deal with large action spaces (is this analogous to advantage functions?)
- Pre-train actor and critic to encourage exploration of the right space
- Task 1: Correct corrupt character sequence. AC outperforms LL training. Longer sequences lead to stronger lift.
- Task 2: GER-ENG Machine Translation: Beats LL and Reinforce models
- Qualitatively, critic assigns high values to words that make sense
- BLUE scores during training are lower than those of LL model - Why? Strong regularization? Can't overfit the training data.

#### Notes

- Why does the sequence length for spelling prediction only go up to 30? This seems very short to me and something that an LSTM should be able to handle quite easily. Would've like to see much longer sequences.

