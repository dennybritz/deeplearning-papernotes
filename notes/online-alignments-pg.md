## [Learning Online Alignments with Continuous Rewards Policy Gradient](https://arxiv.org/abs/1608.01281)

TLDR; The authors use policy gradients on an RNN to train a "hard" attention mechanism that decides whether to output something at the current timestep or not. Their algorithm is online, which means it does not need to see the complete sequence before making a prediction, as is the case with soft attention. The authors evaluate their model on small- and medium-scale speech recognition tasks, where they achieve performance comparable to standard sequential models.

#### Notes:

- Entropy regularization and baselines were critical to make the model learn
- Neat trick: Increase dropout as training progresses
- Grid LSTMs outperformed standard LSTMs