## [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473)

TLDR; The authors train an Generative Adversarial Network where the generator is an RNN producing discrete tokens. The discriminator is used to provide a reward for each generated sequence (episode) and to train the generator network via via Policy Gradients. The discriminator network is a CNN in the experiments. The authors evaluate their model on a synthetic language modeling task and 3 real world tasks: Chinese poem generation, speech generation and music generation. Seq-GAN outperforms competing approaches (MLE, Schedule Sampling, PG-BLEU) on the synthetic task and outperforms MLE on the real world task based on a BLEU evaluation metric.

#### Key Points

- Code: https://github.com/LantaoYu/SeqGAN
- RL Problem setup: State is already generated partial sequence. Action space is the space of possible tokens to output at the current step. Each episode is a fully generated sequence of fixed length T.
- Exposure Bias in the Maximum Likelihood approach: During decoding the model generates the next token based on a series previously generated tokens that it may never have seen during training leading to compounding errors.
- A discriminator can provide a reward when no task-specific reward (e.g. BLEU score) is available or when it is expensive to obtain such a reward (e.g. human eval).
- The reward is provided by the discriminator at the end of each episode, i.e. when the full sequence is generated. To provide feedback at intermediate steps the rest of the sequence is sampled via Monte Carlo search.
- Generator and discriminator are trained alternatively and strategy is defined by hyperparameters g-steps (# of Steps to train generator), d-steps (number of steps to train discriminator with newly generated data) and k (number of epochs to train discriminator with same set of generated data).
- Synthetic task: Randomly initialized LSTM as oracle for a language modeling task. 10,000 sequences of length 20.
- Hyperparameters g-steps, d-steps and k have a huge impact on training stability and final model performance. Bad settings lead to a model that is barely better than the MLE baseline.

#### My notes:

- Great paper overall. I also really like the synethtic task idea, I think it's a neat way to compare models.
- For the real-world tasks I would've liked to see a comparison to PG-BLEU as they do in the synthetic task. The authors evaluate on BLEU score so I wonder how much difference a direct optimization of the evaluation metric makes.
- It seems like SeqGAN outperforms MLE significantly only on the poem generation task, not the other tasks. What about the other baselines on the other tasks? What is it about the poem generation that makes SeqGAN perform so well?
