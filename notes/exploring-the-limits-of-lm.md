## [Exploring the Limits of Language Modeling](http://arxiv.org/abs/1602.02410)

TLDR; The authors train large-scale language modeling LSTMs on the 1B word dataset to achieve new state of the art results for single models (51.3 -> 30 Perplexity) and ensemble models (41 -> 24.2 Perplexity). The authors evaluate how various architecture choices impact the model performance: Importance Sampling Loss, NCE Loss, Character-Level CNN inputs, Dropout, character-level CNN output, character-level LSTM Output.

#### Key Points

- 800k vocab, 1B words training data
- Using a CNN on characters instead of a traditional softmax significantly reduces number of parameters, but lacks the ability to differentiate between similar-looking words with very different meanings. Solution: Add correction factor
- Dropout on non-recurrent connections significantly improves results
- Character-level LSTM for prediction performs significantly worse than softmax or CNN softmax
- Sentences are not pre-processed, fed in 128-sized batches without resetting any LSTM state in between examples. Max word length for character-level input: 50
- Training: Adagrad and learning rate of 0.2. Gradient norm clipping 1.0. RNN unrolled for 20 steps. Small LSTM beats state of the art after just 2 hours training, largest and best model trained for 3 weeks on 32 K40 GPUs.
- NC vs. Importance Sampling: IC is sufficient
- Using character-level CNN word embeddings instead of a traditional matrix is sufficient and performs better

#### Notes/Questions

- Exact hyperparameters in table 1 are not clear to me.