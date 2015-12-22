## [Semi-supervised Sequence Learning](http://arxiv.org/abs/1511.01432)

TLDR; The authors show that we can pre-train RNNs using unlabeled data by either reconstructing the original sequence (SA-LSTM), or predicting the next token as in a language model (LM-LSTM). We can then fine-tune the weights on a supervised task. Pre-trained RNNs are more stable, generalize better, and achieve state-of-the-art results on various text classification tasks. The authors show that unlabeled data can compensate for a lack of labeled data.

#### Data Sets

Error Rates for SA-LSTM, previous best results in parens.

- IMDB: 7.24% (7.42%)
- Rotten Tomatoes 16.7% (18.5%) (using additional unlabeled data)
- 20 Newsgroups: 15.6% (17.1%)
- DBPedia character-level: 1.19% (1.74%)

#### Key Takeaways

- SA-LSTM: Predict sequence based on final hidden state
- LM-LSTM: Language-Model pretraining
- LSTM, 1024-dimensional cell, 512-dimensional embedding, 512-dimensional hidden affine layer + 50% dropout, Truncated backprop 400 steps. Clipped cell outputs and gradients. Word and input embedding dropout tuned on dev set.
- Linear Gain: Inject gradient at each step and linearly increase weights of prediction objectives

#### Notes / Questions

- Not clear when/how linear gain yields improvements. On some data sets it significantly reduces performance, on other it significantly improves performance. Any explanations?
- Word dropout is used in the paper but not explained. I'm assuming it's replacing random words with `DROP` tokens?
- The authors mention a joint training model, but it's only evaluated on the IMDB data set. I'm assuming the authors didn't evaluate it further because it performed badly, but it would be nice to get an intuition for why it doesn't work, and show results for other data sets.
- All tasks are classification tasks. Does SA-LSTM also improve performance on seq2seq tasks?
- What is the training time? :) (I also wonder how the batching is done, are texts padded to the same length with mask?)
