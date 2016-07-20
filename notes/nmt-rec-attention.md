## [Neural Machine Translation with Recurrent Attention Modeling](https://arxiv.org/abs/1607.05108)

TLDR; The standard attention model does not take into account the "history" of attention activations, even though this should be a good predictor of what to attend to next. The authors augment a seq2seq network with a dynamic memory that, for each input, keep track of an attention matrix over time. The model is evaluated on English-German and Englih-Chinese NMT tasks and beats competing models.

#### Notes

- How expensive is this, and how much more difficult are these networks to train?
- Sequentiallly attending to neighboring words makes sense for some language pairs, but for others it doesn't. This method seems rather restricted because it only takes into account a window of k time steps.