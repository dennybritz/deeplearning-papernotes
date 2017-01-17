[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

TLDR; The authors test variations of the attention mechanism on Neural Machine Translation (NMT) tasks. The authors propose both "global" (attenting over all source words) and "local" (attending over a subset of source words) models. They evaluate their approach on WMT'14 and WMT'15 English <-> German translation and achieve state-of-the-art results.


### Key Points

- Softmax input is "attentional hidden state" which is computed as `W dot(c_t, h_t)` where `c_t` is the attention vector for the source sentence. How `c_t` is calculated is depends on the attention approach.
- Global attention score types `score(h_t, h_s)`:
  - dot: `dot(h_t, h_s)`
  - general: `h_t^T * W * h_s`
  - concat `v_t * tanh(W_a * dot(h_t, h_s)` (Bahdanau)
- Local attention idea: Decoder computes aligned position `p_t` and attends over source hidden state in `[p_t - D, p_t + D]` where `D` is a hyperparameter.
- Training details
  - 50k vocab
  - 4 layers, 1000-dimensional embeddings, LSTM cell, unidirectional encoder, gradient norm clipping at 5; SGD with decay schedule; dropout 0.2.
  - UNK replacement (gives +1.9 BLEU)
- For global attention, dot scores (the simplest choice) seems to peform best.

