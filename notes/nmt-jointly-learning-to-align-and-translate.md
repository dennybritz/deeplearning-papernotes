## [Neural Machine Translation by Jointly Learning to Align and Translate](http://arxiv.org/abs/1409.0473)

TLDR; The authors propose a novel "attention" mechanism that they evaluate on a Machine Translation task, achieving new state of the art (and large improvements in dealing with long sentences). Standard seq2seq models typically try to encode the input sequence into a fixed length vector (the last hidden state) based on which the decoder generates the output sequence. However, it is unreasonable to assume the all necessary information can be encoded in this one vector. Thus, the authors let the decoder depend on a attention vector, which based on the weighted sum (expectation) of the input hidden states. The attention weights are learned jointly, as part of the network architecture.


#### Data Sets and model performance

Bidirectional GRU, 1000 hidden units. Multilayer maxout to compute output probabilities in decoder.

WMT '14 BLEU: 36.15


#### Key Takeaways

- Attention mechanism is a weighted sum of the hidden states computed by the encoder. The weights come from a softmax-normalized attention function (a perceptron in this paper), which are learned during training.
- Attention can be expensive, because it must be evaluated for each encoder-decoder output pair, resulting in a len(x) * len(y) matrix.
- The attention mechanism improves performance across the board, but has a particularly large affect on long sentences, confirming the hyptohesis that the fixed vector encoding is a bottleneck.
- The authors use a bidirectional-GRU, concatenating both hidden states into a final state at each time step.
- It is easy to visualize the attention matrix (for a single input-ouput sequence pair). The authors show that in the case of English to French translations the matrix has large values on the diagonal, showing the these two languages are well aligned in terms of word order.


#### Question/Notes

- The attention mechanism seems limited in that it computes a simple weighted average. What about more complex attention functions that allow input states to interact?