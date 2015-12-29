## [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](http://arxiv.org/abs/1409.1259)

TLDR; The authors empirically evaluate seq2seq Neural Machine Translation systems. They find that performance degrades significantly as sentences get longer, and as the number of unknown words in the source sentence increases. Thus, they propose that more investigation into how to deal with large vocabularies and long-range dependencies is needed. The authors also present a new gated recursive convolutional network (grConv) architecture, which consists of a binary tree using GRU units. While this network architecture does not perform as well as the RNN encoder, it seems to be learning grammatical properties represented in the gate activations in an unsupervised fashion.

#### Key Points

- GrConv: Neuron computed as combination between left and right neuron in previous layer, gated with the activations of those neurons. 3 gates: Left, right, reset.
- In experiments, encoder varies between RNN and grConv. Decoder is always RNN.
- Model size is only 500MB. 30k vocabulary. Only trained on sentences <= 30 tokens. Networks not trained to convergence. 
- Beam search with scores normalized by sequence length to choose translations.
- Hypothesis is that fixed vector representation is a bottleneck, or that decoder is not powerful enough.


#### Notes/Questions

- THe network is only trained on sequences <= 30 tokens. Can we really expect it to perform well on long sequences? Long sequences may inherently have grammatical structures that cannot be observed in short sequences.
- There's a mistake in the new activation formula, wrong time superscript, should be (t-1).