## [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](http://arxiv.org/abs/1406.1078)

TLDR; The authors propose a novel encoder-decoder neural network architecture. The encoder RNN encodes a sequence into a fixed length vector representation and the decoder generates a new variable-length sequence based on this representation. The authors also introduce a new cell type (now called GRU) to be used with this network architecture. The model is evaluated on a statistical machine translation task where it is fed as an additional feature to a log-linear model. It leads to improved BLEU scores. The authors also find that the model learns syntactically and semantically meaningful representations of both words and phrases.

#### Key Points:

- New encoder-decoder architecture, seq2seq. Decoder conditioned on thought vector.
- Architecture can be used for both scoring or generation
- New hidden unit type, now called GRU. Simplified LSTM.
- Could replace whole pipeline with this architecture, but this paper doesn't
- 15k vocabulary (93% of dataset cover). 100d embeddings, 500 maxout units in final affine layer, batch size of 64, adagrad, 384M words, 3 days training time.
- Architecture is trained without frequency information so we expect it to capture linguistic information rather than statistical information.
- Visualizations of both words embeddings and thought vectors.


#### Questions/Notes

- Why not just use LSTM units?