## [Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models](http://arxiv.org/abs/1604.00788)

TLDR; The authors train a word-level NMT where UNK tokens in both source and target sentence are replaced by character-level RNNs that produce word representations. The authors can thus train a fast word-based system that still generalized that doesn't produce unknown words. The best system achieves a new state of the art BLEU score of 19.9 in WMT'15 English to Czech translation.


#### Key Points

- Source Sentence: Final hidden state of character-RNN is used as word representation.
- Source Sentence: Character RNNs always initialized with 0 state to allow efficient pre-training 
- Target: Produce word-level sentence including UNK first and then run the char-RNNs
- Target: Two ways to initialize char-RNN: With same hidden state as word-RNN (same-path), or with its own representation (separate-path)
- Authors find that attention mechanism is critical for pure character-based NMT models


#### Notes

- Given that the authors demonstrate the potential of character-based models, is the hybrid approach the right direction? If we had more compute power, would pure character-based models win?