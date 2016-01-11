## [Multi-Way, Multilingual Neural Machine Translation with a Shared Attention Mechanism](http://arxiv.org/abs/1601.01073)

TLDR; The authors train a *single* Neural Machine Translation model that can translate between N*M language pairs, with a parameter spaces that grows linearly with the number of languages. The model uses a single attention mechanism shared across encoders/decoders. The authors demonstrate the the model performs particularly well for resource-constrained languages, outperforming single-pair models trained on the same data.

#### Key Points

- Attention mechanism: Both encoder and decoder output attention-specific vectors, which are then combined. Thus, adding a new source/target language does not result in a quadratic explosion of parameters.
- Bidirectional RNN, 620-dimensional embeddings, GRU with 1k units, 1k affine layer tanh. Adam, minibatch 60 examples. Only use sentence up to length 50.
- Model clearly outperforms single-pair models when parallel corpora are constrained to small size. Not so much for large corpora.
- The single model doesn't fit on a GPU.
- Can in theory be used to translate between pairs that didn't have a bilingual training corpus, but the authors don't evaluate this in the paper. 
- Main difference to "Multi-task Sequence to Sequence Learning": Uses attention mechanism


#### Notes / Questions

- I don't see anything that would force the encoders to map sequences of different languages into the same representation (as the authors briefly mentioned). Perhaps it just encodes language-specific information that the decoders can use to decide which source language it was? 