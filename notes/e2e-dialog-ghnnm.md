## [Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](http://arxiv.org/abs/1507.04808)

TLDR; The authors train a Hierarchical Recurrent Encoder-Decoder (HRED) network for dialog generation. The "lower" level encodes a sequence of words into a though vector, and the higher-level encoder uses these thought vectors to build a representation of the context. The authors evaluate their model on the *MoviesTriples* dataset using perplexity measures and achieve results better than plain RNNs and the DCGM model. Pre-training with a large Question-Answer corpus significantly reduces perplexity.

#### Key Points

- Three RNNs: Utterance encoder, context encoder, and decoder. GRU hidden units, ~300d hidden state spaces.
- 10k vocabulary. Preprocessing: Remove entities and numbers using NLTK
- The context in the experiments is only a single utterance
- MovieTriples is a small dataset, about 200k training triples. Pretraining corpus has 5M Q-A pairs, 90M tokens.
- Perplexity is used as an evaluation metric. Not perfect, but reasonable.
- Pre-training has a much more significant impact than the choice of the model architecture. It reduces perplexity ~10 points, while model architecture makes a tiny difference (~1 point).
- Authors suggest exploring architectures that separate semantic from syntactic structure
- Realization: Most good predictions are generic. Evaluation metrics like BLEU will favor pronouns and punctuation marks that dominate during training and are therefore bad metrics.


#### Notes/Questions

- Does using a larger dataset eliminate the need for pre-training?
- What about the more challenging task for longer contexts?