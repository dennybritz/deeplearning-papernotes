## [Larger-Context Language Modeling](http://arxiv.org/abs/1511.03729)

TLDR; The authors propose new ways to incorporate context (previous sentences) into a Recurrent Language Model (RLM). They propose 3 ways to model the context, and 2 ways to incorporate the context into the predictions for the current sentence. Context can be modeled with BoW, Sequence BoW (BoW for each sentence), and Sequence BoW with attention. Context can be incorporated using "early fusion", which gives the context as an input to the RNN, or "late fusion", which modifies the LSTM to directly incorporate the context. The authors evaluate their architecture on IMDB, BBC and Penn TreeBank corpora, and show that most approaches perform well (reducing perplexity), with Sequence BoW with attention + late fusion outperforming all others.

#### Key Points:

- Context as BoW: Compress N previous sentences into a single BoW vector
- Context as Sequential Bow: Compress each of the N previous sentences into a BoW vector and use an LSTM to "embed" them. Alternatively, use an attention mechanism.
- Early Fusion: Give the context vector as an input to the LSTM, together with the current word.
- Late Fusion: Add another gate to the LSTM that incorporates the context vector. Helps to combat vanishing gradients.
- Interestingly the Sequence BoW without attention performs very poorly. The reason here seems to be the same as for seq2seq, it's hard to compress the sentence vectors into a single fixed-length representation using an LSTM.
- LSTM models trained with 1000 units, Adadelta. Only sentences up to 50 words are considered.
- Noun phrases seem to benefit the most from the context, which makes intuitive sense.


#### Notes/Questions:

- A problem with current Language Models is that they are corpus-specific. A model trained on one corpus doesn't do well on another corpus because all sentences are treated as being independent. However, if we can correctly incorporate context we may be able to train a general-purpose LM that does well across various corpora. So I think this is important work.
- I am surprised that the authors did not try using a sentence embedding (skip-thought, paragraph-vector) to construct their context vectors. That seems like an obvious choice over using BoW.
- The argument for why the Sequence BoW without attention model performs poorly isn't convincing. In the seq2seq work the argument for attention was based on the length of the sequence. However, here the sequence is very short, so the LSTM should be able to capture all the dependencies. The performance may be poor due to the BoW representation, or due too little training data.
- Would've been nice to visualize what the attention mechanism is modeling.
- I'm not sure if I agree with the authors that relying explicit sentence boundaries is an advantage, I see it as a limiting factor.