### [Generating Sentences from a Continuous Space](http://arxiv.org/abs/1511.06349)

TLDR; The authors present an RNN-based variational autoencoder that can learn a  latent sentence representation while learning to decode. A linear layer that predicts the parameter of a Gaussian distribution is inserted between encoder and decoder. The loss is a combination of the reconstruction objective and the KL divergence with the prior (Gaussian) - similar to the "standard" VAE does. The authors evaluate the model on Language Modeling and Impution (Inserting Missing Words) tasks and also present a qualitative analysis of the latent space.

#### Key Points

- Training is tricky. Vanilla training results in the decoder ignoring the encoder and the KL error term becoming zero.
- Training Trick 1: KL Cost Annealing. During training, increase weight on the KL term of the cost to anneal from vanilla to VAE.
- Training Trick 2: Word dropout using a word keep rate hyperparameter. This forces the decoder to rely more on the global representation.
- Results on Language Modeling: Standard model (without cost annealing and word dropout) trails Vanilla RNNLM model, but not by much. KL cost term goes to zero in this setting. In an inputless decoder setting (word keep prob = 0) the VAE outperforms the RNNLM (obviously)
- Results on Imputing Missing Words: Benchmarked using an adversarial error classifier. VAE significantly outperforms RNNLM. However, the comparison is somewhat unfair since the RNNML has nothing to condition on and relies on unigram distribution for the first token.
- Qualitative: Can use higher word dropout to get more diverse sentences
- Qualitative: Can walk the latent space and get grammatical and meaningful sentences.
