## [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843)

TLDR; The authors combine a standard LSTM softmax with [Pointer Networks](https://arxiv.org/abs/1506.03134) in a mixture model called Pointer-Sentinel LSTM (PS-LSTM). The pointer networks helps with rare words and long-term dependencies but is unable to refer to words that are not in the input. The oppoosite is the case for the standard softmax. By combining the two approaches we get the best of both worlds. The probability of an output words is defined as a mixture of the pointer and softmax model and the mixture coefficient is calculated as part of the pointer attention. The authors evaluate their architecture on the PTB Language Modeling dataset where they achieve state of the art. They also present a novel WikiText dataset that is larger and more realistic then PTB.


### Key Points:

- Standard RNNs with softmax struggle with rare and unseen words, even when adding attention.
- Use a window of the most recent`L` words to match against.
- Probability of output with gating: `p(y|x) = g * p_vocab(y|x) + (1 - g) * p_ptr(y|x)`.
- The gate `g` is calcualted as an extra element in the attention module. Probabilities for the pointer network are then normalized accordingly.
- Integrating the gating funciton computation into the pointer network is crucial: It needs to have access to the pointer network state, not just the RNN state (which can't hold long-term info)
- WikiText-2 dataset: 2M train tokens, 217k validation tokens, 245k test tokens. 33k vocab, 2.6% OOV. 2x larger than PTB.
- WikiText-1-3 dataset: 103M train tokens, 217k validation tokens, 245k test tokens. 267k vocab, 2.4% OOV. 100x larger than PTB.
- Pointer Sentiment Model leads to stronger improvements for rare words - that makes intuitive sense.

