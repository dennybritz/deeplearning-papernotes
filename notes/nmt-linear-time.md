## [Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099)

TLDR; The authors apply a [WaveNet](https://arxiv.org/abs/1609.03499)-like architecture to the task of Machine Translation. Encoder ("Source Network") and Decoder ("Target Network") are CNNs that use Dilated Convolutions and they are stacked on top of each other. The Target Network uses [Masked Convolutions](https://arxiv.org/abs/1606.05328) to ensure that it only relies on information from the past. Crucially, the time complexity of the network is `c(|S| + |T|)`, which is cheaper than that of the common seq2seq attention architecture (`|S|*|T|`). Through dilated convolutions the network has constant path lengths between [source input -> target output] and [target inputs -> target output] nodes. This allows for efficient propagation of gradients. The authors evlauate their model on Character-Level Language Modeling and Character-Level Machine Translation (WMT EN-DE) and achieve state-of-the-art on the former and a competitive BLEU score on the latter.


### Key Points

- Problems with current approaches
  - Runtime is not linear in the length of source/target sequence. E.g. seq2seq with attention is `O(|S|*|T|)`.
  - Some architectures compress the source into a fixed-length "though-vector", putting a memorization burden on the model.
  - RNNs are hard to parallelize
- ByteNet: Stacked network of encoder/decoder. In this work the authors use CNNs, but the network could be RNNs.
- ByteNet properties:
  - Resolution preserving: The representation of the source sequence is linear in the length of the source. Thus, a longer source sentence will have a bigger representation.
  - Runtime is linear in the length of source and target sequences: `O(|S| + |T|)`
  - Source network can be run in parallel, it's a CNN.
  - Distance (number of hops) between nodes in the network is short, allowing for efficient backprop.
- Architecture Details
  - Dynamic Unfolding: `representation_t(source)` is fed into time step `t` of the target network. Anything past the source sequence length is zero-padded. This is possible due to the resolution preserving property which ensures that the source representation is the same width as the source input.
  - Masked 1D Convolutions: The target network uses masked convolutions to prevent it from looking at the future during training.
  - Dilation: Dilated Convoltuions increase the receptive field exponentially in higher layers. This leads to short connection paths for efficient backprop.
  - Each layer is wrapped in a residual block, either with ReLUs or multiplicative units (depending on the task).
  - Sub-Batch Normalization: To preven the target network from conditioning on future tokens (similar to masked convolutions) a new variant of Batch Normalization is used.
- Recurrent ByteNets, i.e. ByteNets with RNNs instead of CNNs, are possible but are not evaluated.
- Architecture Comparison: Table 1 is great. It compares various enc-dec architectures across runtime, resolution preserving and path lengths properties.
- Character Prediction Experiments:
  - [Hutter Prize Version of Wikipedia](http://prize.hutter1.net/): ~90M characters
  - Sample a batch of 515 characters predict latter 200 from the first 315
  - New SOTA: 1.33 NLL (bits/chracter)
-  Character-Level Machine Translation
  - [WMT](http://www.statmt.org/wmt16/translation-task.html) EN-DE. Vocab size ~140
  - Bags of character n-grams as additional embeddings
  - Examples are bucketed according to length
  - BLEU: 18.9. Current state of the art is ~22.8 and standard attention enc-dec is 20.6


### Thoughts

- Overall I think this a very interesting contribution. The ideas here are pretty much identical to the [WaveNet](https://arxiv.org/abs/1609.03499) + [PixelCNN](https://arxiv.org/abs/1606.05328) papers. This paper doesn't have much detail on any of the techniques, no equations whatsoever. Implementing the ByteNet architecture based on the paper alone would be very challenging. The fact that there's no code release makes this worse.
- One of the main arguments is the linear runtime of the ByteNet model. I would've liked to see experiments that compare implementations in frameworks like Tensorflow to standard seq2seq implementations. What is the speedup in *practice*, and how does it scale with increased paralllelism? Theory is good and all, but I want to know how fast I can train with this.
- Through dynamic unfolding  target inputs as time t depend directly on the source representation at time t. This makes sense for language pairs that are well aligned (i.e. English/German), but it may hurt performance for pairs that are not aligned since the the path length would be longer. Standard attention seq2seq on the other hand always has a fixed path length of 1. Experiments on this would've been nice.
- I wonder how much difference the "bag of character n-grams" made in the MT experiments. Is this used by the other baselines?




