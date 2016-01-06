## [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626)

TLDR; The authors evaluate the use for 9-layer deep CNNs on large-scale data sets for text classification, operating directly on one-hot encodings of characters. The architecture achieves competitive performance across datasets.

#### Key Points

- 9 Layers, 6 conv/ppol layers, 3 affine layers. 1024-dimensional input features for large model, 256-dimensional input features for small model.
- Authors optionally use English thesaurus for training data augmentation
- Fixed input length l: 1014 characters
- Simple n-gram models performs very well on these data sets and beats other models and the smaller data sets (<= 500k examples). CNN wins on the larger data sets (>1M examples)

#### Notes / Questions

- Comparing the CNN with input restricted to 1014 characters to models that operate on words seems unfair. Also, how long is the average document? Would've liked to see some dataset statistics. The fixed input length doesn't make a lot of sense to me.
- Contribution of this paper is that the architecture works without word knowledge and for any language, but at the same time the authors use a word-level English thesaurus to improve their performance? To be fair, the thesaurus doesn't seem to make a huge difference.
- The reason this architecture requires so much data is probably because it's very deep (How many parameters?). Did the authors experiment with fewer layers? Did they perform much worse?
- What about unsupervised pre-training? Can that reduce the amount of data required to achieve good performance. Currently this model doesn't seem very useful in practice as there are very few datasets of such size out there.