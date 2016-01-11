## [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

TLDR; The authors evaluate the impact of hyperparameters (embeddings, filter region size, number of feature maps, activation function, pooling, dropout and l2 norm constraint) on Kim's (2014) CNN for sentence classification. The authors present empirical findings with variance nunbers based on a large number of experiments on 7 classification data sets, and give practical recommendation for architecture decisions.


#### Key Points

- Recommended Baseline configuration: word2vec, (3,4,5) filter regions, 100 feature maps per region size, ReLU activation, 1-max-pooling, 0.5 dropout, l2 norm constraint on weight vector of 3.
- One-hot vectors perform worse than pre-trained embeddings. word2vec outperforms GloVe most of the time.
- Filter region size is dependent on data set in the range of 2-25. Recommended to do a line search over single region size and then combine multiple sizes.
- Increasing the number of feature maps per filter region to more than 600 doesn't seem to help much.
- ReLU almost always best activation function
- Max-pooling almost always best pooling strategy
- Dropout from 0.1 to 0.5 helps, l2 norm constraint not much


#### Notes/Questions

- All datasets analyzed in this paper are rather similar. They have similar average and max sentence length, and even the number of examples is of roughly the same magnitude. It would be interesting to see how the result change with very different datasets, such as long documents, or very large numbers of training examples.