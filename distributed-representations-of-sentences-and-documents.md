## [Distributed Representations of Sentences and Documents](http://arxiv.org/abs/1405.4053)

TLDR; The authors present Paragraph Vector, which learns fixed-length, semantically meaningful vector representations for text of any length (sentences, paragraphs, documents, etc). The algorithm works by training a word vector model with an additional paragraph embedding vector as an input. This paragraph embedding is fixed for each paragraph, but varies across paragraphs. Similar to word2vec, PV comes in 2 flavors:

- A Distributed Memory Model (PV-DM) that predicts the next word based on the paragraph and preceding words
- A BoW model (PW-BoW) that predicts context words for a given paragraph

A notable property of PV is that during inference (when you see a new paragraph) it requires training of a new vector, which can be slow. The learned embeddings can used as the input to other models. In their experiments the authors train both variants and concatenate the results. The authors evaluate PV on Classification and Information Retrieval Tasks and achieve new state-of-the-art.


#### Data Sets / Results

Stanford Sentiment Treebank Polar error: 12.2%
Stanford Sentiment Treebank Fine-Grained error: 51.3%
IMDB Polar error: 7.42%
Query-based search result retrieval (internal) error: 3.82%


#### Key Points

- Authors use 400-dimensional PV and word embeddings. The window size is a hyperparameter chosen on the validation set, values from 5-12 seem to work well. In IMDB, window size resulted in error fluctuation of ~0.7%.
- PV-DM performs well on its own, but concatenating PV-DM and PV-BoW consistently leads to (small) improvements.
- When training the PV-DM model, use concatenation instead of averaging to combine words and paragraph vectors (this preserves ordering information)
- Hierarchical Softmax is used to deal with large vocabularies.
- For final classification, authors use LR or MLP, depending on the task (see below)
- IMDB Training (25k documents, 230 average length) takes 30min on 16 core machine, CPU I assume.


#### Notes / Question

- How did the authors choose the final classification model? Did they cross-validate this? The authors mention that NN performs better than LR for the IMDB data, but they don't show how large the gap is. Does PV maybe perform significantly worse with a simpler model?
- I wonder if we can train hierarchical representations of words, sentences, paragraphs, documents, keep the vectors of each one fixed at each layer, and predicting sequences using RNNs.
- I wonder how PV compares to an attention-based RNN autoencoder approach. When training PV you are in a way attending to specific parts of the paragraph to predict the missing parts.