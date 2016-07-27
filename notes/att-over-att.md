## [Attention-over-Attention Neural Networks for Reading Comprehension](https://arxiv.org/abs/1607.04423)

TLDR; The authors present a novel Attention-over-Attention (AoA) model for Machine Comprehension. Given a document and cloze-style question, the model predicts a single-word answer. The model,

1. Embeds both context and query using a bidirectional GRU
2. Computes a pairwise matching matrix between document and query words
3. Computes query-to-document attention values
4. Computes document-to-que attention averages for each query word
5. Multiplies the two attention vectors to get final attention scores for words in the document
6. Maps attention results back into the vocabulary space

The authors evaluate the model on the CNN News and CBTest Question Answering datasets, obtaining state-of-the-art results and beating other models including EpiReader, ASReader, etc.


#### Notes:

- Very good model visualization in the paper
- I like that this model is much simpler than EpiReader while also performing better