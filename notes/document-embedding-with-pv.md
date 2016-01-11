## [Document Embedding with Paragraph Vectors](http://arxiv.org/abs/1507.07998)

TLDR; The authors evaluate Paragraph Vectors on large Wikipedia and arXiv document retrieval tasks and compare the results to LDA, BoW and word vector averaging models. Paragraph Vectors either outperform or match the performance of other models. The authors show how the embedding dimensionality affects the results. Furthermore, the authors find that one can perform arithemetic operations on paragraph vectors and obtain meaningful results and present qualitative analyses in the form of visualizations and document examples.


#### Data Sets

Accuracy is evaluated by constructing triples, where a pair of items are close to each other and the third one is unrelated (or less related). Cosine similarity is used to evaluate semantic closeness.

Wikipedia (hand-built) PV: 93%
Wikipedia (hand-built) LDA: 82%
Wikipedia (distantly supervised) PV: 78.8%
Wikipedia (distantly supervised) LDA: 67.7%
arXiv PV: 85%
arXiv LDA: 85%


#### Key Points

- Jointly training PV and word vectors seems to improve performance.
- Used Hierarchical Softmax as Huffman tree for large vocabulary
- The use only the PV-BoW model, because it's more efficient.

#### Questions/Notes

- Why the performance discrepancy between the arXiv and Wikipedia tasks? BoW performs surprisingly well on Wikipedia, but not arXiv. LDA is the opposite. 