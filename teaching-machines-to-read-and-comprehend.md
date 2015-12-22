## [Teaching Machines to Read and Comprehend](http://arxiv.org/abs/1506.03340)

TLDR; The authors generate a large dataset (~1M examples) for question answering by using cloze deletion on summaries of crawled CNN and Daily Mail articles. They evaluate 2 baselines, 2 symbolic models (frame semantic, word distance), and 4 neural models (Deep LSTM, Uniform Reader, Attentive Reader, Impatient Reader) on the dataset. The neural models, particularly those with attenton, beat the syntactic models.

- Deep LSTM: 2-layer bidirectional LSTM without attention mechanism
- Attentive reader: 1-layer bidirectional LSTM with attention mechanism for the whole query
- Impatient Reader: 1-layer bidirectional LSTM with attention mechanism for each token in the query (can be interpreted as being able to re-read the document at each token)
- Uniform Reader: Uniform attention to all document tokens

In their experiments, the authors randomize document entities to avoid letting the models rely on world knowledge or co-occurence statistics, and intead purely testing document comprehension. This is done by replacing entities with consistent ids *within* a document, but using different ids across documents.

#### Data and model performance

All numbers are accuracies on two datasets (CNN, Daily Mail)

- Maximum Frequency Entity Baseline: 33.2 / 25.5
- Exclusive Frequence Entity Baseline: 39.3 / 32.8
- Frame-semantic model: 40.2 / 35.5
- Word distance model: 50.9 / 55.5
- Deep LSTM Reader: 57.0 / 62.2
- Uniform Reader: 39.4 / 34.4
- Attentive Reader: 63.0 / 69.0
- Impatient Reader: 63.8 / 68.0


#### Key Takeaways

- The input to the RNN is defined as `QUERY <DELIMITER> DOCUMENT `, which is then embdedded with or without attention and run through `softmax(W*x)` .
- Some sequences are very long, up to 2000 tokens, and the average length was 763 tokens. All LSTM models seem to be able to deal with this, but the attention models show significantly higher accuracy.
- Very nice attention visualizations and negative examples analysis that show the attention-based models focusing on the relevant parts of the document to answer the questions.


#### Notes / Questions

- How does document length affect the Deep LSTM reader? The appendix shows an anlysis for attention models, but not for the Deep LSTM. A goal of the paper was to show that attention mechanisms are well suited for long documents because the fixed vector encoding is a bottleneck. The reuslts here aren't clear.
- Are the gradient truncated? I can't imagine the network is unrolled for 2000 steps. The training parameters details don't mention this.
- The mathematical notation in this paper needs some love. The concepts are relatively simple, but the formulas are hard to parse.
- What if you limited the output vocabulary to words appearing in the query document?
- Can you apply the same "attention-based embedding" mechanism to text classification?
