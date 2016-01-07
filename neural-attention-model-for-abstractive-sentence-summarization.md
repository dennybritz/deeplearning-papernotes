## [A Neural Attention Model for Abstractive Sentence Summarization](http://arxiv.org/abs/1509.00685)

TLDR; The authors apply a neural seq2seq model to sentence summarization. The model uses an attention mechanism (soft alignment).


#### Key Points

- Summaries generated on the sentence level, not paragraph level
- Summaries have fixed length output
- Beam search decoder
- Extractive tuning for scoring function to encourage the model to take words from the input sequence
- Training data: Headline + first sentence pair.
