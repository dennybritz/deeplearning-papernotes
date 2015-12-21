## [Multilingual Language Processing From Bytes](http://arxiv.org/abs/1512.00103)

TLDR; The authors train a deep seq-2-seq LSTM directly on byte-level input of several langauges (shuffling the examples of all languages) and apply it to NER and POS tasks, achieving state-of-the-art or close to that. The model outputs spans of the form `[START_POSITION, LENGTH, LABEL]`, where each span element is a separate token prediction. A single model works well for all languages and learns shared high-level representations. The authors also present a novel way to dropout input tokens (bytes in their case), by randomly replacing them with a `DROP` symbol.

#### Data and model performance

Data:

- POS Tagging: 13 languages, 2.87M tokens, 25.3M training segments
- NER: 4 languags, 0.88M tokens, 6M training segments

Results:

- POS CRF Accuracy (average across languages): 95.41
- POS BTS Accuracy (average across languages): 95.85
- NER BTS en/de/es/nl F1: 86.50/76.22/82.95/82.84
- (See paper for NER comparsion models)

#### Key Takeaways

- Surprising to me that the span generations works so well without imposing independence assumptions on it. It's state the LSTM has to keep in memory.
- 0.2-0.3 Dropout, 320-dimensional embeddings, 320 units LSTM, 4 layers seems to perform well. The resulting model is surprisingly compact (~1M parameters) due to the small vocabulary size of 256 bytes. Changing input sequence order didn't have much of an effect. Dropout and Byte Dropout significantly (74 -> 78 -> 82) improved F1 for NER.
- To limit sequence length the authors split the text into k=60 sized segment, with 50% overlap to avoid splitting mid-span.
- Byte Dropout can be seen as "blurring text". I believe I've seen the same technique applied to words before and labeled word dropout. 
- Training examples for all languages are shuffled together. The biggest improvements in scores are seen observed for low-resource languages.
- Not clear how to tune recall of the model since non-spans are simply not annotated.

#### Notes / Questions

- I wonder if the fixed-vector embedding of the input sequence is a bottleneck since the decoder LSTM has to carry information not only about the input sequence, but also about the structure that has been produced so far. I wonder if the authors have experimented with varying `k`, or using attention mechanisms to deal with long sequences (I've seen papers dealing with sequences of 2000 tokens?). 60 seems quite short to me. Of course, output vocabulary size is also a concern with longer sequences.
- What about LSTM initialization? When feeding spans coming from the same document, is the state kept around or re-initialized? I strongly suspect it's kept since 60 bytes probably don't contain enough information for proper labeling, but didn't see an explicit reference.
- Why not a bidirectional LSTM? Seems to be the standard in most other papers.
- How exactly are multiple languages encoded in the LSTM memories? I *kind of* understand the reasoning behind this, but it's unclear what these "high-level" representations are. Experiments that demonstrate what the LSTM cells represent would be valuable.
- Is there a way to easily re-train the model for a new language?

