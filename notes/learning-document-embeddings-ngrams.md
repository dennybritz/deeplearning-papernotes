## [Learning Document Embeddings by Predicting N-grams for Sentiment Classification of Long Movie Reviews](http://arxiv.org/abs/1512.08183)

TLDR; The authors present DV-ngram, a new method to learn document embeddings. DV-ngrams is a variation on Paragraph Vectors with a training objective of predicting words and n-grams solely based on the document vector, forcing the embedding to capture the semantics of the text. The authors evaluate their model on the IMDB data sets, beating both n-gram based and Deep Learning models.

#### Key Points

- When the word vectors are already sufficiently predictive of the next words, the standard PV embedding cannot learn anything useful.
- Training objective: Predict words and n-grams solely based on document vector. Negative Sampling to deal with large vocabulary. In practice, each n-gram is treated as a special token and appended to the document.
- Code will be at https://github.com/libofang/DV-ngram


#### Question/Notes

- The argument that PV may not work when the word vectors themselves are predictive enough makes intuitive sense. But what about applying word-level dropout? Wouldn't that also force the PV to learn the document semantics?
- It seems to be that predicting n-grams leads to a huge sparse vocabulary space. I wonder how this method scales, even with negative sampling. I am actually surprised this works well at all.
- The authors mention that they beat "other Deep Learning models, including PV, but neither their model nor PV are "deep learning". The networks are not deep ;)