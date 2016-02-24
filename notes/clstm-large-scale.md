## [Contextual LSTM (CLSTM) models for Large scale NLP tasks](http://arxiv.org/abs/1602.06291)

TLDR; The authors propose a Contextual LSTM (CLSTM) model that appends a context vector to input words when making predictions. The authors evaluate the model and Language Modeling, next sentence selection and next topic prediction tasks, beating standard LSTM baselines.


#### Key Points

- The topic vector comes from an internal classifier system and is supervised data. Topics can also be estimated using unsupervised techniques.
- Topic can be calculated either based on the previous words of the current sentence (SentSegTopic), all words of the previous sentence (PrevSegTopic), and current paragraph (ParaSegTopic). Best CLSTM uses all of them. 
- English Wikipedia Dataset: 1400M words train, 177M validation, 178M words test. 129k vocab.
- When current segment topic is present, the topic of the previous sentence doesn't matter.
- Authors couldn't compare to other models that incorporate topics because they don't scale to large-scale datasets.
- LSTMs are a long chain and authors don't reset the hidden state between sentence boundaries. So, a sentence has implicit access to the prev. sentence information, but explicitly modeling the topic still makes a difference.

#### Notes/Thoughts

- Increasing number of hidden units seems to have a *much* larger impact on performance than increasing model perplexity. The simple word-based LSTM model with more hidden units significantly outperforms the complex CLSTM model. This makes me question the practical usefulness of this model.
- IMO the comparisons are somewhat unfair because by using an external classifier to obtain topic labels you are bringing in external data that the baseline models didn't have access to.
- What about using other unsupervised sentence embeddings as context vectors, e.g. seq2seq autoencoders or PV?
- If the LSTM was perfect in modeling long-range dependencies then we wouldn't need to feed extra topic vectors. What about residual connections?