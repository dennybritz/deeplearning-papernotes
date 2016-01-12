## [Target-Dependent Sentiment Classification with Long Short Term Memory](http://arxiv.org/abs/1512.01100)

TLDR; The authors propose two LSTM-based models for target-dependent sentiment classification. TD-LSTM uses two LSTM networks running towards to target word from left and right respectively, making a prediction at the target time step. TC-LSTM is the same, but additionally incorporates the an averaged target word vector as an input at each time step. The authors evaluate their models with pre-trained word embeddings on a Twitter sentiment classification dataset, achieving state of the art.

#### Key Points

- TD-LSTM: Two LSTM networks, running from left to right towards the target. The final states of both networks are concatenated and the prediction is made at the target word.
- TC-LSTM: Same architecture as TD-LSTM, but also incorporates the word vector as an input at each time step. The word vector is the average of the word vectors for the target phrase.
- Embeddings seem to make a huge difference, state of the art is only obtained with 200-dimensional GloVe embeddings.


#### Notes/Questions

- A *huge* fraction of the performance improvement comes from pre-trained word embeddings. Without these, the proposed models clearly underperforms simpler models. This raises the question of whether incorporating the same embeddings into the simpler models would do.
- Would've liked to see performance without *any* pre-trained embeddings.
- The authors also experimented with attention mechanisms, but weren't able to achieve good results. Small size of training corpus may be the reason for this.
