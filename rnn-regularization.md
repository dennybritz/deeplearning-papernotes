## [Recurrent Neural Network Regularization](http://arxiv.org/abs/1409.2329)

TLDR; The authors show that applying dropout to only the **non-recurrent** connections (between layers of the same timestep) in an LSTM works well, improving the scores on various sequence tasks.

#### Data Sets and model performance

- PTB Language Modeling Perplexity: 78.4
- Google Icelandic Speech Dataset WER Accuracy: 70.5
- WMT'14 English to French Machine Translation BLEU: 29.03
- MS COCO Image Caption Generation BLEU: 24.3