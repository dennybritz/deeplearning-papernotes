### [Grammar as a Foreign Langauage](http://arxiv.org/abs/1412.7449)

TLDR; Authors apply 3-layer seq2seq LSTM with 256 units and attention mechanism to consituency parsing task and achieve new state of the art. Attention made a huge difference for a small dataset (40k examples), but less so for a noisy large dataset (~11M examples).

#### Data Sets and model performance

- WSJ (40k examples): 90.5
- Large distantly supervised corpus (90k gold examples, 11M noisy examples): 92.8

#### Key Takeaways

- The authors use existing parsers to label a large dataset to be used for training. The trained model then outperforms the "teacher" parsers. A possible explanation is that errors of supervising parsers look like noise to the more powerful LSTM model. These results are extremely valuable, as data is typically the limiting factor, but existing models almost always exist.
- Attention mechanism can lead to huge improvements on small data sets.
- All of the learned LSTM models were able to deal with long (~70 tokens) sentences without a significant impact of performance.
- Reversing the input in seq2seq tasks is common. However, reversing resulted in only a 0.2 point bump in accuracy.
- Pre-trained word vectors bumped scores by 0.4 (92.9 -> 94.3) only.

#### Notes/Questions

- How much does the ouput data representation matter? The authors linearized the parse tree using depth-first traversal and parentheses. Are there more efficient representations that may lead to better results?
- How much does the noise in the auto-labeled training data matter when compared to the data size? Are there systematic errors in the auto-labeled data that put a ceiling on model performance?
- Bidirectional LSTM?
