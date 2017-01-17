## [A Character-level Decoder without Explicit Segmentation for Neural Machine Translation](http://arxiv.org/abs/1603.06147)

TLDR; The authors evaluate the use of a character-level decoder in Neural Machine Translation (NMT), while keeping the encoder at the subword level using BPE. The authors also propose a biscale architecture with one a slow and fast layer in the decoder. In each of the cases (biscale or base), they show that character-level decoding outperforms word-level decoding on WMT EN-DE, EN-CS, EN-RU and EN-FI datasets.


### Key Points

- Use BPE subword units in encoder, characters in decoder. No explicit segmentation.
- Novel architecture: Biscale RNN as decoder. However, this did not seem to make a huge difference in the experiments.
- Data Processing: Moses Tokenizer, limit sequences to 50 subword symbols in source and 100 subwords symbols and 500 characters in target.
- Model based on [Bahdanau et al.](https://arxiv.org/abs/1409.0473) with similar hyperparameters. Bidirectional encoder has 512 units and decoder has 1024 units per layer. 2 decoder layers. Adam; batch size 128; gradient clipping at norm 1;
- Attention visualizations show that decoded characters are actually well-aligned with translated source subword units.


### Thoughts

- Really well-written paper with good explanations and visualizations and an excellent table reporting variance. It's rare to see this.
- With all the architecture in place it seemed a bit strange to me that the authors didn't also evaluate character-level encoding. Either they did not have enough time for the experiments or they didn't yield good results.