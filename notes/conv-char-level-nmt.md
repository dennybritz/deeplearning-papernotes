## [Fully Character-Level Neural Machine Translation without Explicit Segmentation](https://arxiv.org/abs/1610.03017)

TLDR; The authors propose a character-level Neural Machine Translation (NMT) architecture. The encoder is a convolutional network with max-pooling and highway layers that reduces size of the source representation. It does not use explicit segmentation. The decoder is a standard RNN. The authors apply their model to WMT'15 DE-EN, CS-EN, FI-EN and RU-EN data in bilingual and multilingual settings. They find that their model is competitive in bilingual settings and significantly outperforms competing models in the multilingual setting with a shared encoder.

#### Key Points

- Challenge: Apply standard seq2seq models to characters is hard because representation is too long. Attention network complexity grows quadratically with sequence length.
- Word-Level models are unable to model rare and out-of-vocab tokens and softmax complexity grows with vocabulary size.
- Character level models are more flexible: No need for explicit segmentation, can model morphological variants, multilingual without increasing model size.
- Reducing the length of the source sentence is key to fast training in char models.
- Encoder Network: Embedding -> Conv -> Maxpool -> Highway -> Bidirectional GRU
- Attenton Network: Single Layer
- Decoder: Two Layer GRU
- Multilingual setting: Language examples are balanced within each batch. No language identifier is provided to the encoder
- Bilingual Results: char2char performs as well as or better than bpe2char or bpe2bpe
- Multilingual Results: char2char outperforms bpe2char
- Trained model is robust to spelling mistakes and unseen morphologies
- Training time: Single Titan X training time for bilingual model is ~2 weeks. ~2.5 updates per second with batch size 64.


#### Notes

- I wonder if you can extract segmentation info from the network post training.