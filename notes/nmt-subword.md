## [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)

TLDR; The authors propose a technique novel technique to segment words into "subword units" based on the Byte Pair Encoding (BPE) algorithm in order to handle open vocabularies in Neural Machine Translation (NMT). Using this technique the authors achieve significant improvements over baseline systems without needing to resort to tricks such as UNK replacement and backoff dictionary alignments.

### Key Points

- Code at https://github.com/rsennrich/subword-nmt
- Intution: Many words are translatable using smaller units than words (e.g. word stems, suffixes, etc).
- Translation of rare words is an open problem. Typically, NMT vocab is limited to fixed 30k-50k words.
- Technique is purely a data pre- and post-processing step. Nothing in the model needs to change.
- Basic idea behind algorithm: Start with a vocabulary of all character in the text, then iteratively merge the most frequent pair of characters or character sequences. Thus, the final vocabulary size is `num_chars + num_merge_operations` where `num_merge_operations` is a hyperparameter of the method.
- Can learn BPE for source and target separately, or joint. Joint BPE has the advantage that words appearing in both source and target are split exactly the same way, making it easier for the model to learn the alignments.
- Training details:
  - Trained using [Groundhog](https://github.com/sebastien-j/LV_groundhog)
  - Model based on [Bahdanau's alignment paper](https://arxiv.org/abs/1409.0473)
  - Beam size 12; Adadelta with batch size 80;
  - Authors try 60k separate and 90k joint BPE vocab for English-German


### Thoughts

- Vocab size of 90k or 60k seems quite large to me. I wonder if a much smaller BPE vocab size (20k, 30k) is enough.
