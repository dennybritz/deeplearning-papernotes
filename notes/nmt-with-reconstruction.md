## [Neural Machine Translation with Reconstruction](https://arxiv.org/abs/1611.01874v1)

TLDR; The authors add a reconstruction objective to the standard seq2seq model by adding a "Reconstructor" RNN that is trained to re-generate the source sequence based on the hidden states of the decoder. A reconstruction cost is then added to the cost function and the architecture is trained end-to-end. The authors find that the technique improves upon the baseline both when 1. used during training only and 2. when used as a rankign objective during beam search decoding.

#### Key Points

- Problem to solve:
  - Standard seq2seq models tend to under- and over-translate because they don't ensure that all of the source information is covered by the target side.
  - The MLE objective only captures information from source -> target, which favors short translations. Thus, Increasing the beam size actually lowers translation quality
- Basic Idea
  - Reconstruct source sentences form the latent representations of the decoder
  - Use attention over decoder hidden states
  - Add MLE reconstruction probability to the training objective
- Beam Decoding is now two-phase scheme
  1. Generate candidates using the encoder-decoder
  2. For each candidate, compute a reconstruction score and use it to re-rank  together with the likelihood
- Training Procedure
  - Params Chinese-English: `vocab=30k, maxlen=80, embedding_dim=620, hidden_dim=1000, batch=80`.
  - 1.25M pairs trained for 15 epochs using Adadelta, the train with reconstructor for 10 epochs.
- Results:
  - Model increases BLEU from 30.65 -> 31.17 (beam size 10) when used for training only and decoding stays unchaged
  - BLEU increase from 31.17 -> 31.73 (beam size 10) when also used for decoding
  - Model successfully deals with large decoding spaces, i.e. BLEU now increases together with beam size


#### Notes

- [See this issue for author's comments](https://github.com/dennybritz/deeplearning-papernotes/issues/3)
- I feel like "adequacy" is a somewhat strange description of what the authors try to optimize. Wouldn't "coverage" be more appropriate?
- In Table 1, why does BLEU score still decrease when length normalization is applied? The authors don't go into detail on this.
- The training curves are a bit confusing/missing. I would've liked to see a standard training curve that shows the MLE objective loss and the finetuning with reconstruction objective side-by-side.
- The training procedure somewhat confusing. The say "We further train the model for 10 epochs" with reconstruction objective, byt then "we use a trained model at iteration 110k". I'm assuming they do early-stopping at 110k * 80 = 8.8M steps. Again, would've liked to see the loss curves for this, not just BLEU curves.
- I would've liked to see model performance on more "standard" NMT datasets like EN-FR and EN-DE, etc.
- Is there perhaps a smarter way to do reconstruction iteratively by looking at what's missing from the reconstructed output? Trainig with reconstructor with MLE has some of the same drawbacks as training standard enc-dec with MLE and teacher forcing.
