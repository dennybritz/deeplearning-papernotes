## [Sequence to Sequence Learning with Neural Networks](http://arxiv.org/abs/1409.3215)

TLDR; The authors show that a multilayer LSTM RNN (4 layers, 1000 cells per layer, 1000d embeddings, 160k source vocab, 80k target vocab) can achieve competitive results on Machine Translation tasks. The authors find that reversing the input sequence leads to significant improvements, most likely due to the introduction of short-term dependencies that are more easily captured by the gradients. Somewhat surprisingly, the LSTM did not have difficulties on long sentences. The model is evaluated on MT tasks and achieves competitive results (34.8 BLEU) by itself, and close to state of the art if coupled with existing baseline systems (36.5 BLEU).

#### Key Points

- Invert input sequence leads to significant improvement
- Deep LSTM performs much better than shallow LSTM.
- User different parameters for encoder/decoder. This allows parallel training for multiple languages decoders.
- 4 Layers, 1000 cells per layer. 1000-dimensional words embeddings. 160k source vocabulary. 80k target vocabulary.Trained on 12M sentences (652M words). SGD with fixed learning rate of 0.7, decreasing by 1/2 every epoch after 5 initial epochs. Gradient clipping. Parallelization on GPU leads to 6.3k words/sec.
- Batching sentences of approximately the same length leads to 2x speedup.
- PCA projection shows meaningful clusters of sentences robust to passive/active voice, suggesting that the fixed vector representation captures meaning. 
- "No complete explanation" for why the LSTM does so much better with the introduced short-range dependencies.
- Beam size 1 already performs well, beam size 2 is best in deep model.

#### Notes/Questions

- Seems like the performance here is mostly due to the computational resources available and optimized implementation. These models are pretty big by most standards, and other approaches (e.g. attention) may lead to better results if they had more computational resources.
- Reversing the input still feels like a hack to me, there should be a more principled solution to deal with long-range dependencies.