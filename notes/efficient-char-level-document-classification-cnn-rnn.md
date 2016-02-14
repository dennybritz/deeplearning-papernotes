## [Efficient Character-level Document Classification by Combining Convolution and Recurrent Layers](http://arxiv.org/abs/1602.00367)

TLDR; The authors use a CNN to extract features from character-based document representations. These features are then fed into a RNN to make a final prediction. This model, called ConvRec, has significantly fewer parameters (10-50x) then comparable convolutional models with more layers, but achieves similar to better performance on large-scale document classification tasks.

#### Key Points

- Shortcomings of word-level approach: Each word is distinct despite common roots, cannot handle OOV words, many parameters.
- Character-level Convnets need many layers to capture long-term dependencies due to the small sizes of the receptive fields.
- Network architecture: 1. Embedding 8-dim 2. Convnet: 2-5 layers, 5 and 3-dim convolutions, 2-dim pooling, ReLU activation, 3. RNN LSTM with 128d hidden state. Dropout after conv and recurrent layer.
- Training: 96 characters, Adadelta, batch size of 128, Examples are padded and masked to longest sequence in batch, gradient norm clipping of 5, early stopping
- Models tends to outperform large CNN for smaller datasets. Maybe because of overfitting?
- More convolutional layers or more filters doesn't impact model performance much

#### Notes/Questions

- Would've been nice to graph the effect of #params on the model performance. How much do additional filters and conv layers help?
- hat about training time? How does it compare? 