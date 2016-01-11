## [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://arxiv.org/abs/1502.03044)

TLDR; The authors use an attention mechanism in image caption generation, allowing the decoder RNN focus on specific parts of the image. In order find the correspondence between words and image patches, the RNN uses a lower convolutional layer as its input (before pooling). The authors propose both a "hard" attention (trained using sampling methods) and "soft" attention (trained end-to-end) mechanism, and show qualitatively that the decoder focuses on sensible regions while generating text, adding an additional layer of interpretability to the model. The attention-based models achieve state-of-the art on Flickr8k, Flickr30 and MS Coco.

#### Key Points

- To find image correspondence use lower convolutional layers to attend to.
- Two attention mechanisms: Soft and hard. Depending on evaluation metric (BLEU vs. METERO) one or the other performs better.
- Largest data set (MS COCO) takes 3 days to train on Titan Black GPU. Oxford VGG.
- Soft attention is same as for seq2seq models.
- Attention weights are visualized by upsampling and applying a Gaussian

#### Notes/Questions

- Would've liked to see an explanation of when/how soft vs. hard attention does better.
- What is the computational overhead of using the attention mechanism? Is it significant?
