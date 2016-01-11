## [A Unified Tagging Solution: Bidirectional LSTM Recurrent Neural Network with Word Embedding](http://arxiv.org/abs/1511.00215)

TLDR; The authors evaluate the use of a Bidirectional LSTM RNN on POS tagging, chunking and NER tasks. The inputs are task-independent input features: The word and its capitalization. The authors incorporate prior knowledge about the taging tasks by restricting the decoder to output valid sequences of tags, and also propose a novel way of learning word embeddings: Randomly replacing words in a sequence and using an RNN to predict which words are correct vs.  incorrect. The authors show that their model combined with pre-trained word embeddings performs on par state of the art models.


#### Key Points

- Bidirectional LSTM with 100-dimensional embeddings, and 100-dimensional cells. Both 1 and 2 layers are evaluated. Predict tags at each step. Higher dimensionality of cells resultes in little improvement.
- Word vector pretraining: Randomly replace words and use LSTM to predict correct/incorrect words.


#### Notes/Questions

- The fact that we need a task-specific decoder kind of defeats the purpose of this paper. The goal was to create a "task-independent" system. To be fair, the need for this decoder is probably only due to the small size of the training data. Not all tag combination appear in the training data.
- The comparisons with other state of the art systems are somewhat unfair since the proposed model heavily relies on pre-trained word embeddings from external data (trained on more than 600M words) to achieve good performance. It also relies on external embeddings trained in yet another way.
- I'm surprised that the authors didn't try combining all of the tagging tasks into one model, which seem like an obvious extension.