### [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)

TLDR; The authors propose a recurrent memory-based model that can reason over multiple hops and be trained end to end with standard gradient descent. The authors evaluate the model on QA and Language Modeling Tasks. In the case of QA, the network inputs are a list of sentences, a query and (during training) an answer. The network then attends to the sentences at each time step, considering the next piece information relevant to the question. The network outperforms baseline approaches, but does not come close to a strongly supervised (relevant sentences are pre-selected) approach.


#### Key Takeaways

- Sentence Representation: 1. Word embeddings are averaged (BoW) 2. Positional Encoding (PE)
- Synthetic dataset with vocabulary size of ~180. Version one has 1k training example, version 2 has 10k training examples.
- The model is similar to Bahdanau seq2seq attention model, only that it operates on sentences and does not output at every step and used a simpler scoring function.


#### Questions / Notes

- The positional encoding formula is not explained neither is it intutiive.
- There are so many hyperparameters and model variations (jittering, linear start) that it's easy to lose track of the essential.
- No intuitive explanation of what the model does. The easiest way for me to understand this model was to look at it as a variation of Bahdanau's attention model, which is very intuitive. I don't understand the intuition behind the proposed weight constraints.
- The LM results are not convincing. The model beats the baselines by a little bit, but probably only due to very time-intensive hyperparameter optimization.
- What is the training complexity and training time?

