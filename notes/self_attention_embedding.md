## A Structured Self-attentive Sentence Embedding [[arXiv](https://arxiv.org/abs/1703.03130)]

TLDR; The authors present a new sentence embedding algorithm where each embedding corresponds to a 2-d matrix. This is done by 1. Running the sentence through an RNN, 2. learning **multiple** attention values for each RNN state, 3. encouraging each attention vector to focus on different parts of the sentence by adding a penalty term. The model is evaluated on several classification tasks and outperforms other methods.

## Key Points

- Sentence Encoder is a bidirectional RNN with each state being concatenated forward and backward states
- Attention values are calculated over encoder states as `A = softmax(U*tanh(W*H^t))`. The number of rows in `U` defines how many attention vectors we want.
  - This is just basic attention, nothing special here, except that it's calculated multiple times with different weights
- Penalization term encourages rows in attention matrix to be different from one another.
  - `P = ||AA^T - I||^2`
- Learned sentence embeddings can easily be visualized through attention scores
- The authors empirically show that the penalization term helps and that multiple attention vectors are important

## Thoughts

- The term multi-hop attention or is misleading here. There are no multiple hops of attention, just a single hop of multiple attention calculations independent of one another. Multiple hops would corresponds to taking attention over attention, etc.
- Nice analysis on the hyperparameter choices of the mechanism
- I was surprised that the authors subsample data for some of the dataset, I wonder what the reason for that is.
- I think this technique can eaisly be extended to other tasks than use attention. E.g. it would be interesting to apply it to NMT.
- Overall I think this is a simple but cool technique