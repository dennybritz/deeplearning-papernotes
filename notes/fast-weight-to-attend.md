## [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258)

TLDR; The authors propose "fast weights", a type of attention mechanism to the recent past that performs multiple steps of computation between each hidden state computation step in an RNN. The authors evaluate their architecture on various tasks that require short-term memory, arguing that the fast weights mechanism frees up the RNN from memorizing sthings in the hidden state which is freed up for other types of computation.

### Key Points

- Currently, RNNs have slow-changing long-term memory (Permanent Weights) and fast changing short-term memory (hidden state). We want something in the middle: Fast weights with higher storage capacity.
- For each transition in the RNN, multiple transitions can be made by the fast weights. They are a kind of attention mechanism to the recent past that is not parameterized separately but depends on the past states.
- Fast weights are decayed over time and based on the outer product of previous hidden states: `A(t+1) = lambdaA(t) + eta*h(t)h(t)^T`.
- The next hidden state of the RNN is computed by a regular transition based on input adn previous state combined by an "inner loop" of S steps of the fast weights.
- "At each iteration of the inner loop the fast weight matrix A is eqivalent to attending to past hidden vectors in proportion to their scalar product with the current hidden state, weighted by a decay factor" - And this is efficient to compute.
- Added Layer Normalization to fast weights to prevent exploding/vanishign gradients.
- Associative Retrieval Toy Task: Memorize recent key-value pairs. Fast weights siginifcantly outperform RNN, LSTM and Associative LSTM.
- Visual Attention on MNIST: Beats RNN/LSTM and is comparable to CovnNet for large number of features.
- Agents with Memory: Fast Weight net learns faster in a partially obseverable environment where the networks must remember the previous states.

### Thoughts

-Overall I think this is very exciting work. It kind of reminds me of Adaptive Computation Time where you dynamically decide how many steps to "ponder" before making another outputs. However, it is also quite different in that this work explicitly "attends" over past states and isn't really about computation time.
- In the experiments the authors say they set S=1 (i.e. just one inner loop step). Why is that? I thought one of the more important points of fast weights would be to have additional computation betwene each slow step. This also raises the question of how to pick this hyperparameter.
- A lot of references to Machine Translation models with attention but not NLP experiments.