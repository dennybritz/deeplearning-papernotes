## [A Neural Network Approach to Context-Sensitive Generation of Conversational Responses](http://arxiv.org/abs/1506.06714)

TLDR; The authors propose three neural models to generate a response (r) based on a context and message pair (c,m). The context is defined as a single message. The first model, RLMT, is a basic Recurrent Language Model that is fed the whole (c,m,r) triple. The second model, DCGM-1, encodes context and message into a BoW representation, put it through a feedforward neural network encoder, and then generates the response using an RNN decoder. The last model, DCGM-2, is similar but keeps the representations of context and message separate instead of encoding them into a single BoW vector. The authors train their models on 29M triple data set from Twitter and evaluate using BLEU, METEOR and human evaluator scores.

#### Key Points:

- 3 Models: RLMT, DCGM-1, DCGM-2
- Data: 29M triples from Twitter
- Because (c,m) is very long on average the authors expect RLMT to perform poorly.
- Vocabulary: 50k words, trained with NCE loss
- Generates responses degrade with length after ~8 tokens


#### Notes/Questions:

- Limiting the context to a single message kind of defeats the purpose of this. No real conversations have only a single message as context, and who knows how well the approach works with a larger context?
- Authors complain that dealing with long sequences is hard, but they don't even use an LSTM/GRU. Why?


