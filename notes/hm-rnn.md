## [Hierarchical Multiscale Recurrent Neural Networks](https://arxiv.org/abs/1609.01704)

TLDR; The authors propose a new "Hierarchical Multiscale RNN" (HM-RNN) architecture. This models explicitly learns both temporal and hierarchical (character -> word -> phrase -> ...) representations without needing to be told what the structure or timescale of the hierarchy is. This is done by adding binary boundary detectors at each layer. These detectors activate based on whether the segment in a certain layer is finished or not. Based on the activation of these boundary detectors information is then propagated to neighboring layers. Because this model involves discrete decision making based on binary outputs it is trained using a straight-through estimator. The authors evaluate the model on Language Modeling and Handwriting Sequence Generation tasks, where it outperforms competing models. Qualitatively the authors show that the network learns meaningful boundaries (e.g. spaces) without being needing to be told about them.

### Key Points

- Learning both hierarchical and temporal representations at the same time is a challenge for RNNs
- Observation: High-level abstractions (e.g. paragraphs) change slowly, but low-level abstractions (e.g. words) change quickly. These should be updated at different timescales.
- Benefits of HN-RNN: (1) Computational Efficiency (2) Efficient long-term dependency propagation (vanishing gradients) (3) Efficient resource allocation, e.g. higher layers can have more neurons
- Binary boundary detector at each layer is turned on if the segment of the corresponding layer abstraction (char, word, sentence, etc) is finished.
- Three operations based on boundary detector state: UPDATE, COPY, FLUSH
- UPDATE Op: Standard LSTM update. This happens when the current segment is not finished, but the segment one layer below is finished.
- COPY Op: Copies previous memory cell. Happens when neither the current segment nor the segment one layer below is finished. Basically, this waits for the lower-level representation to be "done".
- FLUSH Op: Flushes state to layer above and resets the state to start a new segment. Happens when the segment of this layer is finished.
- Boundary detector is binarized using a step function. This is non-differentiable and training is done with a straight-through estimator that estimates the gradient using a similar hard sigmoid function.
- Slope annealing trick: Gradually increase the slop of the hard sigmoid function for the boundary estimation to make it closer to a discrete step function over time. Needed to be SOTA.
- Language Modeling on PTB: Beats state of the art, but not by much.
- Language Modeling on other data: Beats or matches state of the art.
- Handwriting Sequence Generation: Beats Standard LSTM


### My Notes

- I think the ideas in this paper are very important, but I am somewhat disappointed by the results. The model is significantly more complex with more knobs to tune than competing models (e.g. a simple batch-normalized LSTM). However, it just barely beats those simpler models by adding new "tricks" like slope annealing. For example, the slope annealing schedule with a `0.04` constant looks very suspicious.
- I don't know much about Handwriting Sequence Generation, but I don't see any comparisons to state of the art models. Why only compare to a standard LSTM?
- The main argument is that the network can dynamically learn hierarchical representations and timescales. However, the number of layers implicitly restricts how many hierarchical representations the network can and cannot learn. So, there still is a hyperparameter involved here that needs to be set by hand.
- One claim is that the model learns boundary information (spaces) without being told about them. That's true, but I'm not convinced that's as novel as the authors make it out to be. I'm pretty sure that a standard LSTM (perhaps with extra skip connections) will learn the same and that it's possible to tease these out of the LSTM parameter matrices.
- Could be interesting to apply this to CJK languages where boundaries and hierarchical representations are more apparent.
- The authors claim that "computational efficiency" is one of the main benefits of this model because higher level representations need to be updated less frequency. However, there are no experiments to verify this claim. Obviously this is true in theory, but I can imagine that in practice this model is actually slower to train. Also, what about convergence time?

