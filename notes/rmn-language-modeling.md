## [Recurrent Memory Network for Language Modeling](http://arxiv.org/abs/1601.01272)

TLDR; the authors present Recurrent Memory Network. These networks use an attention mechanism (memory bank, MB) to explicitly incorporate information about preceding into the predictions at each time step. The MB is a layer that can be incorporated into any RNN, and the authors evaluate a total of 8 model variants: Optionally stacking another LSTM layer on top of the MB, optionally including a temporal matrix in the attention calcuation, and using a gating vs. linear function for the MB  output. The authors apply the model to Language Modeling tasks, achieving state of the art performance, and demonstrating that inspecting the attention weights yields intuitive insights into what the network learns: Co-occurence statistics and dependency type information. The authors also evaluate the models on a sentence completion task, achieving new state of the art.


#### Key Points

- RM: LSTM with MB as the top layer. No "horizontal" connections from MB to MB.
- RMR:  LSTM with MB and another LSTM stacked on top.
- RM with gating typically outperforms RMR.
- Memory Bank (MB): Input is current hidden state, and n preceding inputs including the current one. Attention is then calculated over the inputs based on the hidden state. The Output is a new hidden state, which can be calculated with or without gating. Optionally apply temporal bias matrix to attention calculation.
- Experiments: Hidden states and embeddings all of size 128. Memory size 15. SGD 15 epochs, halved each epoch after the forth.
- Attention Analysis (Language Model): Obviously, most attention is given to current and recent words. But long-distance dependencies are also captured, e.g. separable verbs in German. Networks also discovers dependency types.


#### Notes/Questions

- This works seems related to "Alternative structures for character-level RNNs" where the authors feed n-grams from previous words into the classification layer. The idea is to relieve the network from having to memorize these. I wonder how the approaches compare. 
- No related work section? I don't know if I like the name memory bank and the reference to Memory Networks here. I think the main idea behind Memory Networks was to reason over multiple hops. The authors here only make one hop, which is essentially just a plain attention mechanism.
- I wonder why exactly the RMR performs worse than the RM. I can't easily find an intuitive explanation for why that would be. Maybe just not enough training data?
- How did the authors arrive at their hyperparameters (128 dimensions)? 128 seems small compared to other models.
