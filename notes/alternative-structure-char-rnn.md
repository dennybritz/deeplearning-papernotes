## [Alternative structures for character-level RNNs](http://arxiv.org/abs/1511.06303)

TLDR; The authors propose two different architectures to improve the performance of character-level RNNs. In the first architecture ("mixed") the authors condition the model on the state of a word-level RNN. In the second architecture ("cond") they condition the output classifier on character n-grams. The authors show that the proposed architecture outperform plain character-level RNNs in terms of entropy in bits per character.

#### Key Points

- Plain character-level RNNs need a huge hidden representation in order to model long-term dependencies. But Word-level RNNs can't generalize to new vocabulary and may require a huge output vocab.
- Model 1: Jointly train word-level and char-level CNN. Interpolate the losses of the two models.
- Model 2: Condition softmax on n-grams before character, "relieving" the network of memorizing some of the sequence.
- Training: Constant learning rate, reduce every epoch when validation accuracy decreases
- N-gram model can be applied to arbitrary data, not just characters. Authors evaluate on binary data.

#### Notes / Questions

- In the comparison table the authors don't show the number of parameters for the models. They compare models with the same number of hidden units, but their proposed architecture need extra parameters and computation. Unfair comparison?
- People typically use LSTMs/GRUs for language modeling. Of course the proposed techniques can be applied to LSTM/GRU networks, but the experimental result may look very different. Do these architectures result in any benefit when using LSTM/GRU char data?
- Entropy in bits per character seems like somewhat of a strange evaluation metric. I don't really know what to make of it, and no intuitive explanations are given.
- One argument the authors make in the paper is that character-level models can be applied to arbitrary input data (different languages, binary data, code, etc). But their mixed is clearly very language-specific. It can't be applied to arbitrary data, and many languages don't have clear word boundaries. Similarly, n-grams may be prohibituvely expensive depending on what kind of data we're working with. 
- The n-gram conditioned models isn't clearly explained, I *think* I understand what it does, but I'm not quite sure. No intuitive explanations what any of the models are learning are given.