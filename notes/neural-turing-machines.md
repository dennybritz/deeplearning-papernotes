## [Neural Turing Machines](http://arxiv.org/abs/1410.5401)

TLDR; The authors propose Neural Turing Machines (NTMs). A NTM consists of a memory bank and a controller network. The controller network (LSTM or MLP in this paper) controls read/write heads by focusing their attention softly, using a distribution over all memory addresses. It can learn the parameters for two addressing mechanisms: Content-based addressing ("find similar items") and location-based addressing. NTMs can be trained end-to-end using gradient descent. The authors evaluate NTMs on program generations tasks and compare their performance against that of LSTMs. Tasks include copying, recall, prediction, and sorting binary vectors. While both LSTMs and NTMs seems to perform well on training data, only NTMs are able to generalize to longer sequences.


#### Key Observations

- Controller network tried with LSTM or MLP. Which one works better is task-dependent, but LSTM "cache" can be a bottleneck.
- Controller size, number  of read/write heads, and memory size are hyperparameters. 
- Monitoring the memory addressing shows that the NTM actually learns meaningful programs.
- Number LSTM parameters grow quadratically with hidden unit size due to recurrent connection, not so for NTMs, leading to models with fewer parameters.
- Example problems are very small, typically using sequences 8 bit vectors.


#### Notes/Questions

- At what length to NTMs stop to work? Would've liked to see where results get significantly worse.
- Can we automatically transform fuzzy NTM programs into deterministic ones?
