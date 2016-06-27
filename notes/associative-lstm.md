## [Associative Long Short-Term Memory](http://arxiv.org/abs/1602.03032)

TLDR; The authors propose Associate LSTMs, a combination of external memory based on Holographic Reduced Representations and LSTMs. The memory provides noisy key-value lookup based on matrix multiplications without introducing additional parameters. The authors evaluate their model on various sequence copying and memorization tasks, where it outperforms vanilla LSTMs and competing models with a similar number of parameters.

#### Key Points

- Two limitations of LSTMs: 1. N cells require NxN weight matrices. 2. Lacks mechanism to index memory
- Idea of memory comes from "Holographic Reduced Representations" (Plate, 2003), but authors add multiple redundant memory copies to reduce noise.
- More copies of memory => Less noise during retrieval
- In the LSTM update equations input and output keys to the memory are computed
- Compared to: LSTM, Permutation LSTM, Unitary LSTM, Multiplicative Unitary LSTM
- Tasks: Episodic Copy, XML modeling, variable assignment, arithmetic, sequence prediction

#### Notes

- Only brief comparison with Neural Turing Machines in appendix. Probably NTMs outperform this and are simpler. No comparison with attention mechanisms, memory networks, etc. Why?
- It's surprising to me that deep LSTM without any bells and whistles actually perform pretty well on many of the tasks. Is the additional complexity really worth it?