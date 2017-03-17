## [A recurrent neural network without Chaos](https://arxiv.org/abs/1612.06212)

TLDR; The authors propose the Chaos-Free-Network (CFN), a mathematically simpler alternative to LSTM and GRU cells that exhibits interpretable non-chaotic dynamics but (nearly) matches the performance of the LSTM cell on language modeling tasks;

### Key Points

- In the absence of inputs, LSTM/GRU/etc give rise to chaotic dynamical systems, i.e. their activations don't follow a predictable path.
  - Experiments show that small perturbations (1e-7) to the initial state result in widely different trajectories
  - When inputs at each time step are present, LSTM/GRU are *externally-driven* and tend to converge to the same path even with small perturbations in initial states. Still, it is difficult to interpret this trajectory due to their chaotical nature.
- The CFN has trivial dynamics. In the absence of inputs its state will decay to 0. Higher layers decay more slowly.
  - This makes the trajectory taken by CFN interpretable, even when inputs are present
  - First layer retains information for 10-20 timesteps, secnd layer for ~100.
- Surprisingly, despite their simply trajectories, CFNs can match LSTM performance. This proves that LSTM do not perform well solely due to their chaotic nature.
- PTB Language Modeling (with dropout)
  - 2-layer LSTM: 71.8
  - 2-layer CFN: 72.2 (Same # of parameters as 1 layer LSTM)
- Text8 Language Modeling (no dropout)
  - 2-layer LSTM: 139.9
  - 2-layer CFN: 142.0 (Same # of parameters as 1 layer LSTM)

### Thoughts

- This paper has a really good introduction/review of dynamic systems and chaotical behavior. Without any background on this topic I was able to follow the paper.
- Very cool result that the chaotic nature is not necessary to achieve good performance in language modeling (and probably other) tasks.
- The fact that upper layers decay more slowly seems like a useful property. I wonder if this can be used to model long-term dependencies more efficiently than one could with GRU/LSTM.
- It seems like the practical applicability of the results would be limited to improving interoperability of RNNs, since LSTM/GRUs are externally driven when presented with inputs, which is almost always the case.

