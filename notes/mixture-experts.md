## [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://openreview.net/forum?id=B1ckMDqlg)

TLDR; The authors introduce a new type of layer, the Sparsely-Gated Mixture-of-Experts (MoE). The layer consists of many experts subnetwork (feedforward networks in this work) and a parameterized gating network that decides which experts to use. Using this approach the authors can train a network with tens of billions of parameters while keeping computation cost constant. The architecture achieves state of the art results on Language Modeling (1 Billion Words Benchmark) and Machine Translation (WMT EN->FR, EN-DE) tasks.

### Key Points

- MoE Layer: Consists of n experts networks `E_i`. Outputs are chosen using a gating network `G(x)` and the output of the layer is given by `y = sum(G(x)_i*E_i(x))`.
- `G(x)` is sparse, e.g. a softmax followed by a top-k mask, but could be something more complex.
- Ensuring expert utilization (i.e. exploration) is a challenge. An extra loss term is added to encourage this.
- Shrinking batch problem: Because experts are chosen per example, each experts may receive a much smaller batch than the original batch size. Authors propose several batch combination strategies to combat this.
- Billion Word Language Modeling: 29.9 perplexity with 34B parameters, 15h training time on 128 k40s. Each experts is a 512-dimensional linear feedforward network.
- NMT EN-FR: 40.56 BLEU with 8.7B parameters. Each expert is a 2-layer network with ReLU activation,

### Notes

- This work reminds of of Highway Networks, with an additional constraint to make the gates sparse to save computation.
- It's surprising to me that the authors didn't evaluate using different architectures for each experts network. That would've been the first use case that comes to my mind. They mention this possibility in the paper but I would've loved to see experiments for this.
- `We used the Adam optimizer (Kingma & Ba, 2015). The learning rate was
increased linearly for the first 200 training steps, held constant for the next 200 steps, and decreased
after that so as to be proportional to the inverse square root of the step number.` - Hmmm.... :)
