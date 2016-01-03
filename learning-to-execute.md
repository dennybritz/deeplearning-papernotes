## [Learning to Execute](http://arxiv.org/abs/1410.4615)

TLDR; The authors show that seq2seq LSTM networks (2 layers, 400-dims) can learn to evaluate short Python programs (loops, conditionals, addition, subtraction, multiplication). The program code is fed one character at a time, and the LSTM is tasked with generating an output number (12 character vocab). The authors also present a new curriculum learning strategy, where the network is fed with a sensible mixture of easy and increasingly difficult examples, allowing it to gradually build up the concepts required to evaluate these programs.

#### Key Points

- LSTM unrolled for 50 steps, 2 layer, 400 cells per layer, ~2.5M parameters. Gradient norm constrained to 5.
- 3 Curriculum Learning strategies: 1. Naive (increase example difficulty) 2. Mixed: Randomly sample easy and hard problems, 3. Combined: Sample from Naive and Mixed strategy. Mixed or Combined almost always performs better.
- Output Vocabulary: 10 digits, minus, dot
- For evaluation teacher forcing is used: Feed correct output when generating target sequence
- Evaluation Tasks: Program Evaluation, Addition, Memorization
- Tricks: Reverse Input sequence, Double input sequence. Seem to make big difference.
- Nesting loops makes the tasks difficult since LSTMs can't deal with compositionality.
- Feeding easy examples and before hard examples may require the LSTM to restructure its memory.

#### Notes / Questions

- I wonder if there's a relation between regularization/dropout and curriculum learning. The authors propose that mixing example difficulty forces a more general representation. Shouldn't dropout be doing a similar thing?