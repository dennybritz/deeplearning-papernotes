## [Professor Forcing: A New Algorithm for Training Recurrent Networks](https://arxiv.org/abs/1610.09038)

TLDR; The authors adopt Generative Adversarial Networks (GANs) to RNNs and train a discriminator to distinguish between sequences generated using teacher forcing (feeding ground truth inputs to the RNN) and scheduled sampling (feeding generated outputs as the next inputs). The inputs to the discriminator are both the predictions and the hidden states of the generative RNN. The generator is trained to fool the discriminator, forcing the dynamics of teacher forcing and scheduled sampling to become more similar. This procedure acts as regularizer, and results in better sample quality and generalization, particularly for long sequences. The authors evaluate their framework on Language Model (PTB), Pixel Generation (Sequential MNIST), Handwriting Generation, and Musisc Synthesis.

### Key Points

- Problem: During inference, errors in an RNN easily compound because the conditioning context may diverge from what is seen during training when the ground-truth labels are fed as inputs (teacher forcing).
- Goal of professor forcing: Make the generative (free-run) behavior and the teacher-forced behavior match as closely as possible.
- Discriminator Details
  - Input is a behavior sequence `B(x, y, theta)` from the generative RNN that contains the hidden states and outputs.
  - The training objective is to correctly classify whether or not a behavior sequence is generated using teacher forcing vs. scheduled sampling.
- Generator
  - Standard RNN with MLE training objective and an additional term to fool the discrimator: Change the free-running behavior as to match the teacher-forced behavior while keeping the latter constant.
  - Second optional another term: Change the teacher-forced behavior to match the free-running behavior.
  - Like GAN, backprop from discriminator into generator.
- Architectures
  - Generator is a standard GRU Recurrent Neural Network with softmax
  - Behavior function `B(x, y, theta)` outputs pre-tanh activation of GRU states and tje softmax output
  - Discriminator: Bidirectional GRU with 3-layer MLP on top
  - Training trick: To prevent "bad gradients" the authors backprop from the discriminator into the generator only if the classification accuracy is between 75% and 99%.
  - Trained used Adam optimizer
- Experiments
  - PTB Chracter-Level Modeling: Reduction in test NLL, profesor forcing seem to act as a regularizier. 1.48 BPC
  - Sequential MNIST: Second-best NLL (79.58) after PixelCNN
  - Handwriting generation: Professor forcing is better at generating longer sequences than seen during training as per human eval.
  - Music Synthesis: Human eval significantly better for Professor forcing
  - Negative Results on word-level modeling: Professor forcing doesn't have any effect. Perhaps because long-term dependencies are more pronounced in character-level modeling.
  - The authors show using t-SNE that the hidden state distributions actually become more similar when using professor forcing

### Thoughts

- Props to the authors for a very clear and well-written paper. This is rarer than it should be :)
- It's an intersting idea to also match the states of the RNN instead of just the outputs. Intuitively, matching the outputs should implicitly match the state distribution. I wonder if the authors tried this and it didn't work as expected.
- Note from [Ethan Caballero](https://github.com/ethancaballero) about why they chose to match hidden states: It's significantly harder to use GANs on sampled (argmax) output tokens because they are discrete as (as opposed to continuous like the hidden states and their respective softmaxes). They would have had to estimate discrete outputs with policy gradients like in [seqGAN](https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/seq-gan.md) which is [harder to get to converge](https://www.quora.com/Do-you-have-any-ideas-on-how-to-get-GANs-to-work-with-text), which is why they probably just stuck with the hidden states which already contain info about the discrete sampled outputs (the index of the highest probability in the the distribution) anyway. Professor Forcing method is unique in that one has access to the continuous probability distribution of each token at each timestep of the two sequence generation modes trying to be pushed closer together. Conversely, when applying GANs to pushing real samples and generated samples closer together as is traditionally done in models like seqGAN, one only has access to the next dicrete token (not continuous probability distributions of next token) at each timestep, which prevents straight-forward differentiation (used in professor forcing) from being applied and forces one to use policy gradient estimation. However, there's a chance one might be able to use straight-forward differentiation to train seqGANs in the traditional sampling case if one swaps out each discrete sampled token with its continuous distributional word embedding (from pretrained word2vec, GloVe, etc.), but no one has tried it yet TTBOMK.
- I would've liked to see a comparison of  the two regularization terms in the generator. The experiments don't make it clear if both or only of them them is used.
- I'm guessing that this architecture is quite challenging to train. Woul've liked to see a bit more detail about when/how they trade off the training of discriminator and generator.
- Translation is another obvious task to apply this too. I'm interested whether or not this works for seq2seq.

