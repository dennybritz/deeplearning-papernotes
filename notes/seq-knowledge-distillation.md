## [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)

TLDR; The authors train a standard Neural Machine Translation (NMT) model (the teacher model) and distill it by having a smaller student model learn the distribution of the teacher model. They investigate three types of knowledge distillation for sequence models: 1. Word Level Distillation 2. Sequence Level Distillation and 3. Sequence Level Interpolation. Experiments on WMT'14 and IWSLT 2015 show that it is possible to significantly reduce the parameters of the model with only a minor loss in BLEU score. The experiments also demonstrates that the distillation techniques are largely complementary. Interestingly, the perplexity of distilled models is significantly higher than that of the baselines without leading to a loss in BLEU score.

### Key Points

- Knowledge Distillation: Learn a smaller student network from a larger teacher network.
- Approach 1 - Word Level KD: This is standard Knowledge Distillation applied to sequences where we match the student output distribution of each word to the teacher's using the cross-entropy loss.
- Approach 2 - Sequence Level KD: We want to mimic the distribution of a full sequence, not just per word. To do that we sample outputs from the teacher using beam search and then train the student on these "examples" using Cross Entropy. This is a very sparse approximation of the true objective.
- Approach 3: Sequence-Level Interpolation: We train the student on a mixture of training data and teacher-generated data. We could use the approximation from #2 here, but that's not ideal because it doubles size of training data and leads to different targets conditioned on the same source. The solution is to use generate a response that has high probability under the teacher model and is similar to the ground truth and then have both mixture terms use it.
- Greedy Decoding with seq-level fine-tuned model behaves similarly to beam search on original model.
- Hypothesis: KD allows student to only model the mode of the teacher distribution, not wasting other parameters. Experiments show good evidence of this. Thus, greedy decoding has an easier time finding the true max whereas beam search was necessary to do that previously.
- Lower perplexity does not lead to better BLEU. Distilled models have significantly higher perplexity (22.7 vs 8.2) but have better BLEU (+4.2).
