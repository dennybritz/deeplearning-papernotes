## [Strategies for Training Large Vocabulary Neural Language Models](http://arxiv.org/abs/1512.04906)

TLDR; The authors evaluate softmax, hierarchical softmax, target sampling, NCE, self-normalization and differentiated softmax (novel technique presented in the paper) on data sets with varying vocabulary size (10k, 100k, 800k) with a fixed-time training budget. The authors find that techniques that work best for small vocabluaries are not necessarily the ones that work best for large vocabularies.

#### Data and Models

Models:

- Sotmax
- Hierarchical Softmax (cross-validation of clustering techniques)
- Differentiated softmax, adjusting capacity based on token frequency (cross-validation of number of frequency bands and size)
- Target Sampling (cross-validation of number of distractors)
- NCE (cross-validation of noise ratio)
- Self-normalization (cross-validation of regularization strenth)

Data:

- PTB (1M tokens, 10k vocab)
- Gigaword  (5B tokens, 100k vocab)
- billionW (800M tokens, 800k vocab)


#### Key Takeaways

- Techniques that work best for small vocabluaries are not necessarily the ones that work best for large vocabularies.
- Differentiated softmax varies the capacity (size of matrix slice in the last layer) based on token frequency. In practice, it's implemented as separate matrices with different sizes.
- Perplexity doesn't seem to improve much after ~500M tokens
- Models are trained for 1 week each
- The competitiveness of softmax diminishes with vocabulary sizes. It seems to perform relatively well on 10k and 100k, but poorly on 800k since it need more processing time per example.
- Traning time, not training data, is the main factor of limiting performance. The authors found that very large models are still making progress after one week and may eventually beat if the other models if allowed to run longer.


#### Questions / Notes

- What about the hyperparameters for Differentiated Softmax? The paper doesn't show an analysis. Also, the fact that this method introduces two additional hyperparameters makes it harder to apply in practice.
- Would've liked to see more comparisons for Softmax, which is the simplest technique of all and doesn't need hyperparameter tuning. It doesn't work well on 800k vocab, but it does for 100k. So, the authors only show how it breaks down for one dataset.