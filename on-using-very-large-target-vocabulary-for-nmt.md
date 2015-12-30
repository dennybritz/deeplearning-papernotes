## [On Using Very Large Target Vocabulary for Neural Machine Translation](http://arxiv.org/abs/1412.2007)

TLDR; The authors propose an importance-sampling approach to deal with large vocabularies in NMT models. During training, the corpus is partitioned, and for each partition only target words occurring in that partition are chosen. To improve decoding speed over the full vocabulary, the authors build a dictionary mapping from source sentence to potential target vocabulary. The authors evaluate their approach on standard MT tasks and perform better than the baseline models with smaller vocabulary.

#### Key Points:

- Computing partition function is the bottleneck. Use sampling-based approach.
- Dealing with large vocabulary during training is separate from dealing with large vocab during decoding. Training is handled with importance sampling. Decoding is handled with source-based candidate list.
- Decoding with candidate list takes around 0.12s (0.05) per token on CPU (GPU). Without target list 0.8s (0.25s).
- Issue: Candidate list is depended on source sentence, so it must be re-computed for each sentence. 
- Reshuffling the data set is expensive as new partitions need to be calculated (not necessary, but improved scores).

#### Notes:

- How is the corpus partitioned? What's the effect of the partitioning strategy?
- The authors say that they replace UNK tokens using "another word alignment model" but don't go into detail what this is. The results show that doing this results in much larger score bump than increasing the vocab does. (The authors do this for all comparison models though).
- Reshuffling the dataset also results in a significant performance bump, but this operation is expensive. IMO the authors should take all these into account when reporting performance numbers. A single training update may be a lot faster, but the setup time increases. I'd would've like to see the authors assign a global time budget to train/test and then compare the models based on that.
- The authors only briefly mentioned that re-building the target vocab for each source sentence is an issue and how they solve it, no details given.