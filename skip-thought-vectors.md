## [Skip-Thought Vectors](http://arxiv.org/abs/1506.06726)

TLDR; The authors apply the skip-thoguth word2vec model to the sentence level, training auto-encoders that predict the previous and next sentences. The resulting general-purpose vector representations are called skip-thought vectors. The authors evaluate the performance of these vectors as features on semantic relatedness and classification tasks, achieving competitive results, but not beating fine-tuned models.

#### Key Points

- Code at https://github.com/ryankiros/skip-thoughts
- Training is done on large book corpus (74M sentences, 1B tokens), takes 2 weeks. 
- Two variations: Bidirectional encoder and unidirectional encoder with 1200 and 2400 units per encoder respectively. GRU cell, Adam optimizer, gradient clipping norm 10.
- Vocabulary can be expanded by learning a mapping from a large word2vec voab to the smaller skip-thought vocab. Could also used sampling/hierarchical softmax during training for larger vocab, or train on characters.

#### Questions/Notes

- Authors clearly state that this is not the goal of the paper, though I'd be curious how more sophisticated (non-linear) classifiers perform with skip-thought vectors. Authors probably tried this but it didn't do well ;)
- The fact that the story generation doesn't seem work well shows that the model has problems learning or understanding long-term dependencies. I wonder if this can be solved by deeper encoders or attention.
