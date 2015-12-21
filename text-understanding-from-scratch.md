## [Text Understanding from Scratch (2015-02)](http://arxiv.org/abs/1502.01710)

TLDR; Authors apply 6-layer and 9-layer (+3 affine) convolutional nets to character-level input and evaluate their models on Sentiment Analysis and Categorization tasks using (new) large-scale data sets. The authors don't use pre-trained word-embeddings, or any notion of words, and instead learn directly from character-level input with characters being encoded as one-hot vetors. This means the same model can be applied to any language (provided the vocabulary is small enough). The models presented in this paper beat BoW and word2vec baseline models.

## Data and model performance

Because existing ones were too small the authors collected several new datasets that don't have standard benchmarks. 

- DBpedia Ontology Classification: 560k training, 70k test.
- Amazon Reviews 5-class: 3M train, 650k test
- Amazon Reviews polar: 3.6M train, 400k test
- Yahoo! Answer topics 10-class: 1.4M train, 60k test
- AG news classification 4-class: 120k train, 1.9k test
- Sogou Chinese News 5-class: 450k train, 60k test

Model accuracy for small and large models:

- DBpedia: 98.02 / 98.27
- Amazon 5-class: 59.47 / 58.69
- Amazon 2-class: 94.50 / 94.49
- Yahoo 10-class: 70.16 / 70.45
- AG 4-class: 84.35 / 87.18
- Chinese 5-class: 91.35 / 95.12

#### Key Takeaways

- Pretty Standard CNN architecture applied to characters. Conv, ReLU, Maxppol, fully-connected. Filter sizes of 7 and 3. See paper for parameter details.
- Training takes a long time, presumably due to the size of the data. The authors quote 5 days per epoch on the large Amazon data set and large model.
- Authors can't handle large vocabularies, they romanize Chinese. 
- Authors experiment with randomly replacing words with synonyms, seems to give a small improvements:


#### Notes / Questions

- The authors claim to do "text understanding" and learn representations, but all experiments are on simple classification tasks. There is no evidence that the network actually learns meaningful high-level representations, and doesn't just memorize n-grams for example.
- These data sets are large, and the authors claim that they need large data sets, but there are no experiments in the paper that show this. How does performance vary with data size?
- The comparision with other models is lacking. I would have liked to see some of the other state-of-the-art model being compared, e.g. Kim's CNN. Comparing with BoW doesn't show much. As these models are openly available the comparison should have been easy.
- The romanization of Chinese is an ugly "hack" that goes against what the authors claim: Being language-independent and learning "from scratch".
- It's strange that the authors use a thesaurus as a means for training example augmentation, as a theraus is word-level and language-specific, something that the authors explicitly argue against in this paper. Perhaps could have used word (character-level) dropout instead.
- Are there any hyperparameters that were optimized? Authors don't mention any dev sets.
- Have the datasets been made publicly available? The authors complain that "the unfortunate fact in literature is that there are no large openly accessible datasets", but fail to publish their own.
- I'd expect the confustion matrix for the 5-star Amazon reviews to show mistakes coming from negations, but it doesn't, which suggests that the model really learns meaningful representations (such as negation).


