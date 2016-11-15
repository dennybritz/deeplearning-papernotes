## [Adversarial Training Methods for Semi-Supervised Text Classification](https://arxiv.org/abs/1605.07725)

TLDR; The authors apply adversarial training on labeld data and virtual adversarial training on unlabeled data to the embeddings in text classification tasks. Their models, which are straightforward LSTM architectures, either match or surpass the current state of the art on several text classification tasks. The authors also show that the embeddings learned using adversarial training tend to be tuned better to the corresponding classification task.

#### Key Points

- In Image classification we can apply adversarial training directly to the inputs. In Text classification the inputs are discrete and we cannot make small perturbations, but we can instead apply adversarial training to embeddings.
- Trick: To prevent the model from making perturbations irrelevant by learning embeddings with large norms: Use normalized embeddings.
- Adversarial Training (on labeled examples)
  - At each step of training, identify the "worst" (in terms of cost) perturbation `r_adv` to the embeddings within a given constant epsilon, which a hyperparameter. Train on that. In practice `r_adv` is estimated using a linear approximation.
  - Add a `L_adv` adversarial loss term to the cost function.
- Virtual Adversarial Training (on unlabeled examples)
  - Minimize the KL divergence between the outputs of the model given the regular and perturbed example as inputs.
  - Add `L_vad` loss to the cost function.
- Common misconception: Adversarial training is equivalent to training on noisy examples, but it actually is a stronger regularizer because it explicitly increases cost.
- Model Architectures:
  - (1) Unidirectional LSTM with prediction made at the last step
  - (2) Bidirectional LSTM with predictions based on concatenated last outputs
- Experiments/Results
  - Pre-Training: For all experiments a 1-layer LSTM language model is pre-trained on all labeled and unlabeled examples and used to initialize the classification LSTM.
  - Baseline Model: Only embedding dropout and pretraining
  - IMDB: raining curves show that adversarial training acts as a good regularizer and prevents overfitting. VAT matches state of the art using a unidirectional LSTM only.
  - IMDB embeddings: Baseline model places "good" close to "bad" in embedding space. Adv. training ensures that small perturbations in embeddings don't change the sentiment classification result so these two words become properly separated.
  - Amazon Reviews and RCV1: Adv. + Vadv. achieve state of the art.
  - Rotten Tomatoes: Adv. + Vadv. achieve state of the art. Because unlabeled data overwhelms labeled data vadv. training results in decrease of performance.
  - DBPedia: Even the baseline outperforms state of the art (better optimizer?), adversarial training improves on that.

### Thoughts

- I think this is a very well-written paper with impressive results. The only thing that's lacking is a bit of consistency. Sometimes pure virtual adversarial training wins, and sometimes adversarial + virtual adversarial wins. Sometimes bi-LSTMs make things worse, sometimes better. What is the story behind that? Do we really need to try all combinations to figure out what works for a given dataset?
- Not a big deal, but a few bi-LSTM experiments seem to be missing. This just always makes me if they are "missing for a reason" or not ;)
- There are quite a few differences in hyperparameters and batch sizes between datasets. I wonder why. Is this to stay consistent with the models they compare to? Were these parameters optimized on a validation set (the authors say only dropout and epsilon were optimized)?
- If Adversarial Training is a stronger regularizer than random permutations I wonder if we still need dropout in the embeddings. Shouldn't adversarial training take care of that?