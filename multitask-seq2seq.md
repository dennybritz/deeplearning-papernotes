## [Multi-task Sequence to Sequence Learning](http://arxiv.org/abs/1511.06114)

TLDR; The authors show that we can improve the performance of a reference task (like translation) by simultaneously training other tasks, like image caption generation or parsing, and vice versa. The authors evaluate 3 MLT (Multi-Task Learning) scenarios: One-to-many, many-to-one and many-to-many. The authors also find that using skip-thought unsupervised training works well for improving translation performance, but sequence autoencoders don't.

#### Key Points

- 4-Layer seq2seq LSTM, 1000-dimensional cells each layer and embedding, batch size 128, dropout 0.2, SGD wit LR 0.7 and decay.
- The authors define a mixing ratio for parameter updates that is defined with respect to a reference tasks. Picking the right mixing ratio is a hyperparameter.
- One-To-Many experiments: Translation (EN -> GER) + Parsing (EN). Improves result for both tasks. Surprising that even a very small amount of parsing updates significantly improves MT result.
- Many-to-One experiments: Captioning + Translation (GER -> EN). Improves result for both tasks (wrt. to reference task)
- Many-to-Many experiments: Translation (EN <-> GER) + Autoencoders or Skip-Thought. Skip-Thought vectors improve the result, but autoencoders make it worse.
- No attention mechanism


#### Questions / Notes

- I think this is very promising work. it may allow us to build general-purpose systems for many tasks, even those that are not strictly seq2seq. We can easily substitute classification.
- How do the authors pick the mixing ratios for the parameter updates, and how sensitive are the results to these ratios? It's a new hyperparameter and I would've liked to see graphs for these. Makes me wonder if they picked "just the right" ratio to make their results look good, or if these architectures are robust.
- The authors found that seq2seq autoencoders don't improve translation, but skip-thought does. In fact, autoencoders made translation performance significantly worse. That's very surprising to me. Is there any intuition behind that? 