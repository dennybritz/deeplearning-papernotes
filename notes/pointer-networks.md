## [Pointer Networks](http://arxiv.org/abs/1506.03134)

TLDR; The authors propose a new architecture called "Pointer Network". A Pointer Network is a seq2seq architecture with attention mechanism where the output vocabulary is the set of input indices. Since the output vocabulary varies based on input sequence length, a Pointer Network can generalize to variable-length inputs. The attention method trough which this is achieved is O(n^2), and only a sight variation of the standard seq2seq attention mechanism. The authors evaluate the architecture on tasks where the outputs correspond to positions of the inputs: Convex Hull, Delaunay Triangulation and Traveling Salesman problems. The architecture performs well these, and generalizes to sequences longer than those found in the training data.


#### Key Points

- Similar to standard attention, but don't blend the encoder states, use the attention vector directory.
- Softmax probabilities of outputs can be interpreted as a fuzzy pointer.
- We can solve the same problem artificially using seq2seq and outputting "coordinates", but that ignores the output constraints and would be less efficient.
- 512 unit LSTM, SGD with LR 1.0, batch size of 128, L2 gradient clipping of 2.0.
- In the case of TSP, the "student" networks outperforms the "teacher" algorithm.


#### Notes/  Questions

- Seems like this architecture could be applied to generating spans (as in the newer "Text Processing From Bytes" paper), for POS tagging for example. That would require outputting classes in addition to input pointers. How?
