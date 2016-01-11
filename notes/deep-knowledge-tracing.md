## [Deep Knowledge Tracing](http://arxiv.org/abs/1506.05908)

TLDR; The authors apply an RNN to modeling the students knowledge. The input is an exercise question and answer (correct/incorrect), either as one-hot vectors or embedded. The network then predicts whether or not the student can answer a future question correctly. The authors show that the RNN approach results in significant improvement over previous models, can be used for curriculum optimization, and also discovers the latent structure in exercise concepts.

#### Key Points

- Two encodings tried: One hot, embedded
- RNN/LSTM, 200-dimensional hidden layer, output dropout, NLL. 
- No expert annotation for concepts or question/answers are needed
- Blocking (series of exercises of same type) vs Mixing for curriculum optimization: Blocking seems to perform better
- Lots of cool future direction ideas

#### Question / Notes

- Can we not only predict whether an exercise is answered correctly, but also what the most likely student answer would be? My give insight into confusing concepts.