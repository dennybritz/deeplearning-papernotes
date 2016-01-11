## [A Neural Conversational Model](http://arxiv.org/abs/1506.05869)

TLDR; The authors train a seq2seq model on conversations, building a chat bot. The first data set is an IT Helpdesk dataset with 33M tokens. The trained model can help solve simple IT problems. The second data set is the OpenSubtitles data with ~1.3B tokens (62M sentences). The resulting model learns simple world knowledge, can generalize to new questions, but lacks a coherent personality.

#### Key Points

- IT Helpdesk: 1-layer LSTM, 1024-dimensional cells, 20k vocabulary. Perplexity of 8.
- OpenSubtitles: 2-layer LSTM, 4096-dimensional cells, 100k vocabulary, 2048 affine layer. Attention did not help.
- OpenSubtitles: Treat two consecutive sentences as coming from different speakers. Noisy dataset.
- Model lacks personality, gives different answers to similar questions (What do you do? What's your job?)
- Feed previous context (whole conversation) into encoder, for IT data only.
- In both data sets, the neural models achieve better perplexity than n-gram models.

#### Notes / Questions

- Authors mention that Attention didn't help in OpenSubtitles. It seems like the encoder/decoder context is very short (just two sentences, not a whole conversation). So perhaps attention doesn't help much here, as it's meant for long-range dependencies (or dealing with little data?)
- Can we somehow encode conversation context in a separate vector, similar to paragraph vectors?
- It seems like we need a principled way to deal with long sequences and context. It doesn't really make sense to treat each sentence tuple in OpenSubtitles as a separate conversation. Distant Supervision based on subtitles timestamps could also be interesting, or combine with multimodal learning.
- How we can learn a "personality vector"? Do we need world knowledge or is it learnable from examples?