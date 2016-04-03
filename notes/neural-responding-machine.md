## [Neural Responding Machine for Short-Text Conversation](http://arxiv.org/abs/1503.02364)

TLDR; The author train a three variants of a seq2seq model to generate a response to social media posts taken from Weibo. The first variant, NRM-glo is the standard model without attention mechanism using the last state as the decoder input. The second variant, NRM-loc, uses an attention mechanism. The third variant, NRM-hyb combines both by concatenating local and global state vectors. The authors use human users to evaluate their responses and compare them to retrievel-based and SMT-based systems. The authors find that SRM models generate reasonable responses ~75% of the time.

#### Key Points

- STC: Short-text conversation. Generate only a response to a post. Don't need to keep track of a whole conversation.
- Training data: 200k posts, 4M responses.
- Authors use GRU with 1000 hidden units. 
- Vocabulary: Most frequent 40k words for both input and response.
- Retrieval is done using beam search with beam size 10.
- Hybrid model is difficult to train jointly. The authors train the model individually and then fine-tune the hybrid model.
- Tradeoff with retrieval based methods: Responses are written by a human and don't have grammatical errors, but cannot easily generalize to unseen inputs.