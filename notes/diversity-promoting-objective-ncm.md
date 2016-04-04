## [A Diversity-Promoting Objective Function for Neural Conversation Models](http://arxiv.org/abs/1510.03055)

TLDR; The authors use a Maximum Mutual Information (MMI) objective function to generate conversational responses. They still train their models with maximum likelihood, but use MMI to generate responses during decoding. The idea behind MMI is that it promotes more diversity and penalizes trivial responses. The authors evaluate their method using BLEU scores, human evaluators, and qualitative analysis and find that the proposed metric indeed leads to more diverse responses.

#### Key Points

- In practice, NCM (Neural Conversation Models) often generate trivial responses using high-frequency terms partly due to the likelihood objective function.
- Two models: MMI-antiLM and MMI-bidi depending on the formulation of the MMI objective. These objectives are used during response generation, not during training.
- Use Deep 4-layer LSTM with 1000-dimensional hidden state, 1000-dimensional word embeddings.
- Datasets: Twitter triples with 129M context-message-response triples. OpenSubtitles with 70M spoken lines that are noisy and don't include turn information.
- Authors state that perplexity is not a good metric because their objective is to explicitly steer away from the high probability responses.


#### Notes

- BLEU score seems like a bad metric for this. Shouldn't more diverse responses result in a lower BLEU score?
- Not sure if I like the direction of this. To me it seems wrong to "artificially" promote diversity. Shouldn't diversity come naturally as a function of context and intention?