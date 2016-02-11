## [WebNav: A New Large-Scale Task for Natural Language based Sequential Decision Making](http://arxiv.org/abs/1602.02261)

TLDR; The authors propose a web navigation task where an agent must find a target page containing a search query (typically a few sentences) by navigating a web graph with restrictions on memory, path length and number of exlorable nodes. Tey train Feedforward and Recurrent Neural Networks and evaluate their performance against that of human volunteers.


#### Key Points


- Datasets: Wiki-[NUM_ALLOWED_HOPS]: WikiNav-4 (6k train), WikiNav-8 (890k train), WikiNav-16 (12M train). Authors evaluate variosu query lengths for all data sets.
- Vector representation of pages: BoW of pre-trained word2vec embeddings. 
- State-dependent action space: All possible outgoing links on the current page. At each step, the agent can peek at the neighboring nodes and see their full content.
- Training, a single correct path is fed to the agent. Beam search to make predictions.
- NeuAgent-FF uses a single tanh layer. NeuAgent-Rec uses LSTM.
- Human performance typically worse than that of Neural agents


#### Notes/Questions

- Is it reasonable to allow the agents to "peek" at neighboring pages? Humans can make decisions based on the hyperlink context. In practice, peaking at each page may not be feasible if there are many links on the page.
- I'm not sure if I buy the fact that this task requires Natural Language Understanding. Agents are just matching query word vectors against pages, which is no indication of NLU. An indication of NLU would be if the query was posed in a question format, which is typically short. But here, the authors use several sentences as queries and longer queries lead to better results, suggesting that the agents don't actually have any understanding of language. They just match text.
- Authors say that NeuAgent-Rec performed consistently better for high hop length, but I don't see that in the data.
- The training method seems a bit strange to me because the agent is fed only one correct path, but in reality there are a large number of correct paths and target pages. It may be more sensible to train the agent with all possible target pages and paths to answer a query.