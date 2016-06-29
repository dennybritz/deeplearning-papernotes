## [SQuAD: 100,000+ Questions for Machine Comprehension of Text](http://arxiv.org/abs/1606.05250)

TLDR; A new dataset of ~100k questions and answers based on ~500 articles from Wikipedia. Both questions and answers were collected using crowdsourcing. Answers are of various types: 20% dates and numbers, 32% proper nouns, 31% noun phrase answers and 16% other phrases. Humans achieve an F1 score of 86%, and the proposed Logistic Regression model gets 51%. It does well on simple answers but struggles with more complex types of reasoning. Tge data set is publicly available at https://stanford-qa.com/.

#### Key Points

- System must select answers from all possible spans in a passage. O(N^2) possibilities for N tokens in passage.
- Answers are ambiguous. Humans achieve 77% on exact match and 86% on F1 (overlap based). Humans would probably achieve close to 100% if the answer phrases were unambiguous.
- Lexicalized and dependency tree path features are most important for the LR model
- Model performs best on dates and numbers, single tokens, and categories with few possible candidates
