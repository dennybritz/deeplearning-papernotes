## [Natural Language Comprehension with the EpiReader](https://arxiv.org/abs/1606.02270)

TLDR; The authors prorpose the "EpiReader" model for Question Answering / Machine Comprehension. The model consists of two modules: An Extractor that selects answer candidates (single words) using a Pointer network, and a Reasoner that rank these candidates by estimating textual entailment. The model is trained end-to-end and works on cloze-style questions. The authors evaluate the model on CBT and CNN datasets where they beat Attention Sum Reader and MemNN architectures.


#### Notes

- In most architectures, the correct answer is among the top5 candidates 95% of the time.
- Soft Attention is a problem in many architectures. Need a way to do hard attention.