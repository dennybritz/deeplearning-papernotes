## [Character-Aware Neural Language Models](http://arxiv.org/abs/1508.06615)

TLDR; The authors build an LSTM Neural Language model, but instead of using word embeddings as inputs, they use the per-word outputs of a character-level CNN, plus a highway layer. This architecture results in state of the art performance and significantly fewer parameters. It also seems to work well on languages with rich morphology.


#### Key Points 

- Small Model: 15-dimensional char embeddings, filter sizes 1-6, tanh, 1-layer highway with ReLU, 2-layer LSTM with 300-dimensional cells. 5M Parameters. Hiearchical Softmax.
- Large Model: 15-dimensional char embeddings, filter sizes 1-7, tanh, 2-layer highway with ReLU, 2-layer LSTM with 670-dimensional cells. 19M Parameters. Hiearchical Softmax.
- Can generalize to out of vocabulary words due to character-level representations. Some datasets already had OOV words replaced with a special token, so the results don't reflect this.
- Highway Layers are key to performance. Susbtituting HW with MLP does not work well. Intuition is that HW layer adaptively combines different local features for higher-level representation.
- Nearest neighbors after Highway layer are more smenatic than before highway layer. Suggests compositional nature.
- Surprisingly combinbing word and char embeddings as LSTM input results in worse performance - Characters alone are sufficient?
- Can apply same architecture to NML or Classification tasks. Highway Layers at the output may also help these tasks.


#### Notes / Questions

- Essentially this is a new way to learn word embeddings comprised of lower-level character embeddings. Given this, what about stacking this architecture and learn sentence representations based on these embeddings?
- It is not 100% clear to me why the MLP at the output layer does so much worse. I understand that the highway layer can adaptively combine feature, but what if you combined MLP and plain representations and add dropout? Shouldn't that result in similar perfomance?
- I wonder if the authors experimented with higher-dimensional character embeddings. What is the intuition behind the very low-dimensional (15) embeddings?