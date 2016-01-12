## [Attention with Intention for a Neural Network Conversation Model](http://arxiv.org/abs/1510.08565)

TLDR; The authors propose an Attention with Intention (AWI) model for Conversation Modeling. AWI consists of three recurrent networks: An encoder that embeds the source sentence from the user, an intention network that models the intention of the conversation over time, and a decoder that generates responses. The authors show that the network can general natural responses.

#### Key Points

- Intuition: Intention changes over the course of a conversation, e.g. communicate problem -> resolve issue -> acknowledge.
- Encoder RNN: Depends on last state of the decoder. Reads the input sequence and converts it into a fixed-length vector.
- Intention RNN: Gets encoder representation, previous intention state, and previous decoder state as input and generates new representation of the intention.
- Decoder RNN: Gets current intention state and attention vector over the encoder as an input. Generates a new output.
- Architecture is evaluated on an internal helpdesk chat dataset with 10k dialogs, 100k turns and 2M tokens. Perplexity scores and a sample conversation are reported.

#### Notes/Questions

- It's a pretty short paper and not sure what to make of the results. The PPL scores were not compared to alternative implementations and no other evaluations (e.g. crowdsourced as in Neural Conversational Model) are done.