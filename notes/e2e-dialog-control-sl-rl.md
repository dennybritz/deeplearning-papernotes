## [End-to-end LSTM-based dialog control optimized with supervised and reinforcement learning](https://arxiv.org/abs/1606.01269)]

TLDR; The author present and end-2-end dialog system that consists of an LSTM, action templates, an entity extraction system, and custom code for declaring business rules. They test the systme on a toy task where the goal is to call a person from an address book. They train the system on 21 dialogs using Supervised Learning, and then optimize it using Reinforcement Learning, achieving 70% task completion rates.

#### Key Points

- Task: User asks to call person. Action: Find in address book and place call
- 21 example dialogs
- Several hundred lines of Python code to block certain actions
- External entity recognition API
- Hand-crafted features as input to the LSTM. Hand-crafted action template.
- RNN maps from sequence to action template, First pre-train LSTM to reproduce dialogs using Supervised Learning, then train using RL / policy gradient
- The system doesn't generate text, it picks a template


#### Notes

- I wonder how well the system would generalize to a task that has a larger action space and more varied conversations. The 21 provided dialogs cover a lot of the taks space already. Much harder to do that in larger spaces.
- I wouldn't call this approach end-to-end ;)
