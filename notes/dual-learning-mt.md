## Dual Learning for Machine Translation

TLDR; The authors finetune an FR -> EN NMT model using a RL-based dual game. 1. Pick a French sentence from a monolingual corpus and translate it to EN. 2. Use an EN language model to get a reward for the translation 3. Translate the translation back into FR using an EN -> FR system. 4. Get a reward based on the consistency between original and reconstructed sentence. Training this architecture using Policy Gradient authors can make efficient use of monolingual data and show that a system trained on only 10% of parallel data and finetuned with monolingual data achieves comparable BLUE scores as a system trained on the full set of parallel data.


### Key Points

- Making efficient use of monolingual data to improve NMT systems is a challenge
- Two Agent communication game: Agent A only knows language A and agent B only knows language B. A send message through a noisy translation channel, B receives message, checks its correctness, and sends it back through another noisy translation channel. A checks if it is consistent with the original message. Translation channels are then improves based on the feedback.
- Pieces required: LanguageModel(A), LanguageModel(B), TranslationModel(A->B), TranslationModel(B->A). Monolingual Data.
- Total reward is linear combination of: `r1 = LM(translated_message)`, `r2 = log(P(original_message | translated_message)`
- Samples are based on beam search using the average value as the gradient approximation
- EN -> FR pretrained on 100% of parallel data: 29.92 to 32.06 BLEU
- EN -> FR pretrained on 10% of parallel data: 25.73 to 28.73 BLEU
- FR -> EN pretrained on 100% of parallel data: 27.49 to 29.78 BLEU
- FR -> EN pretrained on 10% of parallel data: 22.27 to 27.50 BLEU


### Some Notes

- I think the idea is very interesting and we'll see a lot related work coming out of this. It would be even more amazing if the architecture was trained from scratch using monolingual data only. Due the the high variance of RL methods this is probably quite hard to do though.
- I think the key issue is that the rewards are  quite noisy, as is the case with MT in general. Neither the language model nor the BLEU scores gives good feedback for the "correctness" of a translation.
- I wonder why there is such a huge jump in BLEU scores for FR->EN on 10% of data, but not for EN->FR on the same amount of data.