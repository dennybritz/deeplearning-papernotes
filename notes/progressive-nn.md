## [Progressive Neural Networks](http://arxiv.org/abs/1606.04671)

TLDR; The authors propose Progressive Neural Networks (ProgNN), a new way to do transfer learning without forgetting prior knowledge (as is done in finetuning). ProgNNs train a neural neural on task 1, freeze the parameters, and then train a new network on task 2 while introducing lateral connections and adapter functions from network 1 to network 2. This process can be repeated with further columns (networks). The authors evaluate ProgNNs on 3 RL tasks and find that they outperform finetuning-based approaches.

#### Key Points

- Finetuning is a destructive process that forgets previous knowledge. We don't want that.
- Layer h_k in network 3 gets additional lateral connections from layers h_(k-1) in network 2 and network 1. Parameters of those connections are learned, but network 2 and network 1 are frozen during training of network 3.
- Downside: # of Parameters grows quadratically with the number of tasks. Paper discussed some approaches to address the problem, but not sure how well these work in practice.
- Metric: AUC (Average score per episode during training) as opposed to final score. Transfer score = Relative performance compared with single net baseline. 
- Authors use Average Perturbation Sensitivity (APS) and Average Fisher Sensitivity (AFS) to analyze which features/layers from previous networks are actually used in the newly trained network.
- Experiment 1: Variations of Pong game. Baseline that finetunes only final layer fails to learn. ProgNN beats other  baselines and APS shows re-use of knowledge.
- Experiment 2: Different Atari games. ProgNets result in positive Transfer 8/12 times, negative transfer 2/12 times. Negative transfer may be a result of optimization problems. Finetuning final layers fails again. ProgNN beats other approaches.
- Experiment 3: Labyrinth, 3D Maze. Pretty much same result as other experiments.


#### Notes

- It seems like the assumption is that layer k always wants to transfer knowledge from layer (k-1). But why is that true? Network are trained on different tasks, so the layer representations, or even numbers of layers, may be completely different. And Once you introduce lateral connections from all layers to all other layers the approach no longer scales.
- Old tasks cannot learn from new tasks. Unlike humans.
- Gating or residuals for lateral connection could make sense to allow to network to "easily" re-use previously learned knowledge.
- Why use AUC metric? I also would've liked to see the final score. Maybe there's a good reason for this, but the paper doesn't explain.
- Scary that finetuning the final layer only fails in most experiments. That's a very commonly used approach in non-RL domains.
- Someone should try this on non-RL tasks.
- What happens to training time and optimization difficult as you add more columns? Seems prohibitively expensive.