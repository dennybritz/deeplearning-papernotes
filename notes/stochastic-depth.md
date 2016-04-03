## [Deep Networks with Stochastic Depth](http://arxiv.org/abs/1603.09382)

TLDR; The authors randomly drop out complete layers during training using a modified ResNet architecture. The dropout probability hyperparameter decreases linearly (higher layers have a higher chance to be dropped) ending at 0.5 at the final layer in the experiments. This mechanisms helps vanishing gradients, diminishing feature reuse, and long training time. The model achieves new records on the CIFAR-10, CIFAR-100 and SVHN dataset.


#### Key Points:

- Can easily modify ResNet architecture to dropout out whole layer by only keeping the identity skip connection
- Lower layers get lower probability of being dropped since they intuitively contain more "stable" features. Authors use linear decay with final value 0.5.
- Training time reduces by 25% - 50% depending on dropout probability hyperparameter
- Authors find that vanishing gradients are indeed reduces by plotting the gradient magnitudes vs. number of epochs
- Can be interpreted as an ensemble of networks with varying depth
- All layers are used during test time and need to scale activations appropriately
- Authors successfully train network with 1000+ layers and achieve further error reduction

