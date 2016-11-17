## [Third-Person Imitation Learning](http://openreview.net/forum?id=B16dGcqlx)

TLDR; The authors propose a new frame for learning a policy from third-person experience. This is different from standard imitation learning which assumes the same "viewpoint" for teacher and student.

The authors build upon [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476), which uses a GAN framework to discriminate between teacher and student experience and force the student to behave close to the teacher. However, when using third-person experience from a different viewpoint the discriminator would simply learn to discriminate between viewpoints instead of behavior and the framework isn't easily applicable.

The authors' solution is to add a second discriminator to maximize a domain confusion loss based on the same feature representation. The objective is to learn the same (viewpoint-independent) feature representation for both teacher and student experience while also learning to discriminate between teacher and student observations. In other words, the objective is to maximize domain confusion while minimizing class loss. In practice, this is another discriminator term in the GAN objective. The authors also found that they need to feed observations at time t+n (n=4 in expeirments) to signal the direction of movement in the environment.
