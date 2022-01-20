 # Generative Models

In this session, we'll learn about Generative models, particularly Auto Encoders and Generative Adversarial Networks.  Generally, "generative" models refer to models that can do a unique task: look at a dataset, and synthesize a new item that is similar to the items already in the dataset.

The difficulties to this, though, are many.  You don't have "new" images for a dataset to directly compare against, so these networks typically fall under the "Unsupervised Learning" category.  There isn't a definitive answer as to what is a new data point so it is a challenging problem.

We'll learn in this session some introductory techniques for generative models, as well as application examples.

## Further Resources

Generative models are a huge area of active research.  Over the past few years, a lot of impactful papers have come out.  For some further reading, please see:

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) - The first GAN paper.
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) - This is the basis of the GAN we train in this tutorial, though ours is much simpler.
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) - The work that turns pictures of horses into Zebras! ([See also ](https://junyanz.github.io/CycleGAN/))
- [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691) - an in depth article about variation autoencoders.
- [Glow: Generative Flow with Invertible 1×1 Convolutions](https://proceedings.neurips.cc/paper/2018/file/d139db6a236200b21cc7f752979132d0-Paper.pdf) - generative _flow_ models, a newer development that is outperforming GANs in scientific generation applications.


