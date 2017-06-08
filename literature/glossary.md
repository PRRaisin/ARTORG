# **Glossary and general comments on ML terms**

## general remarks and questions
* Why has there been a sudden incrase in interest in Neural Networks?
    - More powerful computers
    - Bengio et al.: unsupervised pretraining of deep networsk for unsupervised tasks
    - One of the algorithmic changes was the replacement of the MSE with the cross-entropy. When optimizing the Cross-entropy, we effectively perform maximum likelihood. Along with sigmoid or softmax outputs, the cross entropy loss performed much better than the MSE (which had problems with saturation and slow learning). 
* Distill is a publishing journal for ML focusing on understandability a used to
* Statistic gradient descent was discovered by Cauchy in 1847
* The rectified linear units (ReLU's) proposed by Glorot et al. in 2011 are inspired by biological neurons. For some inputs, the neuron is
    1. completely inactive
    2. outputting a signal proprotional to the input
    
    Most of the time, biological neurons are in the regime where they are not active. This is referred to as **sparse activations**.
* >Feed-forward networks are the culmination of the function approximation task which have been going on for a long time (Fourier, Taylor etc.)


## Optimiziation

### Backpropagation

Backpropagation was first described by Rumelhart et al. in 1986. It is an algorithm to compute the gradient of a function with respect to some parameters. While it is not limited to the case of (deep) neural networks, it has been used extensively (or rather: the only gradient algorithm I've read about in papers) to calculate the gradient of the loss function wrt parameteres of the network. Backprop doesn't optimize the net, that's what SGD does _with_ the gradient.

> We can think of backpropagation as a table-filling algorithm that takes advantage of storing intermediate results. This table-filling strategy is sometimes called **dynamic programming**.

The backpropagation relies on the chain rule. In each step of the algorithm, the gradient of the loss function is calculated with respect to the subsequent layer, starting from the output layer. All the former gradients are stored and are used via the chain rule to compute the next layer of gradients. We assume here that all the functions in the network have to be differentiable (or otherwise the algorithm should tell the user). With backprop, every edge is calculated only once, contrary to the case where no gradients are saved and in order to calculate any gradient in the net, all the previous gradients have to be calculated again. This leads to an exponential growth in computing cost. 

In the case of fc MLPs, the search for parents and kids is easy, because each neuron is child of all the previous neurons and parent of every following neuron in the network, hence we can do straightforward matrix multiplication. In case of sparse connections, we could think of applying a mask after each network to only pass the connections we want. 

Before we can do backprop, we need a forward pass through the network where we have to store all the activations (this can get expensive computationally) as well as the loss function itself. What follows next is the backward pass, where the parameter updates can be calculated. What's interesting to note is that all the parameters are updated using the same set of activations from the unupdated network. Hence, when we change the paramters in later layers as well as in previous layers at the same time, this might lead to problems because the parameter update was based on now-obsolete activations and outputs. This is especially problematic in deeper networks I suppose. This is probably one of the reasons for badge normalization.



## RNNs
* Quote from [Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/): 
>If training vanilla NN is optimizing over functions, traning recurrent networks is optimizing over programs
* Even if the data is not sequential per se, you can still formulate the problem as a recurrent one (e.g. with the same input fed into the system multiple times or just once then letting the network run a few steps)
Ladder networks
* **BPTT:** Back propagation through time
* **Softmax/Cross-Entropy** The softmax is given by

    $P_t(a) = \frac{\exp(q_t(a)/\tau)}{\sum_{i=1}^{n}\exp(q_t(i)/\tau)}$ 

    where $\tau$ refers to the _temperature of the softmax_. The temperature is a hyperparameter: For high temperatures, all activations have similar probabilities (high entropy). For low $\tau$, the highest input activation (even if it's only slightly larger than the other input activations) gets a disproportional probability while the other values vanish.
    Thus, it can be said that _**Higher temperatures give more variability in it's estimates but more mistakes (in case of k-classification) while low temperatures gives conservative estimates and only places probability mass on the largest value.**_
* The main problem with RNNs is managing long-term dependencies (see ATML lecture).
* **End-to-end training** Training a network from raw input data (e.g. camera pixels) to the end task of the whole system (e.g. steering of a car) withouth having a **pipeline of networks** where each network completes a separate task and is trained independently (_divide-and-train_). The end-to-end training might be more beneficial because the network can be trained in one go without having to rely on human-defined subtasks. Instead, it can learn what SGD says is a good subtask. 

* **Embedding**


## Densely connected Conv. Nets
**Philiosophy of DenseNets: Feature reuse**

* CNNs were introduced by LeCun in 1998 (5 layers _LeNet5_). VGG in 2015 has 19 layers.
* **stochastic depth**: A training procedure where layers are randomly dropped during training (during inference, the whole network is present). Some ResNets have been using $L=1202$ layers. According to the authors this shows that many of the layers are actually redundant and maybe it's more important to have better connection schemes. Hence, DenseNets use fewer parameters than traditional CNNs and also they're sort of narrow (e.g. 12 feature maps per layer). There's only a small gain in _collective knowledge_ per layer, but old information is also not lost. In addition, the gradient for early layers doesn't vanish because we have some direct connections.

## Batch normalization

BN is a technique to use much higher learning rates and to be less careful about initialization. It also acts as a regularizer. According to Goodfellow:
> One of the most exiting recent developments in optimizing deep learning. It's a method of adaptive reparametrization.

* It was motivated by the difficutly of training very deep models. One of the problems was that the weights all get updated at the same time but under the assumption that all the other weights are constant (partial derivatives)
* **Covariante shift** refers to changes in the distribution of the input data: This often happens in medicine, e.g. when we look at pig eyes for training but are actually interested in human eye data for inference.
* **Internal covariate shift** thus refers to this aforementioned shift happening between the layers of a NN.

## Learning rate optimization

* **AdaGRAD** accumulates the squared gradients since the beginning and scales the learning rate by it's inverse. This means that the learning rate monotonically decays over time. It's important to notice that the learning rate is thus different for **all** parameters. Paramters with small gradients see a slow decay in learning rate, while high gradients quickly vanish due to low learning rates. The drawback of AdaGRAD is that there is the possibility of premature and excessive decrease in the effective learning rate because it accumulates gradients _from the beginning of training_. It's very well suited for convex optimization. 
* **RMSProp** is a modification to AdaGRAD which performs better in the non-convex world which is NNs. Instead of adding up the squared gradients, it forms an exponentially weighted moving average where the hyperparameter $\rho$ in $r = \rho r + (1-\rho)g$ controls how quickly the average forgets.

* **Adam**: Adaptive moments


## Hyperparameters

* **Learning rate** cited by Goodfellow as the most important hyperparamter to opitmize. When the learning rate is too big, there can be no convergence because the solution will constantly jump around in the loss function space and usually can't go to the minimum. If the learning rate is too small however, convergence is very slow. However, one should still be able to see how it goes down and then it's possible to increase the learning rate again.
* **Dropout-rate** is the probability that each neuron is set to 0 during training. The higher the dropout rate, the stronger the regularization (reduces overfitting) but the longer the training takes because each neuron has to be trained often but might be set to zero 1/10 times. This slow learning and the idea that much of the feature maps were redundant inspired the [DenseNets](#Densely connected Conv. Nets)
*  ** Receptive field **


### Optimizing hyperparameters
There are two ways of optimizing hyperparameters: Either grid search or random search Bergstra and Bengio (2012). Random Search can be shown to perform exponentially better than grid search, especially when some of the hyperparameters do not influence model performance significantly. Random search does not have wasted experimental runs. See page 422 in _Goodfellow: Deep Learning_. 

##### Model based hyperparameter optimization
Here, we cast the problem of finding good hyperparameters as an optimization problem, where the cost is the validation set error. However, this problem differs from the usual SGD based approach used to train neural networks because very often, the gradient is missing: In cases like the learning rate it might still be feasible, but thinking about dropout rate or number of layers (an integer), it becomes clear that both memory and computational cost are high (imagine having to save and train multiple deep networks with different amount of layers at each point) or that integer valued hyperparamters have a nondifferentiaable interaction with the validation set error. There are a few approaches but at this point, hyperparameter search should still be done manually by the practitioner without additional software (unless it's random search).


### Attention networks
When writing, most people only look at one specific point of the canvas instead of looking at the whole picutre all the time. This is what generative models have so far done: All pixels are conditioned on a single latent distribution.



## Practical rules

* Don't use MSE but Cross-Entropy
* play around with the learning rate
* 


