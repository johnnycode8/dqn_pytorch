<a name="readme-top"></a>

# Implement DQN in PyTorch - Beginner Tutorials

This repository contains an implementation of the DQN algorithm from my Deep Q-Learning, aka Deep Q-Network (DQN), YouTube ([@johnnycode](https://www.youtube.com/@johnnycode)) tutorial series. In this series, we code the DQN algorithm from scratch with Python and PyTorch, and then use it to train the Flappy Bird game. If you find the code and tutorials helpful, please consider supporting me:

<a href='https://www.buymeacoffee.com/johnnycode'><img src="https://cdn.buymeacoffee.com/buttons/default-blue.png" alt="Buy Me A Coffee" height="41" width="174"></a>


If you are brand new to Reinforcement Learning, you may want to start with my Q-Learning tutorials first, then continue on to Deep Q-Learning: https://github.com/johnnycode8/gym_solutions

## 1. Install FlappyBird Gymnasium & Setup Development Environment
We'll set up our development environment on VSCode and Conda, and then install Flappy Bird Gymnasium, PyTorch, and Tensorflow (we'll use Tensorflow's TensorBoard to monitor training progress). There are 2 versions of Flappy Bird, one version provides the position of the last pipe, the next pipe, and the bird and the other version that provides RGB (image) frames. The RGB version requires a Convolutional Neural Network, which is more complicated, while the positional values version can be trained using a regular Neural Network. We'll start with the positional version and maybe tackle the RGB version in the future.

<a href='https://youtu.be/arR7KzlYs4w&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi'><img src='https://img.youtube.com/vi/arR7KzlYs4w/0.jpg' width='400' alt='Install FlappyBird Gymnasium'/></a>

## 2. Implement the Deep Q-Network Module
 A Deep Q-Network is nothing more than a regular Neural Network with fully connected layers. This network is the brain of the bird. What makes this neural network special is that the network's input layer represents the State of the environment and the output layer represents the expected Q-values of the set of Actions. The State is a combination of the position of the last pipe, the next pipe, and the bird. The Action with the highest Q-value is the best Action for a given State.

<a href='https://youtu.be/RVMpm86equc&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi'><img src='https://img.youtube.com/vi/RVMpm86equc/0.jpg' width='400' alt='Implement Deep Q-Network Module'/></a>

## 3. Implement Experience Replay & Load Hyperparameters from YAML
The concept of Experience Replay is to collect a large set of "experiences" so that the DQN can be trained using smaller samples. Experience Replay is essential because we need to show the neural network many examples of similar situations to help it learn general patterns. An "experience" consists of the current state, the action that was taken, the resulting new state, the reward that was received, and a flag to indicate if the new state is terminal. When training, we randomly sample from this memory to ensure diverse training data. We also create a separate hyperparameter file to manage parameters like replay memory size, training batch size, and Epsilon for the Epsilon-Greedy algorithm. This way, we can easily change these parameters for different environments.

<a href='https://youtu.be/y3BSPfmMIkA&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi'><img src='https://img.youtube.com/vi/y3BSPfmMIkA/0.jpg' width='400' alt='Implement Experience Replay'/></a>

## 4. Implement Epsilon-Greedy & Debug the Training Loop
The Epsilon-Greedy algorithm is use for exploration (bird taking random action) and exploitation (bird taking best known action at the moment). We start by initializing the Epsilon value to 1, so initially, the agent will choose 100% random actions. As the training progresses, we'll slowly decay Epsilon, making the agent more likely to select actions based on its learned policy. We'll also convert all necessary inputs to tensors before feeding them into our PyTorch-implemented DQN, so we could use CUDA (GPU) to train the network.

<a href='https://youtu.be/2zwbCPpp3do&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi'><img src='https://img.youtube.com/vi/2zwbCPpp3do/0.jpg' width='400' alt='Implement Epsilon-Greedy'/></a>

## 5. Implement the Target Network
Using the DQN module from earlier, we instantiate a Policy Network and a Target Network. The Target Network starts off identical to the Policy Network. The Policy Network represents the brain of the bird; this is the network that we train. The Target Network is used to estimate target Q-values, which is used to train the Policy Network. While it is possible to use the Policy Network to perform the Q-value estimation, the Policy Network is constantly changing during training, so it is more stable to use a Target Network for estimation. After a number of steps (actions), we sync the two networks by copying the Policy Network's weights and biases into the Target Network.

<a href='https://youtu.be/vYRpJo-KMSw&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi'><img src='https://img.youtube.com/vi/vYRpJo-KMSw/0.jpg' width='400' alt='Implement Target Network'/></a>

## 6. Explain Loss, Backpropagation, and Gradient Descent
In case you are not familar with how Neural Networks learn, this video explains the high-level process. The Loss (using Mean Squared Error function, as an example) measures how far our current policy's Q-values are from our target Q-values. Gradient Descent is used to calculate the slope (gradient) of the loss function, which provides an indication of the direction to adjust the weights and biases to lower loss. Backpropagation is the process of performing Gradient Descent and adjusting the weights and biases in the direction that minimizes the loss.

<a href='https://youtu.be/DEqh8rgkLcw&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi'><img src='https://img.youtube.com/vi/DEqh8rgkLcw/0.jpg' width='400' alt='Explain Loss, Backpropagation, Gradient Descent'/></a>

## 7. Optimize Target Network PyTorch Calculations
In the implementation of the Target Network calculations from earlier, we're looping through a batch of experiences and calculating target Q-values for each one. That code is easy to read and understand, however, it is slow to execute because we're processing each experience one at a time. PyTorch is capable of processing the whole batch at once, which is much more efficient. We'll modify the code to take advantage of PyTorch's computational capabilities.

<a href='https://youtu.be/kaXdV1pk8b4&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi'><img src='https://img.youtube.com/vi/kaXdV1pk8b4/0.jpg' width='400' alt='Optimize Target Network PyTorch Calculations'/></a>

## 8. Test DQN Algorithm on CartPole-v1
Reinforcement Learning is fragile as there are many factors that can cause training to fail. We want to make sure that the DQN code we have is bug free. We can test the DQN code on a simple environment that can give us feedback quickly. The Gymnasium Cart Pole environment is perfect for that. Once we are certain that the code is solid, we can finally train Flappy Bird!

<a href='https://youtu.be/Ejv8yv5-i0M&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi'><img src='https://img.youtube.com/vi/Ejv8yv5-i0M/0.jpg' width='400' alt='est DQN Algorithm on CartPole-v1'/></a>

## 9. Train DQN Algorithm on Flappy Bird!
Finally, we can train our DQN algorithm on Flappy Bird! I'll show the results of a 24-hour training session. The bird can fly past quite a few pipe, however, it did not learn to fly indefinitely, that requires perhaps several days of training. I explain why it takes so long to train using DQN.

<a href='https://youtu.be/P7bnuiTVJS8&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi'><img src='https://img.youtube.com/vi/P7bnuiTVJS8/0.jpg' width='400' alt='Train DQN Algorithm on Flappy Bird'/></a>


## 10. Double DQN Explained and Implemented
Since the introduction of DQN, there has been many enhancements to the algorithm. Double DQN (DDQN) was the first major enhancement. I explain the concept behind Double DQN using Flappy Bird as an example. The main objective of Double DQN is to reduce the time wasted exploring paths that don't lead to a good outcomes. However, it's important to note that DDQN may not always lead to significant performance gains in all environments.

<a href='https://youtu.be/FKOQTdcKkN4&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi'><img src='https://img.youtube.com/vi/FKOQTdcKkN4/0.jpg' width='400' alt='Double DQN Explained and Implemented'/></a>


## 11. Dueling DQN Explained and Implemented
Dueling Architecture or Dueling DQN is another enhancement to the DQN algorithm. The main objective of Dueling DQN is to improve training efficiency by splitting the Q-values into two components: Value and Advantages. I explain the concept behind Dueling DQN using Flappy Bird as an example and also implement the Dueling Architecture changes in the DQN module.

<a href='https://youtu.be/3ILECq5qxSk&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi'><img src='https://img.youtube.com/vi/3ILECq5qxSk/0.jpg' width='400' alt='Dueling DQN Explained and Implemented'/></a>


<p align="right">(<a href="#readme-top">back to top</a>)</p>
