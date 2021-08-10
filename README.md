# Defeat Space Invaders with Reinforcement Learning!

## Using a Deep Q-Network to play a Space Invaders clone made in PyGames

### Summary
Reinforcement learning is the future. Developing reinforcement learning models using video games is a great strategy that reduces costs by learning in an environment that can do no damage. An example of training in a real-life environment would be using a reinforcement learning model with a robot in a warehouse stacking boxes. If the model makes a mistake, boxes or even the warehouse itself could be damaged. Starting small, I would like to create a reinforcement learning model to play a simple game of space invaders. The business use for a model like this would be to implement it in future games; an enemy or ally that does not just run “on tracks” provides more enjoyment, and hopefully will provide a better experience.

To run this, you will need to have Tensorflow and PyGames installed. Fitting the model on a large number of parameters is computationally expensive, and I recommend installing the Tensorflow dependency libraries to run off of a GPU if possible. If not possible, the game will still run but it will be painfully slow. 
From the src folder, run python main.py from the command line. From here, the game will launch and the model will start training in a loop of 50 games. The number of games being ran can be modified in the main.py file.

### Contents
1. Defining the Environment
2. Building the Neural Network
3. Model Training
4. Modifying the Rewards
5. Results and Conclusion
6. Sources

### 1. Reading in and Cleaning Data

<img src="img/worked_foodinsecure.png" width="500" height="500">  

### 6. Sources  
Simple Space Invaders clone made in PyGame
Created by Clear Code Projects
https://github.com/clear-code-projects/Space-invaders
