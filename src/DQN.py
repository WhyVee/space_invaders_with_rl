import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, MaxPooling1D, MaxPooling2D, Activation, Dropout, Flatten, Conv1D, Conv2D
from tensorflow.keras.optimizers import Adam
import pickle
from collections import deque
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class DQNagent():
    def __init__(self):
        self.screen_height = 600
        self.screen_width = 600
        self.env_matrix = np.zeros((self.screen_height,self.screen_width)).reshape(600, 600, 1)
        self.old_matrix = self.env_matrix.copy().reshape(-1, 600, 600, 1)

        # action space is: 1) Movement: Left, right, stationary
                        #  2) Shooting: True or False
        self.action_space_size = 6

        # build model, load and set weights
        # 1
        self.model = self.build_model()
        weights = self.load_weights()
        self.model.set_weights(weights)

        # verify model built, check summary
        print(self.model.summary())
        
        # q learning variables
        self.gamma = 0.95
        self.eps = 0.5
        self.decay_factor = 0.95
        self.r_avg_list = []

        # rewards, temp and total
        self.temp_reward = 0
        self.total_reward = 0
        self.reward_list = []

        # instantiate prediction with 0s, len of action space
        self.prediction = np.zeros(self.action_space_size)

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(10, (9, 9), input_shape=self.env_matrix.shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(4,4)))
        model.add(Dropout(0.2))

        model.add(Conv2D(10, (9,9)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(4,4)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(self.action_space_size, activation='softmax'))  # ACTION_SPACE_SIZE = how many choices (6)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def set_reward(self, reward):
        """
        Update the temp and total rewards.
        The reward is:
            -100 when player gets hit 
            +x when player hits an enemy, each enemy has its own score
        """
        self.temp_reward += reward
        self.total_reward += reward

    def add_to_matrix(self, matrix, sprites):
        '''
        Add location of sprites to matrix
        ---
        Inputs:
        matrix: numpy matrix
        sprites: group of sprites

        Returns:
        matrix: inputs the locations of the sprites into numpy matrix
        '''
        for sprite in sprites:
            x, y, w, h = sprite.rect
            y = self.screen_height - y
            matrix[y:y+h, x:x+w] = 1
        return matrix

    def get_state(self, player, blocks, alien_lasers, extra, aliens):
        '''
        Each sprite has a self.rect attribute which shows the sprites x, y alignment
        pass the location of all active sprites to agent
        Store environment in self.env_matrix
        '''
        # reset an empty matrix the size of the screen
        self.env_matrix = np.zeros((self.screen_height,self.screen_width))
        sprite_group_list = [player, blocks, alien_lasers, extra, aliens]

        # for every sprite group, populate the matrix with the position of every sprite
        for sprite_group in sprite_group_list:
            self.env_matrix = self.add_to_matrix(self.env_matrix, sprite_group)

        # reshape the matrix so that it can be passed to model
        self.env_matrix = self.env_matrix.reshape(-1, 600, 600, 1)

    def agent_action(self):
        '''
        Chance of:
        Early in training this is more likely, take a random action for exploration
        -or-
        Based on the current state of the environment, choose the action that leads to the best reward for exploitation
        '''
        if np.random.random() < self.eps:
            self.action = np.random.randint(0, self.action_space_size)
        else:
            self.prediction = self.model.predict(self.old_matrix)
            self.action = np.argmax(self.prediction)

        # give temporary reward based on action
        # no movement, no shooting
        if self.action == 0:
            self.temp_reward -= 500
        # move right, no shooting
        elif self.action == 1:
            self.temp_reward += 50
        # move left, no shooting    
        elif self.action == 2:
            self.temp_reward += 50
        # move right, shooting
        elif self.action == 3:
            self.temp_reward += 500
        # move left, shooting
        elif self.action == 4:
            self.temp_reward += 500
        # no movement, shooting
        elif self.action == 5:
            self.temp_reward += -500

        return self.action

    def decay(self):
        '''
        Decay epsilon after each game
        '''
        self.eps *= self.decay_factor

    def q_learning(self):
        '''
        Use q_learning to improve the neural net
        Determines the reward that previous action received, uses this to train the model to find best rewards
        '''  
        
        target_q = ((self.total_reward*0.1) + self.temp_reward) + (self.gamma * self.action)
        # target_q = self.temp_reward + self.gamma * self.action
        target_vec = self.model.predict(self.old_matrix)[0]
        target_vec[self.action] = target_q
        self.model.fit(self.old_matrix, target_vec.reshape(-1, self.action_space_size), epochs=1, verbose=0)
        
        self.old_matrix = self.env_matrix.copy().reshape(-1, 600, 600, 1)
        self.temp_reward = 0

    def save_weights(self):
        pickle.dump(self.model.get_weights(), open('weights.pkl', 'wb'))

    def load_weights(self):
        weights = pickle.load(open('weights.pkl', 'rb'))
        return weights

    def reset_rewards(self):
        self.reward_list.append(self.total_reward)
        self.total_reward = 0


if __name__ == '__main__':
    dqn = DQNagent()
    action = dqn.agent_action()
    print(action)
