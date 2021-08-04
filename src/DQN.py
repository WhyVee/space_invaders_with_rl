import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, MaxPooling1D, MaxPooling2D, Activation, Dropout, Flatten, Conv1D, Conv2D
from tensorflow.keras.optimizers import Adam
import pickle
from collections import deque
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


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
        self.model = self.build_model()
        # weights = self.load_weights()
        # self.model.set_weights(weights)
        
        # q learning variables
        self.gamma = 0.95
        self.eps = 0.5
        self.decay_factor = 0.999
        self.r_avg_list = []

        # rewards, temp and total
        self.temp_reward = 0
        self.total_reward = 0

        # instantiate prediction with 0s, len of action space
        self.prediction = np.zeros(self.action_space_size)

        # build up envs in x and rewards in y, to fit the model in batches
        self.x = []
        self.y = []


    def build_model(self):
        model = Sequential()

        # model.add(InputLayer(batch_input_shape=self.env_matrix.shape))
        model.add(Conv2D(100, (3, 3), input_shape=self.env_matrix.shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(100, (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
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

        --or--

        take a screenshot? and use image processing
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

        return self.action

    def decay(self):
        '''
        Decay epsilor after each game
        '''
        self.eps *= self.decay_factor

    def q_learning(self):
        '''
        Use q_learning to improve the neural net
        '''  
        
        self.env_matrix, self.temp_reward
        target_q = self.temp_reward + self.gamma * self.action
        self.x.append(self.env_matrix)
        self.y.append(target_q)
        if len(self.x) % 5 == 0:
            self.model.fit(self.x[-5:], np.array(self.y[-5:]), epochs=1, verbose=0)
        self.old_matrix = self.env_matrix.copy().reshape(-1, 600, 600, 1)

    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def save_weights(self):
        weights = self.model.get_weights()
        pickle.dump(weights, open('weights.pkl', 'wb'))

    def load_weights(self):
        weights = pickle.load(open('weights.pkl', 'rb'))
        return weights

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)


if __name__ == '__main__':
    dqn = DQNagent()
    action = dqn.agent_action()
    print(action)
