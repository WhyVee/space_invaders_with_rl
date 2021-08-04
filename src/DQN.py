import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, MaxPooling1D, MaxPooling2D, Activation, Dropout, Flatten, Conv1D, Conv2D
from tensorflow.keras.optimizers import Adam
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"



class DQNagent():
    def __init__(self):
        self.screen_height = 600
        self.screen_width = 600
        self.env_matrix = np.zeros((self.screen_height,self.screen_width)).reshape(600, 600, 1)

        # action space is: 1) Movement: Left, right, stationary
                        #  2) Shooting: True or False
        self.action_space_size = 6

        # build model
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # model.add(InputLayer(batch_input_shape=self.env_matrix.shape))
        model.add(Conv2D(100, (3, 3), input_shape=self.env_matrix.shape))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(100, (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(self.action_space_size, activation='softmax'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def set_reward(self, player, crash):
        """
        Return the reward.
        The reward is:
            -100 when player gets hit 
            +x when player hits an enemy, each enemy has its own score
        """
        pass
        # self.reward = 0
        # if crash:
        #     self.reward = -10
        #     return self.reward
        # if player.eaten:
        #     self.reward = 10
        # return self.reward

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
        Based on the current state of the environment, choose the action that leads to the best reward
        '''
        prediction = self.model.predict(self.env_matrix)
        return np.argmax(prediction)


    # def get_input(self):
	# 	keys = pygame.key.get_pressed()

	# 	if keys[pygame.K_RIGHT]:
	# 		self.rect.x += self.speed
	# 	elif keys[pygame.K_LEFT]:
	# 		self.rect.x -= self.speed

	# 	if keys[pygame.K_SPACE] and self.ready:
	# 		self.shoot_laser()
	# 		self.ready = False
	# 		self.laser_time = pygame.time.get_ticks()
	# 		self.laser_sound.play()

if __name__ == '__main__':
    dqn = DQNagent()
    action = dqn.agent_action()
    print(action)
