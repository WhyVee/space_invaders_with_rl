import numpy as np 

class DQNagent():
    def __init__(self):
        pass

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

    def get_state(self, player, alien_lasers, extra, aliens):
        '''
        Each sprite has a self.rect attribute which shows the sprites x, y alignment
        pass the location of all active sprites to agent

        --or--

        take a screenshot? and use image processing
        '''
        pass

    def agent_action(self):
        '''
        Based on the current state of the environment, choose the action that leads to the best reward
        '''
        return np.random.random()


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
