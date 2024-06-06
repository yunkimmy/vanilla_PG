import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from typing import Tuple
import matplotlib.pyplot as plt
import imageio
from matplotlib.patches import Rectangle


class Policy_gradient:
    def __init__(self, env_name='CartPole-v0', lr=1e-2, batch_size=5000, discount=0.9, epoch=100, h_layer=[32]):
        self.env = gym.make(env_name)
        self.lr = lr
        self.batch_size = batch_size
        self.discount = discount
        self.epoch = epoch
        self.act_dim = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]
        self.h_layer = h_layer
        self.policy = self.build_mlp(h_layer)
        self.optimizer = Adam(self.policy.parameters(), lr=self.lr)

    def build_mlp(self, h_layer, activation=nn.ReLU, output_activation=nn.Identity):
        # Build a feedforward neural network.
        layers = []
        sizes = [self.obs_dim] + h_layer + [self.act_dim] 
        num_layers = len(sizes) - 1
        for i in range(num_layers):
            act = activation
            if i == (num_layers - 1):
                act = output_activation
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(act())
        return nn.Sequential(*layers)
    
    def get_policy(self, obs: torch.tensor) -> torch.distributions.Categorical:
        logits = self.policy(obs)
        return Categorical(logits = logits)

    def get_action(self, obs: torch.tensor) -> int:
        return self.get_policy(obs).sample().item()

    def compute_loss(self, obs: torch.tensor, act: torch.tensor, weights: torch.tensor) -> torch.tensor:
        m = self.get_policy(obs)
        log_prob = m.log_prob(act)
        loss = -(log_prob * weights)
        return loss.mean()
    
    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        batch_obs = []       
        batch_acts = []     
        batch_weights = []    
        batch_rets = []      
        batch_lens = []    

        obs =self.env.reset()
        done = False

        eps_rew = []
        while True:
            act = self.get_action(torch.from_numpy(obs).to(torch.float32))
            batch_obs.append(obs.copy())
            obs, rew, done, _ = self.env.step(act)


            batch_acts.append(act)
            eps_rew.append(rew)

            if done:
                eps_ret = sum(eps_rew)
                eps_len = len(eps_rew)
                eps_weights = self.get_weights(eps_rew)
                batch_weights.extend(eps_weights)
                batch_rets.append(eps_ret)
                batch_lens.append(eps_len)
                obs = self.env.reset()
                done = False
                eps_rew = []
            
                if len(batch_obs) > self.batch_size:
                    break
        
        return np.array(batch_obs), np.array(batch_acts), np.array(batch_weights), np.array(batch_rets), np.array(batch_lens)

    
    def get_weights(self, rewards: list) -> list:
        R = 0
        weights = []

        for reward in reversed(rewards):
            R = reward + self.discount * R
            weights.append(R)

        weights.reverse()

        return weights
          
    def update(self, obs: np.ndarray, actions: np.ndarray, weights: np.ndarray):
        self.optimizer.zero_grad()
        loss = self.compute_loss(torch.tensor(obs, dtype=torch.float32),
                                  torch.tensor(actions, dtype=torch.int32),
                                    torch.tensor(weights, dtype=torch.float32))
        loss.backward()
        self.optimizer.step()
        
        return loss

    def train(self):
        for epoch in range(self.epoch):
            obs, actions, weights, returns, length = self.sample_batch()
            loss = self.update(obs, actions, weights)
            if (epoch+1) % 10 == 0:
                print("epoch: {}, loss: {:.3f}, return: {:.3f}, episode length: {:.3f}".format(
                    epoch+1, loss.item(), np.mean(returns), np.mean(length)
                ))
    
    def save_frames_as_gif(self, frames, path='./', filename='gym_animation.gif'):
        imageio.mimsave(path + filename, frames, duration=1/30)

    def plot_cartpole(self, ax, obs):
        cart_width, cart_height = 0.4, 0.2
        pole_length = 1.0
        x, x_dot, theta, theta_dot = obs

        ax.clear()
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect('equal')

        # Plot cart
        cart = Rectangle((x - cart_width / 2, -cart_height / 2), cart_width, cart_height, color='blue')
        ax.add_patch(cart)

        # Plot pole
        pole_x = [x, x + pole_length * np.sin(theta)]
        pole_y = [0, pole_length * np.cos(theta)]
        ax.plot(pole_x, pole_y, color='red', linewidth=4)

        # Plot ground
        ax.plot([-2.4, 2.4], [-cart_height / 2, -cart_height / 2], color='black')

    def test_policy(self, num_episodes=1):
        frames = []
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)  # Adjust DPI as needed

        for i in range(num_episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                self.plot_cartpole(ax, obs)
                plt.pause(0.001)
                
                # Capture frame
                fig.canvas.draw()
                width, height = fig.canvas.get_width_height()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                buffer_size = width * height * 3
                frame = frame[:buffer_size].reshape((height, width, 3))
                frames.append(frame)

                action = self.get_action(torch.from_numpy(obs).to(torch.float32))
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
            print(f"Episode {i+1}: Total Reward: {total_reward}")

        self.env.close()
        self.save_frames_as_gif(frames)



if __name__ == '__main__':
    agent = Policy_gradient()
    agent.train()
    agent.test_policy(5)