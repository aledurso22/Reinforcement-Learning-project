#libraries
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
#classes
from actor import Actor
from critic import Critic

prova = np.linspace(1,100)

plt.plot(prova)
plt.xlabel("steps")
plt.ylabel("Loss")
plt.title("Actor loss")
plt.savefig('figures/prova.png')
plt.close()


def evaluate_performance(actor, critic, total_steps):
    env = gym.make("CartPole-v1")
    episodic_rewards_10 = []
    v_values = []
    for j in range(10):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            #get the current value function fro one trajectory (j==0)
            if j ==0 : v_values.append( float(critic(torch.from_numpy(state).float()).detach().data.numpy()) ) #shitty with float
            #select the greedy action
            probs = actor(torch.from_numpy(state).float())            
            action = torch.argmax(probs) #greedy policy
            #go to the next state
            next_state, reward, terminated, truncated, _  = env.step(action.detach().data.numpy())
            total_reward += reward
            state = next_state
            done = terminated or truncated
        episodic_rewards_10.append(total_reward)
    print("episodic return of the 20000:", np.mean(episodic_rewards_10))
    plt.plot(v_values)
    plt.title("V values")
    plt.savefig(f'figures/v_values_{total_steps}.png')
    plt.close()




#environment
env = gym.make("CartPole-v1")


#conf
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
torch.manual_seed(42)
actor = Actor(state_dim, n_actions)
critic = Critic(state_dim)
adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-5)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
gamma = 0.99

episode_rewards = []
actor_losses = []
critic_losses = []
total_steps = 0
max_steps = 500000
while total_steps < max_steps:  
    
    done = False
    total_reward = 0
    iteration = 0
    I = 1
    state, _ = env.reset()    
    
    while not done:
        probs = actor(torch.from_numpy(state).float())
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        
        next_state, reward, terminated, truncated, _  = env.step(action.detach().data.numpy())
        
        delta = reward + gamma * (1 - terminated) * critic(torch.from_numpy(next_state).float())
        advantage = delta - critic(torch.from_numpy(state).float())
        
        
        total_reward += reward
        state = next_state

        critic_loss = advantage.pow(2).mean()
        adam_critic.zero_grad()
        critic_loss.backward()
        adam_critic.step()

        actor_loss = - dist.log_prob(action)*advantage.detach()
        adam_actor.zero_grad()
        actor_loss.backward()
        adam_actor.step()

        actor_losses.append(actor_loss.detach().data.numpy())
        critic_losses.append(critic_loss.detach().data.numpy())
        
        done = terminated or truncated    

        if (total_steps % 1000) == 0:
            print("current episodic return:", np.log(total_reward))
            print("critic loss:", np.log(critic_loss.detach().data.numpy()))
            print("actor loss:", np.log(actor_loss.detach().data.numpy()))
            print("\n")

        if (total_steps % 20000 == 0):
            evaluate_performance(actor, critic, total_steps)

        if total_steps >= max_steps:

            break  

        total_steps += 1
        I *= gamma
        iteration += 1          
    
    episode_rewards.append(total_reward)


#actor loss throughout the training
plt.plot(actor_losses)
plt.xlabel("steps")
plt.ylabel("Loss")
plt.title("Actor loss")
plt.savefig('figures/actor_loss.png')
plt.close()
#critic loss throughout the training
plt.plot(critic_losses)
plt.xlabel("steps")
plt.title("Criticloss")
plt.savefig('figures/critic_loss.png')
plt.close()

##########################LOG######################
#actor loss throughout the training
plt.plot(actor_losses)
plt.xlabel("steps")
plt.ylabel("Loss")
plt.title("Actor loss")
plt.savefig('figures/actor_loss.png')
plt.close()
#critic loss throughout the training
plt.plot(critic_losses)
plt.xlabel("steps")
plt.title("Criticloss")
plt.savefig('figures/critic_loss.png')
plt.close()

                


        


            
    