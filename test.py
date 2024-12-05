from magent2.environments import battle_v4
import numpy as np
import matplotlib.pyplot as plt
env = battle_v4.env(
    map_size=45,           
    minimap_mode=False, 
    render_mode="rgb_array" 
)
env.reset()

# print("Agents:", env.agents)

for agent in env.agent_iter():
    
    observation, reward, termination, truncation, info = env.last()
    # observation = env.observe(agent)
    # observation = np.transpose(observation, (2, 0, 1))
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    env.step(action)
    

game_image = env.render()
plt.imshow(game_image)
plt.axis('off')
plt.show()