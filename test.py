from magent2.environments import battle_v4
from models.torch_model import QNetwork
from models.functional_model import FunctionalPolicyAgent
import torch
import numpy as np
import os
import cv2
from models.ppo_model import PPOAgentWithLightning
from models.kaggle_notebook import FunctionalPolicyAgent as BattleAgent
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x  # Fallback: tqdm becomes a no-op


def eval():
    max_cycles = 300
    env = battle_v4.env(map_size=45, max_cycles=max_cycles, render_mode="rgb_array")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def random_policy(env, agent, obs):
        return env.action_space(agent).sample()

    q_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load("red.pt", weights_only=True, map_location="cpu")
    )
    q_network.to(device)
    action_space_size = 21  
    input_dim = 13 * 13 * 5
    f_agent = BattleAgent(action_space_size)
    f_agent.load_state_dict(
        torch.load("blue_trained_agent.pth", weights_only=True, map_location="cpu")
    )
    f_agent.to(device)
    f_agent.eval()

    input_channels = 5
    input_size = 13
    action_space_size = 21
    lr = 3e-4
    gamma = 0.99
    clip_epsilon = 0.2
    replay_buffer = []
    max_buffer_size = 10000
    batch_size = 256
    n_episodes = 40

    ppo_agent = PPOAgentWithLightning.load_from_checkpoint(
        "epoch=39-step=1600.ckpt",
        input_channels=input_channels,
        input_size=input_size,
        action_space_size=action_space_size,
        map_location=torch.device("cpu")
    )    
    ppo_agent.eval()
    def ppo_policy(env, agent, obs):
        observation_tensor = torch.tensor(obs, dtype=torch.float32).squeeze(0)
        
        policy, _ = ppo_agent(observation_tensor)
        action = torch.argmax(policy, dim=-1).item()
        return action
            
    def functional_policy(env, agent, obs):
        observation_tensor = torch.tensor(obs, dtype=torch.float32)
        
        action = f_agent.select_action(observation_tensor, eval_mode=True)
        return action
    
    def pretrain_policy(env, agent, obs):
        observation = (
            torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            q_values = q_network(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]

    def run_eval(env, red_policy, blue_policy, n_episode: int = 100):
        vid_dir = "video"
        os.makedirs(vid_dir, exist_ok=True)
        fps = 35
        
        red_win, blue_win = [], []
        red_tot_rw, blue_tot_rw = [], []
        n_agent_each_team = len(env.env.action_spaces) // 2
        cnt = 0
        
        for _ in tqdm(range(n_episode)):
            env.reset()
            frames = []
            n_dead = {"red": 0, "blue": 0}
            red_reward, blue_reward = 0, 0
            who_loses = None
            blue_agents = [an for an in env.agents if an.startswith("blue")]
            red_agents = [an for an in env.agents if an.startswith("red")]
            agent_alive = {an: True for an in env.agents}

            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                agent_team = agent.split("_")[0]
                if agent_team == "red":
                    red_reward += reward
                else:
                    blue_reward += reward

                if env.unwrapped.frames >= max_cycles and who_loses is None:
                    who_loses = "red" if n_dead["red"] > n_dead["blue"] else "draw"
                    who_loses = "blue" if n_dead["red"] < n_dead["blue"] else who_loses

                if termination:
                    agent_alive[agent] = False

                if termination or truncation:
                    action = None  # this agent has died
                    n_dead[agent_team] = n_dead[agent_team] + 1

                    if (
                        n_dead[agent_team] == n_agent_each_team
                        and who_loses
                        is None  # all agents are terminated at the end of episodes
                    ):
                        who_loses = agent_team
                else:
                    if agent_team == "red":
                        action = red_policy(env, agent, observation)
                    else:
                        action = blue_policy(env, agent, observation)

                env.step(action)
                if agent == "red_0":
                    frames.append(env.render())
            print(len(frames))
            cnt += 1
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter(
                os.path.join(vid_dir, f"test_{cnt}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            frames = []
            blue_alive_count = sum(agent_alive[an] for an in blue_agents)
            red_alive_count = sum(agent_alive[an] for an in red_agents)
            red_win.append(blue_alive_count < red_alive_count)
            blue_win.append(blue_alive_count > red_alive_count)

            red_tot_rw.append(red_reward / n_agent_each_team)
            blue_tot_rw.append(blue_reward / n_agent_each_team)

        return {
            "winrate_red": np.mean(red_win),
            "winrate_blue": np.mean(blue_win),
            "average_rewards_red": np.mean(red_tot_rw),
            "average_rewards_blue": np.mean(blue_tot_rw),
        }

    print("=" * 20)
    print("Eval with random policy")
    print(
        run_eval(
            env=env, red_policy=random_policy, blue_policy=functional_policy, n_episode=10
        )
    )
    print("=" * 20)

    print("Eval with trained policy")
    print(
        run_eval(
            env=env, red_policy=pretrain_policy, blue_policy=functional_policy, n_episode=10
        )
    )
    print("=" * 20)


if __name__ == "__main__":
    eval()
