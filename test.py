from magent2.environments import battle_v4
from models.torch_model import QNetwork
from models.functional_model import FunctionalPolicyAgent
import torch
import numpy as np
import os
import cv2
from models.ppo_model import PPOAgentWithLightning
from models.functional_model import FunctionalPolicyAgent
from models.final_torch_model import QNetwork as FinalQNetwork
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
        torch.load("parameters/red.pt", weights_only=True, map_location="cpu")
    )
    q_network.to(device)

    final_q_network = FinalQNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    final_q_network.load_state_dict(
        torch.load("parameters/red_final.pt", weights_only=True, map_location="cpu")
    )
    final_q_network.to(device)
    final_q_network.eval()
    action_space_size = 21  

    f_agent = FunctionalPolicyAgent(action_space_size, embed_dim=5, height=13, width=13)
    f_agent.load_state_dict(
        torch.load("parameters/final_blue_agent.pth", weights_only=True, map_location="cpu")
    )
    f_agent.to(device)
    f_agent.eval()
    def final_pretrain_policy(env, agent, obs):
        observation = (
            torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            q_values = final_q_network(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]

    def functional_policy(env, agent, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action = f_agent.select_action(obs_tensor)
        return action
    
    def pretrain_policy(env, agent, obs):
        observation = (
            torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            q_values = q_network(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]


    def run_eval(env, red_policy, blue_policy, n_episode: int = 100):
        red_win, blue_win = [], []
        red_tot_rw, blue_tot_rw = [], []
        n_agent_each_team = len(env.env.action_spaces) // 2
        fps = 25
        cnt = 0
        for _ in tqdm(range(n_episode)):
            env.reset()
            n_kill = {"red": 0, "blue": 0}
            red_reward, blue_reward = 0, 0
            frames = []
            idx = 0
            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                agent_team = agent.split("_")[0]
                # print(agent)
                n_kill[agent_team] += (
                    reward > 4.5
                )  # This assumes default reward settups
                if agent_team == "red":
                    red_reward += reward
                else:
                    blue_reward += reward

                if termination or truncation:
                    action = None  # this agent has died
                else:
                    if agent_team == "red":
                        action = red_policy(env, agent, observation)
                    else:
                        action = blue_policy(env, agent, observation)

                env.step(action)
                if agent == "red_0":
                    frames.append(env.render())
                    idx += 1
                    # print(f"Cycle id: {idx}")
                    # print(f"Red: {n_kill["red"]} vs Blue: {n_kill["blue"]}")
            # print(len(frames))
            cnt += 1
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter(
                os.path.join("video", f"random_{cnt}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            frames = []

            who_wins = "red" if n_kill["red"] >= n_kill["blue"] + 5 else "draw"
            who_wins = "blue" if n_kill["red"] + 5 <= n_kill["blue"] else who_wins
            red_win.append(who_wins == "red")
            blue_win.append(who_wins == "blue")

            red_tot_rw.append(red_reward / n_agent_each_team)
            blue_tot_rw.append(blue_reward / n_agent_each_team)

        return {
            "winrate_red": np.mean(red_win),
            "winrate_blue": np.mean(blue_win),
            "average_rewards_red": np.mean(red_tot_rw),
            "average_rewards_blue": np.mean(blue_tot_rw),
        }


    print("Eval with final policy")
    print(
        run_eval(
            env=env, red_policy=random_policy, blue_policy=functional_policy, n_episode=10
        )
    )
    print("=" * 20)


if __name__ == "__main__":
    eval()
