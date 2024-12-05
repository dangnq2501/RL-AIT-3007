from magent2.environments import battle_v4
from models.torch_model import QNetwork
from models.functional_model import FunctionalPolicyAgent
import torch
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x  # Fallback: tqdm becomes a no-op


def eval():
    max_cycles = 300
    env = battle_v4.env(map_size=45, max_cycles=max_cycles)
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
    f_agent = FunctionalPolicyAgent(action_space_size, input_dim)
    f_agent.q_network.load_state_dict(
        torch.load("trained_agent.pth", weights_only=True, map_location="cpu")
    )
    f_agent.to(device)
    f_agent.eval()
    
    def functional_policy(env, agent, obs):
        observation_tensor = torch.tensor(obs, dtype=torch.float32).flatten()
        
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
        red_win, blue_win = [], []
        red_tot_rw, blue_tot_rw = [], []
        n_agent_each_team = len(env.env.action_spaces) // 2

        for _ in tqdm(range(n_episode)):
            env.reset()
            n_dead = {"red": 0, "blue": 0}
            red_reward, blue_reward = 0, 0
            who_loses = None

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

            red_win.append(who_loses == "blue")
            blue_win.append(who_loses == "red")

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
            env=env, red_policy=random_policy, blue_policy=functional_policy, n_episode=30
        )
    )
    print("=" * 20)

    print("Eval with trained policy")
    print(
        run_eval(
            env=env, red_policy=pretrain_policy, blue_policy=functional_policy, n_episode=30
        )
    )
    print("=" * 20)


if __name__ == "__main__":
    eval()
