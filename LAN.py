import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from pettingzoo.sisl import multiwalker_v9
import itertools
import os
import psutil
import uuid
from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # optimization
    gamma: float = 0.99
    lr_high: float = 5e-4
    lr_low: float = 5e-5
    batch_size: int = 32
    updates_per_episode: int = 2
    target_update_freq: int = 200

    rnn_hidden_dim: int = 64
    replay_buffer_size: int = 2500

    num_episodes: int = 10_000
    eval_interval: int = 50
    eval_episodes: int = 10

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 0

    seed: int = 42
    save_dir: str = "./results"

    def __post_init__(self):
        # decay epsilon over first 1/3 of training
        self.epsilon_decay_steps = int(self.num_episodes / 3)


class LocalAdvantageNetwork(nn.Module):
    def __init__(self, obs_dim, n_agents, action_dim, rnn_hidden_dim):
        super().__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(obs_dim + n_agents + action_dim, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, action_dim)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, obs, last_action, agent_ids, hidden_state):
        x_in = torch.cat([obs, last_action, agent_ids], dim=-1)
        x = F.relu(self.fc1(x_in))

        # rnn expects (batch, hidden)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h_out = self.rnn(x, h_in)

        advantages = self.fc2(h_out)
        return advantages, h_out


class CentralizedValueNetwork(nn.Module):
    def __init__(self, global_state_dim, n_agents, rnn_hidden_dim, obs_dim, action_dim, embedding_dim=64):
        super().__init__()
        self.n_agents = n_agents
        self.rnn_hidden_dim = rnn_hidden_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # input is hidden state + obs + action + id per agent
        embedding_input_dim = self.rnn_hidden_dim + self.obs_dim + self.action_dim + self.n_agents
        self.agent_hidden_embedding = nn.Linear(embedding_input_dim, embedding_dim)

        self.value_network = nn.Sequential(
            nn.Linear(global_state_dim + embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, global_state, agent_hidden_states, obs, last_actions_onehot, agent_ids_onehot):
        combined_agent_inputs = torch.cat([
            agent_hidden_states,
            obs,
            last_actions_onehot,
            agent_ids_onehot
        ], dim=-1)

        # embed agent specifics then sum to get a fixed size representation
        agent_embeddings = F.relu(self.agent_hidden_embedding(combined_agent_inputs))

        agent_embeddings_reshaped = agent_embeddings.view(-1, self.n_agents, agent_embeddings.shape[-1])
        summed_embedding = torch.sum(agent_embeddings_reshaped, dim=1)

        # concat global state with the aggregated agent info
        state_history_embedding = torch.cat([global_state, summed_embedding], dim=-1)
        return self.value_network(state_history_embedding)


class EpisodeReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store_episode(self, episode_data):
        self.buffer.append(episode_data)

    def sample(self, batch_size):
        sampled_episodes = random.sample(self.buffer, batch_size)

        # dynamic padding based on max length in batch
        max_len = max(len(ep) for ep in sampled_episodes)

        batch_global_state, batch_obs, batch_actions = [], [], []
        batch_reward, batch_done, batch_last_action = [], [], []

        for ep in sampled_episodes:
            ep_len = len(ep)
            pad_len = max_len - ep_len

            states, obs, acts, rews, dones = zip(*ep)

            # shift actions for last_action input (pre-pad with zeros)
            last_acts = [[0] * len(acts[0])] + list(acts[:-1])

            # padding
            batch_global_state.append(list(states) + [states[-1]] * pad_len)
            batch_obs.append(np.array(list(obs) + [obs[-1]] * pad_len))
            batch_actions.append(list(acts) + [[0] * len(acts[0])] * pad_len)
            batch_reward.append(list(rews) + [0.0] * pad_len)
            batch_done.append(list(dones) + [True] * pad_len)
            batch_last_action.append(last_acts + [[0] * len(acts[0])] * pad_len)

        # create mask for valid steps
        mask = torch.ones((batch_size, max_len))
        for i, ep in enumerate(sampled_episodes):
            mask[i, len(ep):] = 0

        return (
            torch.tensor(np.array(batch_global_state), dtype=torch.float32),
            torch.tensor(np.array(batch_obs), dtype=torch.float32),
            torch.tensor(np.array(batch_actions), dtype=torch.long),
            torch.tensor(np.array(batch_reward), dtype=torch.float32),
            torch.tensor(np.array(batch_done), dtype=torch.bool),
            torch.tensor(np.array(batch_last_action), dtype=torch.long),
            mask
        )

    def __len__(self):
        return len(self.buffer)


class LAN_Controller:
    def __init__(self, obs_dims, action_dims, n_agents, global_state_dim, cfg: Config):
        self.n_agents = n_agents
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.cfg = cfg
        self.gradient_steps = 0

        self.agent_ids = torch.eye(self.n_agents)

        # setup networks
        self.local_advantage_net = LocalAdvantageNetwork(obs_dims[0], n_agents, action_dims[0], cfg.rnn_hidden_dim)
        self.centralized_value_net = CentralizedValueNetwork(global_state_dim, n_agents, cfg.rnn_hidden_dim,
                                                             obs_dims[0], action_dims[0])

        self.target_local_advantage_net = LocalAdvantageNetwork(obs_dims[0], n_agents, action_dims[0],
                                                                cfg.rnn_hidden_dim)
        self.target_centralized_value_net = CentralizedValueNetwork(global_state_dim, n_agents, cfg.rnn_hidden_dim,
                                                                    obs_dims[0], action_dims[0])

        self.params = list(self.local_advantage_net.parameters()) + list(self.centralized_value_net.parameters())
        self.optimizer = optim.Adam(self.params, lr=cfg.lr_high)

        self._update_targets(initial=True)

    def _update_targets(self, initial=False):
        if initial or (self.gradient_steps % self.cfg.target_update_freq == 0):
            self.target_local_advantage_net.load_state_dict(self.local_advantage_net.state_dict())
            self.target_centralized_value_net.load_state_dict(self.centralized_value_net.state_dict())

    def select_actions(self, obs_list, last_actions, last_hidden_states, epsilon):
        actions = []
        hidden_states = []
        with torch.no_grad():
            for i in range(self.n_agents):
                # tensor setup
                obs_tensor = torch.tensor(obs_list[i], dtype=torch.float32).unsqueeze(0)
                h_state_tensor = last_hidden_states[i]
                agent_id_tensor = self.agent_ids[i].unsqueeze(0)

                last_action_tensor = F.one_hot(torch.tensor(last_actions[i]),
                                               num_classes=self.action_dims[i]).float().unsqueeze(0)

                advantages, h_out = self.local_advantage_net(obs_tensor, last_action_tensor, agent_id_tensor,
                                                             h_state_tensor)
                hidden_states.append(h_out)

                # epsilon-greedy
                if np.random.rand() < epsilon:
                    action_idx = np.random.randint(0, self.action_dims[i])
                else:
                    action_idx = advantages.argmax().item()
                actions.append(action_idx)

        return actions, torch.stack(hidden_states).squeeze(1)

    def learn(self, batch):
        self.gradient_steps += 1
        global_state, obs, actions, reward, done, last_action, mask = batch
        batch_size, max_seq_len = global_state.shape[0], global_state.shape[1]

        # init rnn states
        h_state = self.local_advantage_net.init_hidden().expand(batch_size * self.n_agents, -1)
        target_h_state = self.target_local_advantage_net.init_hidden().expand(batch_size * self.n_agents, -1)

        agent_ids_flat = self.agent_ids.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * self.n_agents, -1)

        current_q_proxies_list = []
        target_q_proxies_list = []

        # unroll sequence
        for t in range(max_seq_len):
            obs_t = obs[:, t].reshape(batch_size * self.n_agents, -1)
            actions_t = actions[:, t].reshape(batch_size * self.n_agents, 1)
            last_action_t = last_action[:, t].reshape(batch_size * self.n_agents, -1)
            global_state_t = global_state[:, t]

            # handle last step edge case
            if t < max_seq_len - 1:
                next_obs_t = obs[:, t + 1].reshape(batch_size * self.n_agents, -1)
                next_global_state_t = global_state[:, t + 1]
            else:
                next_obs_t = obs_t
                next_global_state_t = global_state_t

            last_action_onehot_t = F.one_hot(last_action_t.squeeze(1), num_classes=self.action_dims[0]).float()
            actions_onehot_t = F.one_hot(actions_t.squeeze(1), num_classes=self.action_dims[0]).float()

            # forward pass online nets
            advantages, h_state_next = self.local_advantage_net(obs_t, last_action_onehot_t, agent_ids_flat, h_state)
            v = self.centralized_value_net(global_state_t, h_state, obs_t, last_action_onehot_t, agent_ids_flat)

            # Q_tot = V + A (dueling architecture style)
            advantages_taken = torch.gather(advantages.reshape(batch_size * self.n_agents, -1), 1, actions_t).squeeze(1)
            current_q_proxies = v.repeat(1, self.n_agents).reshape(batch_size, self.n_agents) + \
                                advantages_taken.reshape(batch_size, self.n_agents)
            current_q_proxies_list.append(current_q_proxies)

            # target nets
            with torch.no_grad():
                target_adv, target_h_state_next = self.target_local_advantage_net(next_obs_t, actions_onehot_t,
                                                                                  agent_ids_flat, target_h_state)

                # Double Q-Learning: select action with online net, eval with target
                online_next_adv, _ = self.local_advantage_net(next_obs_t, actions_onehot_t, agent_ids_flat, h_state)
                best_next_action_idx = online_next_adv.argmax(dim=1, keepdim=True)

                target_adv_best = torch.gather(target_adv, 1, best_next_action_idx).squeeze(1)
                target_next_v = self.target_centralized_value_net(next_global_state_t, target_h_state, next_obs_t,
                                                                  actions_onehot_t, agent_ids_flat)

                target_q_proxies = target_next_v.repeat(1, self.n_agents).reshape(batch_size, self.n_agents) + \
                                   target_adv_best.reshape(batch_size, self.n_agents)
                target_q_proxies_list.append(target_q_proxies)

            h_state = h_state_next.detach()
            target_h_state = target_h_state_next.detach()

        all_current_q = torch.stack(current_q_proxies_list, dim=1)
        all_target_q = torch.stack(target_q_proxies_list, dim=1)

        # calc loss
        reward_batch = reward.reshape(batch_size, max_seq_len, 1).expand(-1, -1, self.n_agents)
        done_batch = done.reshape(batch_size, max_seq_len, 1).expand(-1, -1, self.n_agents)
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, self.n_agents)

        td_target = reward_batch + self.cfg.gamma * (1 - done_batch.float()) * all_target_q

        loss = (((all_current_q - td_target.detach()) * mask_expanded) ** 2).sum() / mask_expanded.sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10.0)
        self.optimizer.step()

        self._update_targets()


if __name__ == '__main__':
    cfg = Config()

    # repro
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    env = multiwalker_v9.parallel_env(n_walkers=2, render_mode=None, terminate_on_fall=False)
    eval_env = multiwalker_v9.parallel_env(n_walkers=2, render_mode=None, terminate_on_fall=False)
    agents = env.possible_agents
    n_agents = len(agents)

    # discretize continuous action space
    continuous_action_dim = env.action_space(agents[0]).shape[0]
    agent_action_values = [-1.0, 1.0]
    agent_action_space = np.array(list(itertools.product(agent_action_values, repeat=continuous_action_dim)),
                                  dtype=np.float32)

    obs_dims = [env.observation_space(agent).shape[0] for agent in agents]
    action_dims = [len(agent_action_space) for _ in agents]
    global_state_dim = sum(obs_dims)

    controller = LAN_Controller(obs_dims, action_dims, n_agents, global_state_dim, cfg)
    replay_buffer = EpisodeReplayBuffer(cfg.replay_buffer_size)

    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    # use slurm ID if available, else random
    job_id = os.environ.get('SLURM_JOB_ID', str(uuid.uuid4())[:8])

    total_timesteps = 0
    all_episode_rewards_raw = []
    avg_train_rewards, avg_train_episodes = [], []
    all_eval_rewards, all_eval_episodes = [], []

    print(f"Job ID: {job_id} | Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Config: {cfg}")

    for episode in range(cfg.num_episodes):
        observations, _ = env.reset()
        done = False
        current_episode_reward = 0

        hidden_states = controller.local_advantage_net.init_hidden().repeat(n_agents, 1)
        last_actions = np.zeros(n_agents, dtype=int)
        episode_buffer = []

        # lr scheduler
        current_lr = np.interp(episode, [0, cfg.num_episodes], [cfg.lr_high, cfg.lr_low])
        for param_group in controller.optimizer.param_groups:
            param_group['lr'] = current_lr

        # decay
        epsilon = np.interp(episode, [0, cfg.epsilon_decay_steps], [cfg.epsilon_start, cfg.epsilon_end])

        while not done:
            obs_list = [observations[agent].copy() for agent in agents]
            h_state_list = [h.unsqueeze(0) for h in hidden_states]

            action_indices, new_hidden_states = controller.select_actions(obs_list, last_actions, h_state_list, epsilon)

            continuous_actions = [agent_action_space[idx] for idx in action_indices]
            action_dict = {agents[i]: continuous_actions[i] for i in range(n_agents)}
            next_observations, rewards, terminations, truncations, _ = env.step(action_dict)

            # handle dead agents by padding obs with zeros
            next_obs_list = []
            for agent_id in agents:
                if agent_id in next_observations:
                    next_obs_list.append(next_observations[agent_id].copy())
                else:
                    next_obs_list.append(np.zeros(env.observation_space(agent_id).shape, dtype=np.float32))

            global_state = np.concatenate(obs_list)
            team_reward = sum(rewards.values())
            episode_done = any(terminations.values()) or any(truncations.values()) or not next_observations

            episode_buffer.append((global_state, obs_list, action_indices, team_reward, episode_done))

            observations = next_observations
            hidden_states = new_hidden_states.detach()
            last_actions = np.array(action_indices)

            current_episode_reward += team_reward
            total_timesteps += 1
            done = episode_done

        replay_buffer.store_episode(episode_buffer)
        all_episode_rewards_raw.append(current_episode_reward)

        if len(replay_buffer) >= cfg.batch_size:
            for _ in range(cfg.updates_per_episode):
                batch = replay_buffer.sample(cfg.batch_size)
                controller.learn(batch)

        # eval loop
        if (episode + 1) % cfg.eval_interval == 0:
            avg_train = np.mean(all_episode_rewards_raw[-cfg.eval_interval:])
            avg_train_rewards.append(avg_train)
            avg_train_episodes.append(episode + 1)

            eval_scores = []
            for _ in range(cfg.eval_episodes):
                e_obs, _ = eval_env.reset()
                e_done = False
                e_reward = 0
                e_hidden = controller.local_advantage_net.init_hidden().repeat(n_agents, 1)
                e_last_acts = np.zeros(n_agents, dtype=int)

                while not e_done:
                    e_obs_list = [e_obs.get(a, np.zeros(env.observation_space(a).shape)) for a in agents]
                    h_list = [h.unsqueeze(0) for h in e_hidden]

                    e_acts_idx, e_hidden_new = controller.select_actions(e_obs_list, e_last_acts, h_list, epsilon=0)
                    e_hidden = e_hidden_new.detach()
                    e_last_acts = np.array(e_acts_idx)

                    cont_acts = {agents[i]: agent_action_space[e_acts_idx[i]] for i in range(n_agents)}
                    e_next_obs, e_rews, e_terms, e_truncs, _ = eval_env.step(cont_acts)
                    e_done = any(e_terms.values()) or any(e_truncs.values()) or not e_next_obs
                    e_obs = e_next_obs
                    e_reward += sum(e_rews.values())
                eval_scores.append(e_reward)

            avg_eval = np.mean(eval_scores)
            all_eval_rewards.append(avg_eval)
            all_eval_episodes.append(episode + 1)

            mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
            print(
                f"Ep {episode + 1} | Train: {avg_train:.2f} | Eval: {avg_eval:.2f} | Eps: {epsilon:.3f} | RAM: {mem_mb:.1f}MB",
                flush=True)

    filename = os.path.join(cfg.save_dir, f"MW_LAN_ep{cfg.num_episodes}_{job_id}.npz")
    np.savez(filename,
             train_episodes=np.array(avg_train_episodes),
             train_rewards=np.array(avg_train_rewards),
             eval_episodes=np.array(all_eval_episodes),
             eval_rewards=np.array(all_eval_rewards),
             config=asdict(cfg))

    print(f"Saved to {filename}")
    env.close()
    eval_env.close()