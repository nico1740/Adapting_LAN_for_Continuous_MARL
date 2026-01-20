import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from pettingzoo.sisl import multiwalker_v9
import os
import psutil
import uuid
import sys
import itertools
from dataclasses import dataclass, asdict


@dataclass
class Config:
    gamma: float = 0.99
    lr_high: float = 5e-4
    lr_low: float = 5e-5

    # MovingQ Specifics (Crawler Population)
    num_crawlers: int = 50
    crawler_freeze_steps: int = 50
    top_k: int = 5
    starvation_threshold: int = 20
    clashing_threshold: float = 0.05
    maximin_candidates: int = 20
    lr_crawlers_high: float = 1e-4
    lr_crawlers_low: float = 1e-5

    rnn_hidden_dim: int = 64
    replay_buffer_size: int = 2500
    batch_size: int = 32
    updates_per_episode: int = 2
    target_update_frequency: int = 200

    num_episodes: int = 50_000
    eval_interval: int = 50
    eval_episodes: int = 10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 0
    zeta_decay_steps: int = 0

    seed: int = 42
    save_dir: str = "./results"

    def __post_init__(self):
        # Decay epsilon over first 1/3, Zeta (crawler LR) over first 2/3
        self.epsilon_decay_steps = int(self.num_episodes * (1 / 3))
        self.zeta_decay_steps = int(self.num_episodes * (2 / 3))


class ActionCrawlers(nn.Module):

    def __init__(self, num_crawlers, action_dim):
        super(ActionCrawlers, self).__init__()
        # Initialize uniformly in [-1, 1] before tanh
        self.actions = nn.Parameter(torch.rand(num_crawlers, action_dim) * 2.0 - 1.0)

    def forward(self):
        return torch.tanh(self.actions)


class LocalAdvantageNetwork(nn.Module):
    def __init__(self, obs_dim, n_agents, action_dim, rnn_hidden_dim):
        super(LocalAdvantageNetwork, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc_embedding = nn.Linear(obs_dim + n_agents + action_dim, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)

        self.advantage_head = nn.Sequential(
            nn.Linear(rnn_hidden_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def init_hidden(self):
        return self.fc_embedding.weight.new(1, self.rnn_hidden_dim).zero_()

    def get_hidden_state(self, obs, agent_ids, last_action, hidden_state):
        x_in = torch.cat([obs, agent_ids, last_action], dim=-1)
        x = F.relu(self.fc_embedding(x_in))
        h_out = self.rnn(x, hidden_state.reshape(-1, self.rnn_hidden_dim))
        return h_out

    def get_advantage(self, hidden_state, action):
        cat_input = torch.cat([hidden_state, action], dim=-1)
        return self.advantage_head(cat_input)


class CentralizedValueNetwork(nn.Module):
    def __init__(self, global_state_dim, n_agents, rnn_hidden_dim, obs_dim, action_dim, embedding_dim=64):
        super(CentralizedValueNetwork, self).__init__()
        self.n_agents = n_agents
        self.agent_hidden_embedding = nn.Linear(rnn_hidden_dim + obs_dim + action_dim + n_agents, embedding_dim)

        self.value_network = nn.Sequential(
            nn.Linear(global_state_dim + embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, global_state, agent_hidden_states, obs, last_actions, agent_ids, active_masks):
        combined = torch.cat([agent_hidden_states, obs, last_actions, agent_ids], dim=-1)
        embeddings = F.relu(self.agent_hidden_embedding(combined))
        embeddings = embeddings * active_masks.unsqueeze(-1)
        summed = torch.sum(embeddings.view(-1, self.n_agents, embeddings.shape[-1]), dim=1)
        return self.value_network(torch.cat([global_state, summed], dim=-1))


class EpisodeReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store_episode(self, episode_data):
        self.buffer.append(episode_data)

    def sample(self, batch_size):
        sampled = random.sample(self.buffer, batch_size)
        max_len = max(ep[0].shape[0] for ep in sampled)

        b_gs, b_obs, b_act, b_rew, b_don, b_last, b_alive, b_mask = [], [], [], [], [], [], [], []

        for ep in sampled:
            gs, obs, acts, rew, don, last, alive = ep
            curr = gs.shape[0]
            pad = max_len - curr

            b_gs.append(np.pad(gs, ((0, pad), (0, 0)), mode='edge'))
            b_obs.append(np.pad(obs, ((0, pad), (0, 0), (0, 0)), mode='edge'))
            b_act.append(np.pad(acts, ((0, pad), (0, 0), (0, 0)), mode='constant'))
            b_rew.append(np.pad(rew, (0, pad), mode='constant'))
            b_don.append(np.pad(don, (0, pad), mode='constant', constant_values=True))
            b_last.append(np.pad(last, ((0, pad), (0, 0), (0, 0)), mode='constant'))
            b_alive.append(np.pad(alive, ((0, pad), (0, 0)), mode='constant'))

            mask = np.zeros(max_len, dtype=np.float32)
            mask[:curr] = 1.0
            b_mask.append(mask)

        return [torch.tensor(np.array(x), dtype=torch.float32 if i != 4 else torch.bool)
                for i, x in enumerate([b_gs, b_obs, b_act, b_rew, b_don, b_last, b_alive, b_mask])]

    def __len__(self):
        return len(self.buffer)


class MovingQLAN_Controller:
    def __init__(self, obs_dims, action_dim, n_agents, global_state_dim, cfg: Config):
        self.n_agents, self.obs_dims, self.action_dim = n_agents, obs_dims, action_dim
        self.cfg = cfg
        self.gradient_steps = 0
        self.agent_ids = torch.eye(n_agents)

        self.num_crawlers = cfg.num_crawlers

        # Crawler Population (Learnable Parameters)
        self.crawlers = nn.ModuleList(
            [ActionCrawlers(num_crawlers=cfg.num_crawlers, action_dim=action_dim) for _ in range(n_agents)]
        )
        self.target_crawlers = nn.ModuleList(
            [ActionCrawlers(num_crawlers=cfg.num_crawlers, action_dim=action_dim) for _ in range(n_agents)]
        )

        # Population Statistics
        self.crawler_usage = torch.zeros((n_agents, cfg.num_crawlers), dtype=torch.long)
        self.freeze_counters = torch.zeros((n_agents, cfg.num_crawlers), dtype=torch.long)
        self.steps_since_win = torch.zeros((n_agents, cfg.num_crawlers), dtype=torch.long)

        self.reset_counter = 0
        self.pruning_steps = 0
        self.clashing_threshold = cfg.clashing_threshold

        # Networks (Double Critic + Central Value)
        self.local_adv_1 = LocalAdvantageNetwork(obs_dims[0], n_agents, action_dim, cfg.rnn_hidden_dim)
        self.local_adv_2 = LocalAdvantageNetwork(obs_dims[0], n_agents, action_dim, cfg.rnn_hidden_dim)
        self.central_val = CentralizedValueNetwork(global_state_dim, n_agents, cfg.rnn_hidden_dim, obs_dims[0],
                                                   action_dim)

        self.target_local_adv_1 = LocalAdvantageNetwork(obs_dims[0], n_agents, action_dim, cfg.rnn_hidden_dim)
        self.target_local_adv_2 = LocalAdvantageNetwork(obs_dims[0], n_agents, action_dim, cfg.rnn_hidden_dim)
        self.target_central_val = CentralizedValueNetwork(global_state_dim, n_agents, cfg.rnn_hidden_dim, obs_dims[0],
                                                          action_dim)

        # Optimizers
        self.critic_params = list(self.local_adv_1.parameters()) + \
                             list(self.local_adv_2.parameters()) + \
                             list(self.central_val.parameters())
        self.critic_optimizer = optim.Adam(self.critic_params, lr=cfg.lr_high)

        self.crawler_optimizer = optim.Adam(self.crawlers.parameters(), lr=cfg.lr_crawlers_high, weight_decay=1e-5)

        self._update_targets(initial=True)

    def _update_targets(self, initial=False):
        if initial or self.gradient_steps % self.cfg.target_update_frequency == 0:
            self.target_local_adv_1.load_state_dict(self.local_adv_1.state_dict())
            self.target_local_adv_2.load_state_dict(self.local_adv_2.state_dict())
            self.target_central_val.load_state_dict(self.central_val.state_dict())
            for i in range(self.n_agents):
                self.target_crawlers[i].load_state_dict(self.crawlers[i].state_dict())

    def get_reset_stats(self):
        if self.pruning_steps == 0: return 0.0
        avg_pct = (self.reset_counter / (self.pruning_steps * self.cfg.num_crawlers * self.n_agents)) * 100
        self.reset_counter = 0
        self.pruning_steps = 0
        return avg_pct

    def select_actions(self, obs_list, last_actions, last_hidden_states, epsilon):
        actions_taken = []
        new_hidden_states = []
        device = next(self.local_adv_1.parameters()).device
        if self.crawler_usage.device != device: self.crawler_usage = self.crawler_usage.to(device)

        with torch.no_grad():
            for i in range(self.n_agents):
                current_crawlers_i = self.crawlers[i]()

                obs = torch.tensor(obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)
                lact = torch.tensor(last_actions[i], dtype=torch.float32, device=device).unsqueeze(0)
                h_prev = last_hidden_states[i].to(device)
                aid = self.agent_ids[i].unsqueeze(0).to(device)

                h_next = self.local_adv_1.get_hidden_state(obs, aid, lact, h_prev)
                new_hidden_states.append(h_next)

                if np.random.rand() < epsilon:
                    # Explore: Random action in continuous space
                    action = torch.rand(self.action_dim, device=device) * 2.0 - 1.0
                else:
                    # Exploit: Select best Crawler
                    h_expanded = h_next.expand(len(current_crawlers_i), -1)
                    adv_values = self.local_adv_1.get_advantage(h_expanded, current_crawlers_i)
                    best_idx = torch.argmax(adv_values)
                    action = current_crawlers_i[best_idx]
                    self.crawler_usage[i, best_idx.item()] += 1

                actions_taken.append(action.detach().cpu().numpy().flatten().astype(np.float32))

        return actions_taken, torch.stack(new_hidden_states).squeeze(1)

    def learn(self, batch):
        self.gradient_steps += 1
        device = next(self.local_adv_1.parameters()).device
        gs, obs, acts, rew, done, last_act, alive_mask, mask_batch = [b.to(device) for b in batch]
        B, T = gs.shape[0], gs.shape[1]

        # Init RNN States
        h_adv_1 = self.local_adv_1.init_hidden().expand(B * self.n_agents, -1).to(device)
        h_adv_2 = self.local_adv_2.init_hidden().expand(B * self.n_agents, -1).to(device)
        h_trg_1 = self.target_local_adv_1.init_hidden().expand(B * self.n_agents, -1).to(device)
        h_trg_2 = self.target_local_adv_2.init_hidden().expand(B * self.n_agents, -1).to(device)
        ids_flat = self.agent_ids.unsqueeze(0).expand(B, -1, -1).to(device).reshape(B * self.n_agents, -1)

        q1_curr_list, q2_curr_list, q_targ_list = [], [], []
        hidden_states_history = []

        # Critic Update Phase
        for t in range(T):
            obs_t = obs[:, t].reshape(B * self.n_agents, -1)
            act_t = acts[:, t].reshape(B * self.n_agents, -1)
            lact_t = last_act[:, t].reshape(B * self.n_agents, -1)
            mask_t = alive_mask[:, t].reshape(B * self.n_agents)

            idx_next = min(t + 1, T - 1)
            obs_next = obs[:, idx_next].reshape(B * self.n_agents, -1)
            mask_next = alive_mask[:, idx_next].reshape(B * self.n_agents)

            # Online Forward
            h_next_1 = self.local_adv_1.get_hidden_state(obs_t, ids_flat, lact_t, h_adv_1)
            h_next_2 = self.local_adv_2.get_hidden_state(obs_t, ids_flat, lact_t, h_adv_2)

            hidden_states_history.append(h_next_1)

            adv_taken_1 = self.local_adv_1.get_advantage(h_next_1, act_t)
            adv_taken_2 = self.local_adv_2.get_advantage(h_next_2, act_t)

            v_val = self.central_val(gs[:, t], h_next_1, obs_t, lact_t, ids_flat, mask_t)

            q1_curr_list.append(
                (v_val.repeat(1, self.n_agents).reshape(B * self.n_agents, 1) + adv_taken_1).reshape(B, self.n_agents))
            q2_curr_list.append(
                (v_val.repeat(1, self.n_agents).reshape(B * self.n_agents, 1) + adv_taken_2).reshape(B, self.n_agents))

            # Target Forward (Double Q with Crawler Population)
            with torch.no_grad():
                h_trg_next_1 = self.target_local_adv_1.get_hidden_state(obs_next, ids_flat, act_t, h_trg_1)
                h_trg_next_2 = self.target_local_adv_2.get_hidden_state(obs_next, ids_flat, act_t, h_trg_2)

                all_target_crawlers = torch.stack([tc() for tc in self.target_crawlers], dim=0)
                num_targ_crawlers = self.cfg.num_crawlers

                # Expand to evaluate all crawlers for target calculation
                pop_exp = all_target_crawlers.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.n_agents,
                                                                                         num_targ_crawlers,
                                                                                         self.action_dim)
                h_trg_exp_1 = h_trg_next_1.unsqueeze(1).expand(-1, num_targ_crawlers, -1).reshape(
                    B * self.n_agents * num_targ_crawlers, -1)
                h_trg_exp_2 = h_trg_next_2.unsqueeze(1).expand(-1, num_targ_crawlers, -1).reshape(
                    B * self.n_agents * num_targ_crawlers, -1)

                pop_exp_flat = pop_exp.reshape(B * self.n_agents * num_targ_crawlers, self.action_dim)

                adv_all_1 = self.target_local_adv_1.get_advantage(h_trg_exp_1, pop_exp_flat).reshape(B * self.n_agents,
                                                                                                     num_targ_crawlers)
                adv_all_2 = self.target_local_adv_2.get_advantage(h_trg_exp_2, pop_exp_flat).reshape(B * self.n_agents,
                                                                                                     num_targ_crawlers)

                # Clipped Double Q
                min_adv_all = torch.min(adv_all_1, adv_all_2)
                max_adv, _ = torch.max(min_adv_all, dim=1, keepdim=True)

                v_next = self.target_central_val(gs[:, idx_next], h_trg_next_1, obs_next, act_t, ids_flat, mask_next)
                q_targ_list.append(
                    (v_next.repeat(1, self.n_agents).reshape(B * self.n_agents, 1) + max_adv).reshape(B, self.n_agents))

            h_adv_1, h_adv_2 = h_next_1, h_next_2
            h_trg_1, h_trg_2 = h_trg_next_1, h_trg_next_2

        q1_curr = torch.stack(q1_curr_list, dim=1)
        q2_curr = torch.stack(q2_curr_list, dim=1)
        q_targ = torch.stack(q_targ_list, dim=1)

        rew_b = rew.unsqueeze(-1).expand(-1, -1, self.n_agents)
        don_b = done.unsqueeze(-1).expand(-1, -1, self.n_agents)
        mask_b_loss = mask_batch.unsqueeze(-1).expand(-1, -1, self.n_agents)

        final_mask = mask_b_loss * alive_mask

        td_targ = rew_b + self.cfg.gamma * (1 - don_b.float()) * q_targ
        loss_td = (((q1_curr - td_targ.detach()) * final_mask) ** 2 + (
                (q2_curr - td_targ.detach()) * final_mask) ** 2).sum() / (final_mask.sum() + 1e-6)

        self.critic_optimizer.zero_grad()
        loss_td.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_params, 10.0)
        self.critic_optimizer.step()
        self._update_targets()

        # 2. Crawler Update Phase
        if self.gradient_steps % 1 == 0:
            self.crawler_optimizer.zero_grad()
            # Freeze Critic for Crawler Update
            for p in self.local_adv_1.parameters(): p.requires_grad = False

            all_curr_crawlers = torch.stack([c() for c in self.crawlers], dim=0)
            full_hidden = torch.stack(hidden_states_history, dim=0).detach()

            T_len, BN, H = full_hidden.shape
            full_hidden_reshaped = full_hidden.view(T_len, B, self.n_agents, H)

            h_exp = full_hidden_reshaped.view(T_len * B, self.n_agents, 1, H).expand(-1, -1, self.cfg.num_crawlers, -1)
            h_exp_flat = h_exp.reshape(-1, H)

            anc_exp = all_curr_crawlers.unsqueeze(0).expand(T_len * B, -1, -1, -1)
            anc_exp_flat = anc_exp.reshape(-1, self.action_dim)

            advs = self.local_adv_1.get_advantage(h_exp_flat, anc_exp_flat)
            advs = advs.view(T_len * B, self.n_agents, self.cfg.num_crawlers)

            total_climb_loss = 0

            if self.steps_since_win.device != device: self.steps_since_win = self.steps_since_win.to(device)
            if self.freeze_counters.device != device: self.freeze_counters = self.freeze_counters.to(device)

            self.steps_since_win += 1

            for i in range(self.n_agents):
                agent_advs = advs[:, i, :]

                # Competitive Learning: Only Top-K crawlers get gradients
                K = min(self.cfg.top_k, self.cfg.num_crawlers)
                top_k_values, top_k_indices = torch.topk(agent_advs, k=K, dim=1)

                unique_winners = torch.unique(top_k_indices)
                self.steps_since_win[i, unique_winners] = 0

                agent_loss = -torch.mean(top_k_values)
                total_climb_loss += agent_loss

            total_climb_loss.backward()

            # Apply gradient mask for frozen crawlers (Post-Reset Freezing)
            self.freeze_counters = torch.clamp(self.freeze_counters - 1, min=0)

            for i in range(self.n_agents):
                learn_mask = (self.freeze_counters[i] == 0).float().unsqueeze(1)
                if self.crawlers[i].actions.grad is not None:
                    self.crawlers[i].actions.grad = self.crawlers[i].actions.grad * learn_mask

            self.crawler_optimizer.step()
            # Unfreeze Critic
            for p in self.local_adv_1.parameters(): p.requires_grad = True

        # 3. Population Management (Pruning & Resets)
        if self.gradient_steps % 50 == 0:
            with torch.no_grad():
                for agent_idx in range(self.n_agents):
                    current_action_space_pos = self.crawlers[agent_idx]()
                    to_reset = []
                    dists = torch.cdist(current_action_space_pos, current_action_space_pos, p=2)
                    threshold = self.clashing_threshold

                    # Pruning: Proximity
                    if threshold > 1e-5:
                        usage = self.crawler_usage[agent_idx].to(device)
                        close_mask = (dists < threshold) & torch.triu(torch.ones_like(dists), diagonal=1).bool()
                        clashing_pairs = torch.nonzero(close_mask)
                        is_frozen = (self.freeze_counters[agent_idx] > 0)

                        for pair in clashing_pairs:
                            i, j = pair[0].item(), pair[1].item()
                            if is_frozen[i] or is_frozen[j]: continue
                            u_i, u_j = usage[i].item(), usage[j].item()
                            if u_i >= u_j:
                                if j not in to_reset: to_reset.append(j)
                            else:
                                if i not in to_reset: to_reset.append(i)

                    # Pruning: Starvation
                    starved_mask = (self.steps_since_win[agent_idx] > self.cfg.starvation_threshold)
                    starved_indices = torch.nonzero(
                        starved_mask & ~(self.freeze_counters[agent_idx] > 0)).flatten().tolist()
                    for idx in starved_indices:
                        if idx not in to_reset: to_reset.append(idx)

                    # Maximin Resets
                    if len(to_reset) > 0:
                        reset_indices = torch.tensor(to_reset, device=device)
                        K_candidates = self.cfg.maximin_candidates
                        best_candidates = []

                        for _ in range(len(to_reset)):
                            candidates = torch.rand(K_candidates, self.action_dim, device=device) * 4.0 - 2.0
                            cand_tanh = torch.tanh(candidates)
                            pop_tanh = current_action_space_pos
                            c_dists = torch.cdist(cand_tanh, pop_tanh, p=2)
                            min_dists, _ = torch.min(c_dists, dim=1)
                            best_k = torch.argmax(min_dists)
                            best_candidates.append(candidates[best_k])

                        new_pos = torch.stack(best_candidates, dim=0)

                        self.crawlers[agent_idx].actions.data[reset_indices] = new_pos
                        self.target_crawlers[agent_idx].actions.data[reset_indices] = new_pos

                        # Reset Optim State
                        param = self.crawlers[agent_idx].actions
                        if param in self.crawler_optimizer.state:
                            param_state = self.crawler_optimizer.state[param]
                            if 'exp_avg' in param_state: param_state['exp_avg'][reset_indices] = 0.0
                            if 'exp_avg_sq' in param_state: param_state['exp_avg_sq'][reset_indices] = 0.0

                        self.freeze_counters[agent_idx, reset_indices] = self.cfg.crawler_freeze_steps
                        self.steps_since_win[agent_idx, reset_indices] = 0
                        self.reset_counter += len(to_reset)

                self.pruning_steps += 1
                self.crawler_usage.fill_(0)


if __name__ == '__main__':
    cfg = Config()

    # Reproducibility
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # Environments
    env = multiwalker_v9.parallel_env(n_walkers=2, render_mode=None, terminate_on_fall=False)
    eval_env = multiwalker_v9.parallel_env(n_walkers=2, render_mode=None, terminate_on_fall=False)

    agents = env.possible_agents
    n_agents = len(agents)
    action_dim = env.action_space(agents[0]).shape[0]
    obs_dims = [env.observation_space(agent).shape[0] for agent in agents]
    global_state_dim = sum(obs_dims)

    # Initialization
    controller = MovingQLAN_Controller(obs_dims, action_dim, n_agents, global_state_dim, cfg)
    replay_buffer = EpisodeReplayBuffer(cfg.replay_buffer_size)

    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    job_id = os.environ.get('SLURM_JOB_ID', str(uuid.uuid4())[:8])

    # Tracking
    total_timesteps = 0
    all_episode_rewards_raw = []
    avg_train_rewards, avg_train_episodes = [], []
    all_eval_rewards, all_eval_episodes = [], []
    recent_rewards = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)

    print(f"--- Training MW (MovingQ-LAN) | Job ID: {job_id} ---", flush=True)
    print(f"Config: {cfg}", flush=True)

    for episode in range(cfg.num_episodes):
        obs, _ = env.reset()
        done, ep_rew = False, 0.0
        ep_len = 0

        h_state = controller.local_adv_1.init_hidden().repeat(n_agents, 1)
        last_actions = np.zeros((n_agents, action_dim), dtype=np.float32)
        ep_buf = []

        # LR Schedulers
        current_crawler_lr = np.interp(episode, [0, cfg.zeta_decay_steps],
                                       [cfg.lr_crawlers_high, cfg.lr_crawlers_low])
        for param_group in controller.crawler_optimizer.param_groups:
            param_group['lr'] = current_crawler_lr

        current_critic_lr = np.interp(episode, [0, cfg.num_episodes], [cfg.lr_high, cfg.lr_low])
        for param_group in controller.critic_optimizer.param_groups:
            param_group['lr'] = current_critic_lr

        # Epsilon
        eps = np.interp(episode, [0, cfg.epsilon_decay_steps], [cfg.epsilon_start, cfg.epsilon_end])

        while not done:
            obs_list = []
            alive_mask = []
            for agent_id in agents:
                if agent_id in obs:
                    obs_list.append(np.array(obs[agent_id], dtype=np.float32, copy=True))
                    alive_mask.append(1.0)
                else:
                    obs_list.append(np.zeros(obs_dims[0], dtype=np.float32))
                    alive_mask.append(0.0)

            # Select Actions
            actions, h_state = controller.select_actions(obs_list, last_actions, h_state, eps)
            h_state = h_state.detach()

            act_dict = {agents[i]: actions[i] for i in range(n_agents) if agents[i] in obs}
            next_obs, rew, term, trunc, _ = env.step(act_dict)

            global_state = np.array(np.concatenate(obs_list), dtype=np.float32, copy=True)
            team_rew = float(sum(rew.values()))
            ep_done = any(term.values()) or any(trunc.values()) or (len(next_obs) == 0)

            clean_actions = [np.array(a, dtype=np.float32) for a in actions]

            # Storage
            ep_buf.append((global_state, obs_list, clean_actions, team_rew, ep_done, last_actions.copy(), alive_mask))

            last_actions = np.array(actions, dtype=np.float32)
            obs, ep_rew, done = next_obs, ep_rew + team_rew, ep_done
            total_timesteps += 1
            ep_len += 1

        # Store to Buffer
        gs_arr = np.array([x[0] for x in ep_buf], dtype=np.float32)
        obs_arr = np.array([x[1] for x in ep_buf], dtype=np.float32)
        act_arr = np.array([x[2] for x in ep_buf], dtype=np.float32)
        rew_arr = np.array([x[3] for x in ep_buf], dtype=np.float32)
        don_arr = np.array([x[4] for x in ep_buf], dtype=np.bool_)
        last_act_arr = np.array([x[5] for x in ep_buf], dtype=np.float32)
        alive_arr = np.array([x[6] for x in ep_buf], dtype=np.float32)

        replay_buffer.store_episode((gs_arr, obs_arr, act_arr, rew_arr, don_arr, last_act_arr, alive_arr))

        all_episode_rewards_raw.append(ep_rew)
        recent_rewards.append(ep_rew)
        recent_lengths.append(ep_len)

        # Train
        if len(replay_buffer) >= cfg.batch_size:
            for _ in range(cfg.updates_per_episode):
                controller.learn(replay_buffer.sample(cfg.batch_size))

        # Logging
        if (episode + 1) % 1000 == 0:
            np.set_printoptions(threshold=sys.maxsize)
            for i in range(n_agents):
                print(f"\n[Episode {episode + 1}] Agent {i} Crawlers:\n{controller.crawlers[i]().data.cpu().numpy()}\n",
                      flush=True)
            np.set_printoptions(threshold=1000)

        # Eval Loop
        if (episode + 1) % cfg.eval_interval == 0:
            avg_train_reward = np.mean(recent_rewards)
            avg_len = np.mean(recent_lengths) if recent_lengths else 0

            eval_rewards = []
            for _ in range(cfg.eval_episodes):
                eval_obs, _ = eval_env.reset()
                eval_done = False
                eval_total_reward = 0
                eval_h = controller.local_adv_1.init_hidden().repeat(n_agents, 1)
                eval_last_act = np.zeros((n_agents, action_dim), dtype=np.float32)

                while not eval_done:
                    eval_obs_list = []
                    for agent_id in agents:
                        if agent_id in eval_obs:
                            eval_obs_list.append(eval_obs[agent_id])
                        else:
                            eval_obs_list.append(np.zeros(obs_dims[0], dtype=np.float32))

                    # Epsilon 0 ensures we strictly use the best Crawler
                    eval_acts, eval_new_h = controller.select_actions(eval_obs_list, eval_last_act, eval_h,
                                                                      epsilon=0)
                    eval_h = eval_new_h.detach()

                    eval_act_dict = {agents[i]: eval_acts[i] for i in range(n_agents) if agents[i] in eval_obs}
                    eval_next_obs, eval_r, eval_term, eval_trunc, _ = eval_env.step(eval_act_dict)

                    eval_done = any(eval_term.values()) or any(eval_trunc.values()) or (len(eval_next_obs) == 0)
                    eval_obs = eval_next_obs
                    eval_last_act = np.array(eval_acts, dtype=np.float32)
                    eval_total_reward += float(sum(eval_r.values()))
                eval_rewards.append(eval_total_reward)

            avg_eval_reward = np.mean(eval_rewards)
            all_eval_rewards.append(avg_eval_reward)
            all_eval_episodes.append(episode + 1)
            avg_train_rewards.append(avg_train_reward)
            avg_train_episodes.append(episode + 1)

            reset_pct = controller.get_reset_stats()
            try:
                ram_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            except:
                ram_mb = 0.0

            print(
                f"Ep {episode + 1} | Buf: {len(replay_buffer)} | AvgLen: {avg_len:.1f} | RAM: {ram_mb:.2f} MB | "
                f"Train: {avg_train_reward:.2f} | Eval: {avg_eval_reward:.2f} | Rst: {reset_pct:.1f}% | "
                f"Eps: {eps:.3f} | LR: {current_critic_lr:.2e} | CLR: {current_crawler_lr:.2e}", flush=True)

    # Save
    file_name = f"MW_MovingQ_Crawlers{cfg.num_crawlers}_TopK{cfg.top_k}_ID{job_id}.npz"
    save_path = os.path.join(cfg.save_dir, file_name)

    np.savez(save_path,
             train_episodes=np.array(avg_train_episodes),
             train_rewards=np.array(avg_train_rewards),
             eval_episodes=np.array(all_eval_episodes),
             eval_rewards=np.array(all_eval_rewards),
             config=asdict(cfg))

    print(f"Saved results to {save_path}", flush=True)
    env.close()
    eval_env.close()