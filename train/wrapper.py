import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from coinche.gym.env import GymCoinche
from coinche.player import RandomPlayer, AIPlayer, DeterministicPlayer, DeterministicPlayer_v2
from coinche.gym.gymplayer import GymPlayer
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import wandb
from collections import deque, namedtuple
from concurrent.futures import ThreadPoolExecutor
import threading
import os


class CoincheState:
    def __init__(self, players=None, reset=True):
        self.env = GymCoinche(players=players)
        self.observation = None
        if reset:
            self.observation, _ = self.env.reset()
        self.terminated = False

    def current_player(self):
        return self.env._get_current_player().index

    def legal_actions(self):
        return self.env._legal_action()

    def apply_action(self, action):
        obs, _, terminated, _, _ = self.env.step(action)
        self.observation = obs
        self.terminated = terminated

    def is_terminal(self):
        return self.terminated

    def returns(self):
        return self.env._get_return()

    def observation_tensor(self):
        return np.array(self.env._get_current_observation(), dtype=np.float32)

    def information_state_tensor(self):
        return self.observation_tensor()

    def clone(self):
        new_players = [p.clone() for p in self.env.players]
        cloned = CoincheState(new_players, reset=False)
        cloned.env = self.env.clone(new_players)
        cloned.observation = np.copy(self.observation)
        cloned.terminated = self.terminated
        return cloned

# Transitions for advantage and policy memories
AdvTransition = namedtuple('AdvTransition', ['obs', 'action', 'advantage'])
PolTransition = namedtuple('PolTransition', ['obs', 'pi_target', 'player'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, item):
        self.buffer.append(item)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

def make_mlp(input_dim, output_dim, hidden_sizes):
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)

class DeepCFRWrapper:
    def __init__(self, config: dict):

        self.cpu_count = os.cpu_count() or 1
        
        # Setup players (self-play, random or deterministic opponents)
        cfr_players = config.get('cfr_players', 'self_play')
        if cfr_players == 'self_play':
            self.players = [GymPlayer(i, name) for i, name in enumerate(["N", "E", "S", "W"])]
        elif cfr_players == 'random_opponent':
            self.players = [GymPlayer(0, "N"), RandomPlayer(1, "E"),
                            GymPlayer(2, "S"), RandomPlayer(3, "W")]
        elif cfr_players == 'det_opponent':
            self.players = [GymPlayer(0, "N"), DeterministicPlayer(1, "E"),
                            GymPlayer(2, "S"), DeterministicPlayer(3, "W")]
        elif cfr_players == 'det2_opponent':
            self.players = [GymPlayer(0, "N"), DeterministicPlayer_v2(1, "E"),
                            DeterministicPlayer_v2(2, "S"), DeterministicPlayer_v2(3, "W")]
        else:
            raise ValueError("Invalid player configuration.")

        # Initialize state template to get dims
        template = CoincheState(players=self.players, reset=False)
        obs0 = template.observation_tensor()
        self.obs_dim = obs0.shape[0]
        self.num_players = len(template.returns())
        self.num_actions = template.env.action_space.n

        # Hyperparameters and defaults
        defaults = {
            'adv_hidden': [64, 64], 'adv_lr': 1e-4, 'adv_mem_size': 100_000,
            'pol_hidden': [64, 64], 'pol_lr': 1e-4, 'pol_mem_size': 100_000,
            'batch_size': 256, 'num_iterations': 10, 'log_interval': 1,
            'num_traversals': 1, 'adv_epochs': 1, 'pol_epochs': 1, 
        }
        self.num_iterations = config.get('total_timesteps', defaults['num_iterations'])
        self.log_interval = config.get('log_every', defaults['log_interval'])
        self.batch_size = config.get('batch_size', defaults['batch_size'])
        self.adv_hidden = config.get('adv_hidden', defaults['adv_hidden'])
        self.adv_lr = config.get('adv_lr', defaults['adv_lr'])
        self.adv_mem_size = config.get('adv_mem_size', defaults['adv_mem_size'])
        self.pol_hidden = config.get('pol_hidden', defaults['pol_hidden'])
        self.pol_lr = config.get('pol_lr', defaults['pol_lr'])
        self.pol_mem_size = config.get('pol_mem_size', defaults['pol_mem_size'])
        self.num_traversals = config.get('num_traversals', defaults['num_traversals'])
        self.adv_epochs = config.get('adv_epochs', defaults['adv_epochs'])
        self.pol_epochs = config.get('pol_epochs', defaults['pol_epochs'])
        self.max_workers = config.get('max_workers', self.cpu_count)

        # Networks and optimizers
        self.adv_nets = [make_mlp(self.obs_dim, self.num_actions, self.adv_hidden)
                         for _ in range(self.num_players)]
        self.adv_opts = [optim.Adam(net.parameters(), lr=self.adv_lr)
                         for net in self.adv_nets]
        self.policy_net = make_mlp(self.obs_dim + 1, self.num_actions, self.pol_hidden)
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=self.pol_lr)

        # Memories
        self.adv_memory = [ReplayBuffer(self.adv_mem_size) for _ in range(self.num_players)]
        self.pol_memory = ReplayBuffer(self.pol_mem_size)

        self.log_data = {'iter':[], 'policy_loss':[], 'player_rewards':[]}

        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.lock = threading.Lock()
        
    def traverse(self, state, target, pi=1.0, pi_op=1.0):
        if state.is_terminal():
            return state.returns()[target]
        cur = state.current_player()
        obs = state.information_state_tensor()
        legal = state.legal_actions()
        # masked softmax
        inp = torch.tensor(np.insert(obs,0,cur),dtype=torch.float32).unsqueeze(0)
        logits = self.policy_net(inp).squeeze(0)
        mask = torch.zeros(self.num_actions)
        mask[legal]=1
        exp_l = torch.exp(logits)*mask
        pol = exp_l/exp_l.sum()

        if cur == target:
            # compute v_all advantage baseline
            with torch.no_grad():
                v_all = self.adv_nets[cur](torch.tensor(obs,dtype=torch.float32).unsqueeze(0)).squeeze(0)


            def worker(a):
                nxt = state.clone()
                nxt.apply_action(a)
                util = self.traverse(nxt, target, pi*pol[a].item(), pi_op)
                adv = (util - v_all[a].item()) / pi_op
                with self.lock:
                    self.adv_memory[cur].push(AdvTransition(obs, a, adv))
                return a, util

            futures = [self.executor.submit(worker, a) for a in legal]
            utilities = {}
            for f in futures:
                a, util = f.result()
                utilities[a] = util
                #break
            '''def worker(a):
                nxt = state.clone()
                nxt.apply_action(a)
                util = self.traverse(nxt, target, pi*pol[a].item(), pi_op)
                adv = (util - v_all[a].item()) / pi_op
                with self.lock:
                    self.adv_memory[cur].push(AdvTransition(obs, a, adv))
                return a, util

            # parallel execution over legal actions
            utilities = {}
            cpu_count = os.cpu_count() or 1
            max_workers = min(len(legal), self.max_workers, cpu_count)
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                for a, util in exe.map(worker, legal):
                    utilities[a] = util'''

        
            # policy target
            v_np = v_all.cpu().numpy()
            mask_arr = np.zeros(self.num_actions); mask_arr[legal]=1
            adv_pos = np.clip(v_np,0,None)*mask_arr
            pi_tgt = adv_pos/adv_pos.sum() if adv_pos.sum()>0 else mask_arr/mask_arr.sum()
            with self.lock:
                self.pol_memory.push(PolTransition(obs,pi_tgt,cur))

            # sample action to continue
            a_samp = legal[torch.multinomial(pol[legal],1).item()]
            return utilities[a_samp]
        else:
            # opponent node: sample action to continue
            a_samp = legal[torch.multinomial(pol[legal],1).item()]
            nxt = state.clone(); nxt.apply_action(a_samp)
            return self.traverse(nxt, target, pi, pi_op*pol[a_samp].item())

    def optimize_adv(self, player):
        if len(self.adv_memory[player]) < self.batch_size:
            return 0
        batch = self.adv_memory[player].sample(self.batch_size)
        obs = torch.stack([torch.tensor(b.obs, dtype=torch.float32) for b in batch])
        acts = torch.tensor([b.action for b in batch], dtype=torch.long)
        advs = torch.tensor([b.advantage for b in batch], dtype=torch.float32)
        preds = self.adv_nets[player](obs)
        chosen = preds.gather(1, acts.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(chosen, advs)
        opt = self.adv_opts[player]
        opt.zero_grad(); loss.backward(); opt.step()
        return loss

    def optimize_policy(self):
        if len(self.pol_memory) < self.batch_size:
            return
        batch = self.pol_memory.sample(self.batch_size)
        obs_inputs = np.array([np.insert(b.obs, 0, b.player) for b in batch], dtype=np.float32)
        obs_batch = torch.tensor(obs_inputs, dtype=torch.float32)
        pi_targets = torch.tensor([b.pi_target for b in batch], dtype=torch.float32)
        logits = self.policy_net(obs_batch)
        pred_pi = torch.softmax(logits, dim=1)
        loss = nn.MSELoss()(pred_pi, pi_targets)
        self.policy_opt.zero_grad(); loss.backward(); self.policy_opt.step()

    def learn(self, total_timesteps, callback=None, use_masking=False):
        init_zeros = [0, 0, 0, 0]
        metrics = {
            "iteration": 0,           
            **{f"adv_sizes/player_{i}": size
            for i, size in enumerate(init_zeros)},
            "pol_size": len(self.pol_memory),   
            "policy_loss": 0,  
            "policy_avg_eps_length": 0,
            **{f"policy_avg_reward/player_{i}": r
            for i, r in enumerate(init_zeros)},
        }
        wandb.log(metrics)
        for it in range(self.num_iterations):
            # Reset advantage memory
            self.adv_memory = [ReplayBuffer(self.adv_mem_size) for _ in range(self.num_players)]
            # Collect advantage samples
            for p in range(self.num_players):
                for _ in range(self.num_traversals):
                    state = CoincheState(players=self.players)
                    self.traverse(state, p)
                # Train advantage network
                for _ in range(self.adv_epochs):
                    self.optimize_adv(p)
            # Train policy network
            cur_pol_loss = 0
            for _ in range(self.pol_epochs):
                cur_pol_loss += self.optimize_policy()/self.pol_epochs
            if (it + 1) % self.log_interval == 0:
                print(f"[DeepCFR] Iter {it+1}/{self.num_iterations}, "
                      f"adv sizes={[len(buf) for buf in self.adv_memory]}, "
                      f"pol size={len(self.pol_memory)}")
                policy_eval = self.evaluate_policy(self.policy_net, it)
                print(f"[DeepCFR] Policy: {cur_pol_loss}") 
                # Log metrics to Weights & Biases
                metrics = {
                    "iteration": it + 1,           
                    **{f"adv_sizes/player_{i}": size
                    for i, size in enumerate(self.adv_memory)},  
                    "policy_loss": len(self.pol_memory),  
                    "policy_eval_loss": cur_pol_loss,   
                    "policy_avg_eps_length": policy_eval['avg_eps'],
                    **{f"policy_avg_reward/player_{i}": r
                    for i, r in enumerate(policy_eval["avg_reward"])},
                    }
                wandb.log(metrics)  


    def get_policy(self):
        return self.policy_net

    def save(self, prefix):
        for i, net in enumerate(self.adv_nets):
            torch.save(net.state_dict(), f"{prefix}_adv_{i}.pt")
        torch.save(self.policy_net.state_dict(), f"{prefix}_policy.pt")

    def evaluate_policy(self, policy, num_episodes=50, it=None):
        total = np.zeros(self.num_players)
        logger = {}
        eps_len = []
        for _ in range(num_episodes):
            state = CoincheState(players=self.players)
            eps_len_temp = 0
            while not state.is_terminal():
                eps_len_temp += 1
                current = state.current_player()
                obs = state.information_state_tensor()
                logits = policy(
                    torch.tensor(np.insert(obs, 0, current), dtype=torch.float32).unsqueeze(0)
                ).squeeze(0)
                legal = state.legal_actions()
                mask = torch.zeros(self.num_actions)
                mask[legal] = 1
                unscaled = torch.exp(logits) * mask
                probs = (unscaled / unscaled.sum()).detach().cpu().numpy()
                action = legal[np.argmax(probs[legal])]
                state.apply_action(action)
            total += np.array(state.returns(), dtype=float)
            eps_len.append(eps_len_temp)
        #print(f"Average episode length: {np.mean(eps_len)}")
        self.save(f"t_{it}")
        results = {
            'avg_eps': np.mean(eps_len),
            'avg_reward': total / num_episodes}
        return results 



class SingleDeepCFRWrapper:
    def __init__(self, config: dict):
        # Setup players
        cfr_players = config.get('cfr_players', 'self_play')
        if cfr_players == 'self_play':
            self.players = [GymPlayer(i, name) for i, name in enumerate(["N", "E", "S", "W"])]
        elif cfr_players == 'random_opponent':
            self.players = [GymPlayer(0, "N"), RandomPlayer(1, "E"),
                            GymPlayer(2, "S"), RandomPlayer(3, "W")]
        elif cfr_players == 'det_opponent':
            self.players = [GymPlayer(0, "N"), DeterministicPlayer(1, "E"),
                            GymPlayer(2, "S"), DeterministicPlayer(3, "W")]
        else:
            raise ValueError("Invalid player configuration.")

        template = CoincheState(players=self.players, reset=False)
        obs0 = template.observation_tensor()
        self.obs_dim = obs0.shape[0]
        self.num_players = len(template.returns())
        self.num_actions = template.env.action_space.n

        defaults = {
            'adv_hidden': [64, 64], 'adv_lr': 1e-4, 'adv_mem_size': 100_000,
            'pol_hidden': [64, 64], 'pol_lr': 1e-4, 'pol_mem_size': 100_000,
            'batch_size': 256, 'num_iterations': 10000, 'log_interval': 10,
            'num_traversals': 512, 'adv_epochs': 16, 'pol_epochs': 16
        }
        self.num_iterations = config.get('total_timesteps', defaults['num_iterations'])
        self.log_interval = config.get('log_every', defaults['log_interval'])
        self.batch_size = config.get('batch_size', defaults['batch_size'])
        self.adv_hidden = config.get('adv_hidden', defaults['adv_hidden'])
        self.adv_lr = config.get('adv_lr', defaults['adv_lr'])
        self.adv_mem_size = config.get('adv_mem_size', defaults['adv_mem_size'])
        self.pol_hidden = config.get('pol_hidden', defaults['pol_hidden'])
        self.pol_lr = config.get('pol_lr', defaults['pol_lr'])
        self.pol_mem_size = config.get('pol_mem_size', defaults['pol_mem_size'])
        self.num_traversals = config.get('num_traversals', defaults['num_traversals'])
        self.adv_epochs = config.get('adv_epochs', defaults['adv_epochs'])
        self.pol_epochs = config.get('pol_epochs', defaults['pol_epochs'])

        self.adv_nets = [
            make_mlp(self.obs_dim, self.num_actions, self.adv_hidden)
            for _ in range(self.num_players)
        ]
        self.adv_opts = [
            optim.Adam(net.parameters(), lr=self.adv_lr)
            for net in self.adv_nets
        ]
        self.policy_net = make_mlp(self.obs_dim + 1, self.num_actions, self.pol_hidden)
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=self.pol_lr)

        self.adv_memory = [ReplayBuffer(self.adv_mem_size) for _ in range(self.num_players)]
        self.pol_memory = ReplayBuffer(self.pol_mem_size)

    def traverse(self, state, target_player, pi=1.0, pi_op=1.0):
        if state.is_terminal():
            return state.returns()[target_player]

        current = state.current_player()
        obs = state.information_state_tensor()
        legal = state.legal_actions()

        inp = torch.tensor(np.insert(obs, 0, current), dtype=torch.float32).unsqueeze(0)
        logits = self.policy_net(inp).squeeze(0)
        mask = torch.zeros(self.num_actions)
        mask[legal] = 1
        exp_logits = torch.exp(logits) * mask
        probs = exp_logits / exp_logits.sum()

        action_idx = torch.multinomial(probs[legal], num_samples=1).item()
        action = legal[action_idx]

        next_state = state.clone()
        next_state.apply_action(action)

        if current == target_player:
            u = self.traverse(next_state, target_player, pi * probs[action].item(), pi_op)
            with torch.no_grad():
                v = self.adv_nets[current](
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                ).squeeze(0)
            advantage = (u - v[action].item()) / pi_op
            self.adv_memory[current].push(
                AdvTransition(obs, action, advantage)
            )
            v_np = v.detach().cpu().numpy()
            mask_arr = np.zeros(self.num_actions)
            mask_arr[legal] = 1
            adv_pos = np.clip(v_np, 0, None) * mask_arr
            if adv_pos.sum() > 0:
                pi_target = adv_pos / adv_pos.sum()
            else:
                pi_target = mask_arr / mask_arr.sum()
            self.pol_memory.push(
                PolTransition(obs, pi_target, current)
            )
            return u
        else:
            return self.traverse(
                next_state, target_player,
                pi, pi_op * probs[action].item()
            )

    def optimize_adv(self, player):
        if len(self.adv_memory[player]) < self.batch_size:
            return
        batch = self.adv_memory[player].sample(self.batch_size)
        obs = torch.stack([torch.tensor(b.obs, dtype=torch.float32) for b in batch])
        acts = torch.tensor([b.action for b in batch], dtype=torch.long)
        advs = torch.tensor([b.advantage for b in batch], dtype=torch.float32)
        preds = self.adv_nets[player](obs)
        chosen = preds.gather(1, acts.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(chosen, advs)
        opt = self.adv_opts[player]
        opt.zero_grad(); loss.backward(); opt.step()

    def optimize_policy(self):
        if len(self.pol_memory) < self.batch_size:
            return
        batch = self.pol_memory.sample(self.batch_size)
        obs_inputs = np.array([np.insert(b.obs, 0, b.player) for b in batch], dtype=np.float32)
        obs_batch = torch.tensor(obs_inputs, dtype=torch.float32)
        pi_targets = torch.tensor([b.pi_target for b in batch], dtype=torch.float32)
        logits = self.policy_net(obs_batch)
        pred_pi = torch.softmax(logits, dim=1)
        loss = nn.MSELoss()(pred_pi, pi_targets)
        self.policy_opt.zero_grad(); loss.backward(); self.policy_opt.step()

    def learn(self, total_timesteps, callback=None, use_masking=False):
        for it in range(self.num_iterations):
            self.adv_memory = [ReplayBuffer(self.adv_mem_size) for _ in range(self.num_players)]
            for p in range(self.num_players):
                for _ in range(self.num_traversals):
                    state = CoincheState(players=self.players)
                    self.traverse(state, p)
                for _ in range(self.adv_epochs):
                    self.optimize_adv(p)
            for _ in range(self.pol_epochs):
                self.optimize_policy()
            if (it + 1) % self.log_interval == 0:
                print(f"[DeepCFR] Iter {it+1}/{self.num_iterations}, "
                      f"adv sizes={[len(buf) for buf in self.adv_memory]}, "
                      f"pol size={len(self.pol_memory)}")
                print(f"[DeepCFR] Policy: {self.evaluate_policy(self.policy_net)}") 


    def get_policy(self):
        return self.policy_net

    def save(self, prefix):
        for i, net in enumerate(self.adv_nets):
            torch.save(net.state_dict(), f"{prefix}_adv_{i}.pt")
        torch.save(self.policy_net.state_dict(), f"{prefix}_policy.pt")

    def evaluate_policy(self, policy, num_episodes=50):
        total = np.zeros(self.num_players)
        eps_len = []
        for _ in range(num_episodes):
            state = CoincheState(players=self.players)
            eps_len_temp = 0
            while not state.is_terminal():
                eps_len_temp += 1
                current = state.current_player()
                obs = state.information_state_tensor()
                logits = policy(
                    torch.tensor(np.insert(obs, 0, current), dtype=torch.float32).unsqueeze(0)
                ).squeeze(0)
                legal = state.legal_actions()
                mask = torch.zeros(self.num_actions)
                mask[legal] = 1
                unscaled = torch.exp(logits) * mask
                probs = (unscaled / unscaled.sum()).detach().cpu().numpy()
                action = legal[np.argmax(probs[legal])]
                state.apply_action(action)
            total += np.array(state.returns(), dtype=float)
            eps_len.append(eps_len_temp)
        print(f"Average episode length: {np.mean(eps_len)}")
        return total / num_episodes



Transition = namedtuple('Transition', ['obs', 'action', 'advantage', 'player', ]) #'pi'

class SampleDeepCFRWrapper:
    def __init__(self, config: dict):
        if config.get('cfr_players') is not None:
            if config['cfr_players'] == 'self_play':
                self.players = [GymPlayer(i, name) for i, name in enumerate(["N", "E", "S", "W"])]
            elif config['cfr_players'] == 'random_opponent':
                self.players = [GymPlayer(0, "N"), RandomPlayer(1, "E"), GymPlayer(2, "S"), RandomPlayer(3, "W")]
            elif config['cfr_players'] == 'det_opponent':
                self.players = [GymPlayer(0, "N"), DeterministicPlayer(1, "E"), GymPlayer(2, "S"), DeterministicPlayer(3, "W")]
            else:
                raise ValueError("Invalid player configuration. Use 'self_play', 'random_opponent', or 'det_opponent'.")
        else:
            self.players = [GymPlayer(i, name) for i, name in enumerate(["N", "E", "S", "W"])]
        
        template = CoincheState(players=self.players, reset=False)
        obs0 = template.observation_tensor()
        self.obs_dim = len(obs0)
        self.num_players = len(template.returns())
        self.num_actions = template.env.action_space.n

        # Hyperparameters
        defaults = {
            'adv_hidden': [64, 64], 'adv_lr': 1e-4, 'adv_mem_size': 100_000,
            'pol_hidden': [64, 64], 'pol_lr': 1e-4, 'pol_mem_size': 100_000,
            'batch_size': 256, 'num_iterations': 10000, 'log_interval': 10,
            'num_traversals': 512, 'num_samples': 5
        }
        self.num_iterations = config.get('total_timesteps', defaults['num_iterations'])
        self.log_interval = config.get('log_every', defaults['log_interval'])
        self.batch_size = config.get('batch_size', defaults['batch_size'])

        self.adv_hidden = config.get('adv_hidden', defaults['adv_hidden'])
        self.adv_lr = config.get('adv_lr', defaults['adv_lr'])
        self.adv_mem_size = config.get('adv_mem_size', defaults['adv_mem_size'])
        self.pol_hidden = config.get('pol_hidden', defaults['pol_hidden'])
        self.pol_lr = config.get('pol_lr', defaults['pol_lr'])
        self.pol_mem_size = config.get('pol_mem_size', defaults['pol_mem_size'])

        self.num_traversals = defaults['num_traversals']
        self.num_samples = defaults['num_samples']

        # Networks and optimizers
        self.adv_nets = [make_mlp(self.obs_dim, self.num_actions, self.adv_hidden)
                         for _ in range(self.num_players)]
        self.adv_opts = [optim.Adam(net.parameters(), lr=self.adv_lr)
                         for net in self.adv_nets]
        self.policy_net = make_mlp(self.obs_dim + 1, self.num_actions, self.pol_hidden)
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=self.pol_lr)

        # Memories
        self.adv_memory = [ReplayBuffer(self.adv_mem_size) for _ in range(self.num_players)]
        self.pol_memory = ReplayBuffer(self.pol_mem_size)
        

    def traverse(self, state, target_player, pi=1.0, pi_op=1.0, eps_len=0):
        #print(eps_len)
        if state.is_terminal():
            #print("eps_len: ",eps_len)
            #print("state observation: ", state.observation_tensor())
            #print("state returns: ", state.returns())
            return state.returns()[target_player]
        current = state.current_player()
        obs = state.information_state_tensor()
        legal = state.legal_actions()
        #print(legal)
        logits = self.policy_net(
            torch.tensor(np.insert(obs, 0, current), dtype=torch.float32).unsqueeze(0)
        ).squeeze(0)
        
        legal_logits = logits[legal]
        legal_probs = torch.softmax(legal_logits, dim=0)
        probs = torch.zeros_like(logits)
        probs[legal] = legal_probs
        #print("legal ", legal)
        #print("probs", probs)
        #print("probs sum", probs.sum().item())
        #print("probs legal sum ", probs[legal].sum().item())

        if current == target_player:
            u_max = None
            for i in range(self.num_samples):
                action_idx = legal_probs.multinomial(num_samples=1).item()
                print("action_idx", action_idx)
                print("legal", legal)
                action = legal[action_idx]
                next_state = state.clone()
                next_state.apply_action(action)
                u = self.traverse(next_state, target_player, pi * probs[action], pi_op, eps_len + 1)
                with torch.no_grad():
                    v = self.adv_nets[current](torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                    v = v.squeeze(0).cpu().numpy()
                advantage = (u - v[action]) / pi_op
                self.adv_memory[current].push(obs, action, advantage, current)
                # policy target
                mask_arr = np.zeros_like(v)
                mask_arr[legal] = 1
                adv_pos = np.clip(v, 0, None) * mask_arr
                if adv_pos.sum() > 0:
                    pi_target = adv_pos / adv_pos.sum()
                else:
                    pi_target = mask_arr / mask_arr.sum()
                #print("pi_target", pi_target)
                self.pol_memory.push(obs, pi_target, None, current)
                u_max = u if u_max is None else max(u, u_max)
            return u_max
        else:
            action_idx = legal_probs.multinomial(num_samples=1).item()
            action = legal[action_idx]
            next_state = state.clone()
            next_state.apply_action(action)
            return self.traverse(next_state, target_player, pi, pi_op * probs[action], eps_len + 1)

    def optimize_adv(self, player):
        if len(self.adv_memory[player]) < self.batch_size:
            return
        batch = self.adv_memory[player].sample(self.batch_size)
        obs = torch.tensor([b.obs for b in batch], dtype=torch.float32)
        acts = torch.tensor([b.action for b in batch], dtype=torch.long)
        advs = torch.tensor([b.advantage for b in batch], dtype=torch.float32)
        preds = self.adv_nets[player](obs)
        chosen = preds.gather(1, acts.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(chosen, advs)
        opt = self.adv_opts[player]
        opt.zero_grad(); loss.backward(); opt.step()

    def optimize_policy(self):
        if len(self.pol_memory) < self.batch_size:
            return
        batch = self.pol_memory.sample(self.batch_size)
        #obs_batch = torch.tensor([np.insert(b.obs, 0, b.player) for b in batch], dtype=torch.float32)
        obs_batch_numpy = np.array([np.insert(b.obs, 0, b.player) for b in batch], dtype=np.float32)
        obs_batch = torch.tensor(obs_batch_numpy, dtype=torch.float32)
        pi_targets = torch.tensor([b.action for b in batch], dtype=torch.float32)
        logits = self.policy_net(obs_batch)
        pred_pi = torch.softmax(logits, dim=1)
        loss = nn.MSELoss()(pred_pi, pi_targets)
        print("Policy loss", loss.item())
        self.policy_opt.zero_grad(); loss.backward(); self.policy_opt.step()

    def learn(self, total_timesteps, callback=None, use_masking=False):
        for it in range(total_timesteps):
            for player in range(self.num_players):
                for _ in range(self.num_traversals):
                    state = CoincheState(players=self.players)
                    #print()
                    self.traverse(state, player)
                    #state 
                self.optimize_adv(player)
            self.optimize_policy()
            if (it + 1) % self.log_interval == 0:
                print(f"[DeepCFR] Iter {it+1}/{self.num_iterations}, adv_sizes={[len(buf) for buf in self.adv_memory]}, pol_size={len(self.pol_memory)}")
                print(f"[DeepCFR] Policy: {self.evaluate_policy(self.policy_net)}") 


    def get_policy(self):
        return self.policy_net

    def save(self, prefix):
        for i, net in enumerate(self.adv_nets):
            torch.save(net.state_dict(), f"{prefix}_adv_{i}.pt")
        torch.save(self.policy_net.state_dict(), f"{prefix}_policy.pt")

    def evaluate_policy(self, policy, num_episodes=50):
        total = np.zeros(self.num_players)
        eps_len = []
        for _ in range(num_episodes):
            state = CoincheState(players=self.players)
            eps_len_temp = 0
            while not state.is_terminal():
                eps_len_temp += 1
                current = state.current_player()
                obs = state.information_state_tensor()
                logits = policy(
                    torch.tensor(np.insert(obs, 0, current), dtype=torch.float32).unsqueeze(0)
                ).squeeze(0)
                legal = state.legal_actions()
                mask = torch.zeros(self.num_actions)
                mask[legal] = 1
                unscaled = torch.exp(logits) * mask
                probs = (unscaled / unscaled.sum()).detach().cpu().numpy()
                action = legal[np.argmax(probs[legal])]
                state.apply_action(action)
            total += np.array(state.returns(), dtype=float)
            eps_len.append(eps_len_temp)
        print(f"Average episode length: {np.mean(eps_len)}")
        return total / num_episodes


class OnlineCFRWrapper:
    def __init__(self, config: dict):
        # Pull players from config
        if config.get('cfr_players') is not None:
            if config['cfr_players'] == 'self_play':
                self.players = [GymPlayer(i, name) for i, name in enumerate(["N", "E", "S", "W"])]
            elif config['cfr_players'] == 'random_opponent':
                self.players = [GymPlayer(0, "N"), RandomPlayer(1, "E"), GymPlayer(2, "S"), RandomPlayer(3, "W")]
            elif config['cfr_players'] == 'det_opponent':
                self.players = [GymPlayer(0, "N"), DeterministicPlayer(1, "E"), GymPlayer(2, "S"), DeterministicPlayer(3, "W")]
            else:
                raise ValueError("Invalid player configuration. Use 'self_play', 'random_opponent', or 'det_opponent'.")
        else:
            self.players = [GymPlayer(i, name) for i, name in enumerate(["N", "E", "S", "W"])]
        # Build a template State to get dims
        template = CoincheState(players=self.players)
        self.num_players = len(template.returns())
        self.num_actions = template.env.action_space.n
        # CFR tables
        self.regrets = [np.zeros(self.num_actions) for _ in range(self.num_players)]
        self.strategy_sum = [np.zeros(self.num_actions) for _ in range(self.num_players)]

    def learn(self, total_timesteps, callback=None, use_masking=False):
        for t in range(1, total_timesteps + 1):
            state = CoincheState(players=self.players)
            self.cfr(state)
            if t % 100 == 0:
                print(f"Online CFR Iteration {t}/{total_timesteps} done.")

    def get_strategy(self, player_id, legal_actions):
        positive_regrets = np.maximum(self.regrets[player_id], 0)
        total = positive_regrets[legal_actions].sum()
        if total > 0:
            strat = positive_regrets / total
        else:
            strat = np.zeros(self.num_actions)
            strat[legal_actions] = 1.0 / len(legal_actions)
        self.strategy_sum[player_id] += strat
        return strat

    def cfr(self, state, pi=1.0):
        if state.is_terminal():
            return state.returns()
        player = state.current_player()
        legal = state.legal_actions()
        strategy = self.get_strategy(player, legal)
        util = np.zeros((self.num_players,))
        node_util = 0.0
        # For each action, recurse and accumulate
        for a in legal:
            next_state = state.clone()
            next_state.apply_action(a)
            ret = self.cfr(next_state, pi * strategy[a])
            util[a] = ret[player]
            node_util += strategy[a] * ret[player]
        # Update regrets
        for a in legal:
            self.regrets[player][a] += util[a] - node_util
        # Return utility vector
        result = np.zeros((self.num_players,))
        result[player] = node_util
        return result

    def get_policy(self):
        avg = []
        for summ in self.strategy_sum:
            if summ.sum() > 0:
                avg.append(summ / summ.sum())
            else:
                # uniform fallback
                vec = np.ones(self.num_actions) / self.num_actions
                avg.append(vec)
        return avg

    def save(self, filepath):
        policy = self.get_policy()
        policy_dict = {f'player_{i}': policy[i] for i in range(self.num_players)}
        np.savez(filepath, **policy_dict)
