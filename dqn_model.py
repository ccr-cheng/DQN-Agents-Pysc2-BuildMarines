import random, math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.hidden = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.fc(self.dropout(x))
        return x


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def push(self, item):
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.pos] = item
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = zip(*random.sample(self.memory, batch_size))
        return [torch.stack(x, dim=0) for x in batch]

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, n_observe, hidden_size, n_action, device,
                 lr=1e-2, batch_size=64, memory_size=10000, gamma=0.99,
                 clip_grad=1.0, eps_start=0.9, eps_decay=200, eps_end=0.05):
        self.n_observe = n_observe
        self.hidden_size = hidden_size
        self.n_action = n_action

        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.clip_grad = clip_grad
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end

        self.tgt_net = DQN(n_observe, hidden_size, n_action).to(device)
        self.act_net = DQN(n_observe, hidden_size, n_action).to(device)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.act_net.parameters(), lr=lr)
        self.cache = Memory(memory_size)
        self.steps_done = 0

        self.act_net.apply(self.initialize_weights)
        self.update_tgt()
        self.act_net.train()
        self.tgt_net.eval()

    @staticmethod
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    def update_tgt(self):
        self.tgt_net.load_state_dict(self.act_net.state_dict())

    def update_act(self):
        if len(self.cache) < self.batch_size:
            return

        state, action, reward, next_state = self.cache.sample(self.batch_size)
        pred_values = self.act_net(state).gather(1, action.unsqueeze(1)).squeeze(-1)
        tgt_values = self.tgt_net(next_state).max(1)[0].detach() * self.gamma + reward

        self.optimizer.zero_grad()
        loss = self.criterion(pred_values, tgt_values)
        loss.backward()
        clip_grad_norm_(self.act_net.parameters(), self.clip_grad)
        self.optimizer.step()
        return loss.item()

    def action(self, state, choices):
        eps_thres = self.eps_end + (self.eps_start - self.eps_end) * \
                    math.exp(-self.steps_done / self.eps_decay) \
            if self.act_net.training else self.eps_end
        self.steps_done += 1
        if random.random() > eps_thres:
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            with torch.no_grad():
                return self.act_net(state.unsqueeze(0)).argmax(1).item()
        else:
            return random.choice(choices)

    def save(self, path):
        torch.save({
            'model_state': self.act_net.state_dict(),
        }, path)

    def load(self, path):
        model = torch.load(path)
        self.act_net.load_state_dict(model['model_state'])
        self.tgt_net.load_state_dict(model['model_state'])
