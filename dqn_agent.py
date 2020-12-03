from dqn_model import DQNAgent
from marine_agent import BaseBMAgent, BMAction


class DQNBMAgent(BaseBMAgent, DQNAgent):
    def __init__(self, hidden_size, device, lr=1e-2, batch_size=64,
                 memory_size=10000, gamma=0.99, clip_grad=1.0,
                 eps_start=0.9, eps_decay=200, eps_end=0.05):
        BaseBMAgent.__init__(self)
        DQNAgent.__init__(
            self, self.n_state, hidden_size, self.n_action, device,
            lr, batch_size, memory_size, gamma, clip_grad,
            eps_start, eps_decay, eps_end
        )

    def step(self, obs):
        super().step(obs)

        player = obs.observation.player
        checkers = [
            (lambda x: True),
            self.check_make_scv,
            self.check_build_depot,
            self.check_build_barracks,
            self.check_make_marine,
            self.check_kill_marine
        ]
        choices = [i for i in range(self.n_action) if checkers[i](player)]
        if self.in_progress == BMAction.NO_OP:
            self.in_progress = BMAction(self.action(self.get_state(obs), choices))
        act = self.choose_act(obs)
        self.step_reward += min(0, 0.5 - player.minerals / 1000)
        return act
