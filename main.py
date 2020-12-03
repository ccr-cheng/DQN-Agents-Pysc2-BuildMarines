import sys
import argparse

import torch
from absl import flags
from pysc2.env import sc2_env

from dqn_agent import DQNBMAgent, BMAction

FLAGS = flags.FLAGS
FLAGS(sys.argv[:1])

parser = argparse.ArgumentParser(description='DQN for SC2 BuildMarines',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', choices=['train', 'test'], default='train', help='running mode')
# model parameters
parser.add_argument('--hidden-size', type=int, default=256, help='hidden size')
parser.add_argument('--memory-size', type=int, default=10000, help='size of replay memory')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--eps-start', type=float, default=0.9, help='eps start')
parser.add_argument('--eps-decay', type=float, default=200, help='eps decay step')
parser.add_argument('--eps-end', type=float, default=0.05, help='eps end')
parser.add_argument('--clip-grad', type=float, default=1.0, help='clipping threshold')
# training parameters
parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
parser.add_argument('--num-epoch', type=int, default=200, help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--log-step', type=int, default=100, help='logging print step')
parser.add_argument('--update-tgt', type=int, default=1, help='update target net')
parser.add_argument('--render', type=int, default=1, help='whether render')
# saving & checkpoint
parser.add_argument('--save-path', type=str, default='model.pt', help='model path for saving')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint for resuming training')
parser.add_argument('--model-path', type=str, default='model.pt', help='model path for evaluation')
parser.add_argument('--test-epoch', type=int, default=3, help='number of test epochs')
args = parser.parse_args()


def make_env():
    return sc2_env.SC2Env(
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(
                screen=84,
                minimap=32,
            )
        ),
        map_name='BuildMarines',
        step_mul=8,
        visualize=args.render,
    )


env = make_env()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on {device.type}')

agent = DQNBMAgent(args.hidden_size, device, lr=args.lr, batch_size=args.batch_size,
                   memory_size=args.memory_size, gamma=args.gamma, clip_grad=args.clip_grad)
observation_spec = env.observation_spec()[0]
action_spec = env.action_spec()[0]
agent.setup(observation_spec, action_spec)


def train():
    global env
    if args.checkpoint != '':
        print(f'Loading model from {args.checkpoint}')
        agent.load(args.checkpoint)

    cnt = 0
    hist_reward, hist_loss = [], []

    def write_results():
        with open('reward.txt', 'w') as f_r:
            f_r.write(' '.join(map(str, hist_reward)))
        with open('loss.txt', 'w') as f_l:
            f_l.write(' '.join(map(str, hist_loss)))

    for idx in range(args.num_epoch):
        obs = env.reset()[0]
        if obs.observation.player.idle_worker_count > 0:
            # bug happens!
            print('A bug happened! Mineral missing! Restart enviroment')
            env.close()
            env = make_env()
            obs = env.reset()[0]
        agent.reset()

        # initiate
        while agent.in_progress == -1:
            action = agent.step(obs)
            obs = env.step(actions=[action])[0]

        start_state = torch.tensor(agent.get_state(obs), dtype=torch.float, device=device)
        while not obs.last():
            # here action_idx is the previous action
            # after step the `in_progress` updates
            # 0 indicates the end of a sequence
            action_idx = agent.in_progress.value
            action = agent.step(obs)
            obs = env.step(actions=[action])[0]
            reward = agent.step_reward
            if agent.in_progress != BMAction.NO_OP:
                continue

            reward = torch.tensor(reward, dtype=torch.float, device=device)
            action_idx = torch.tensor(action_idx, dtype=torch.long, device=device)
            state = torch.tensor(agent.get_state(obs), dtype=torch.float, device=device)
            agent.cache.push([start_state, action_idx, reward, state])
            start_state = state

            loss = agent.update_act()
            if loss is not None:
                cnt += 1
                hist_loss.append(loss)
                if cnt % args.log_step == 0:
                    print(f'Epoch {idx} | iter {cnt}, loss: {loss:.3f}')

        print('#' * 60)
        print(f'## Epoch: {agent.episodes} | Score: {agent.reward}'.ljust(58) + '##')
        print('#' * 60 + '\n')

        if idx % args.update_tgt == 0:
            agent.update_tgt()
            agent.save(args.save_path)

    # write_results()


def evaluate(n_test=3):
    global env
    print(f'Loading model from {args.model_path}')
    agent.load(args.model_path)
    agent.act_net.eval()

    scores = []
    for i in range(n_test):
        obs = env.reset()[0]
        if obs.observation.player.idle_worker_count > 0:
            print('A bug happened! Mineral missing! Restart enviroment')
            env.close()
            env = make_env()
            obs = env.reset()[0]
        agent.reset()

        # initiate
        while agent.in_progress == -1:
            action = agent.step(obs)
            obs = env.step(actions=[action])[0]

        while not obs.last():
            action = agent.step(obs)
            obs = env.step(actions=[action])[0]
            if agent.in_progress != BMAction.NO_OP:
                continue

        scores.append(agent.reward)
        print('#' * 60)
        print(f'## Epoch: {agent.episodes} | Score: {agent.reward}'.ljust(58) + '##')
        print('#' * 60 + '\n')

    print(f'\nAverage score: {sum(scores) / len(scores):.1f}')


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        evaluate(args.test_epoch)
    env.close()
