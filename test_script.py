import sys
import argparse

from absl import flags
from pysc2.env import sc2_env

from marine_agent import ScriptBMAgent, RandomBMAgent

FLAGS = flags.FLAGS
FLAGS(sys.argv[:1])

parser = argparse.ArgumentParser(description='Scripted agent for SC2 BuildMarines',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--type', choices=['scripted', 'random'], default='scripted', help='agent type')
parser.add_argument('--max-scv', type=int, default=30, help='max number of SCVs')
parser.add_argument('--max-depot', type=int, default=5, help='max number of supply depots')
parser.add_argument('--max-barracks', type=int, default=10, help='max number of barracks')
parser.add_argument('--render', type=int, default=1, help='whether render')
parser.add_argument('--test-epoch', type=int, default=1, help='number of test epochs')
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
agent = ScriptBMAgent(args.max_scv, args.max_depot, args.max_barracks) \
    if args.type == 'scripted' else RandomBMAgent()
observation_spec = env.observation_spec()[0]
action_spec = env.action_spec()[0]
agent.setup(observation_spec, action_spec)

for i in range(args.test_epoch):
    obs = env.reset()[0]
    if obs.observation.player.idle_worker_count > 0:
        print('A bug happened! Mineral missing! Restart enviroment')
        env.close()
        env = make_env()
        obs = env.reset()[0]
    agent.reset()

    while not obs.last():
        action = agent.step(obs)
        obs = env.step(actions=[action])[0]
    print('#' * 60)
    print(f'## Epoch: {agent.episodes} | Score: {agent.reward}'.ljust(58) + '##')
    print('#' * 60 + '\n')

env.close()
