# Scripted and DQN Agents for Pysc2 `BuildMarines` Game

——by ccr

2nd Dec. 2020



## Introduction

This repository provides both a scripted agent and a DQN agent for the Pysc2 `BuildMarines` minigame, the former with hard rules encoded, the latter trained for hundreds of epochs.

The major idea is to reduce the dimensionality of the action space. In this work we abstract all actions into 6 scenarios:

- **No operation**.

- **Make an SCV**, which is 

- - Select the command center, and
  - Train an SCV.

- **Build a supply depot**, which is 

- - Select an SCV, 
  - Build a supply depot at a *predefined* location, and
  - Queue the SCV back to mining, so that we do not need to know when the building finishes.

- **Build a barracks**, which follows basically the same procedure as above.

- **Make marines**, which is 

- - Select *all* barracks, and
  - Building marines.

- **Kill marines**, which is 

- - Select all marines (select the army), and
  - Attack a specific point so that eventually some of them will be killed.

The last action is beneficial because we may save the food supply without hurting the scores.

All scripted agents and RL agents only need to choose between the above 6 actions so it will be much easier.



## Running

### Scripted Agent

The scripted agent incorporates specific rules to take action. It has a limitation on the number of SCVs, the number of supply depots, and the number of barracks.

To test the scripted agent, run

```bash
python -m pysc2.bin.agent --map BuildMarines --agent marine_agent.ScriptBMAgent
```

Or

```bash
python test_script.py
```

This scripted agent (with default limit settings) generally receives a score of **160** for each epoch.

Also, a random scripted agent is provided as another baseline. It randomly chooses from all available actions defined above (instead of full randomness). This agent will typically build all possible supply depots before building barracks because supply depots are cheaper.

To test the random scripted agent, run

```bash
python -m pysc2.bin.agent --map BuildMarines --agent marine_agent.RandomBMAgent
```

Or

```bash
python test_script.py --type random
```

This random scripted agent generally receives a score of **100±10** for each epoch.

You can also run `test_script.py` with your own arguments for the limit of units for a scripted agent. Full arguments are:

```bash
usage: test_script.py [-h] [--type {scripted,random}] [--max-scv MAX_SCV]
                      [--max-depot MAX_DEPOT] [--max-barracks MAX_BARRACKS]
                      [--render RENDER] [--test-epoch TEST_EPOCH]
                      
Scripted agent for SC2 BuildMarines

optional arguments:
  -h, --help                show this help message and exit
  --type {scripted,random}  agent type (default: scripted)
  --max-scv                 max number of SCVs (default: 30)
  --max-depot               max number of supply depots (default: 5)
  --max-barracks            max number of barracks (default: 10)
  --render                  whether render (default: 1)
  --test-epoch              number of test epochs (default: 1)
```

### DQN Agent

To train the DQN agent, run

```bash
python main.py
```

To test the DQN agent with the provided fine-tuned model, run

```bash
python main.py --mode test --model-path fine_tune.pt
```

The full arguments are as following:

```bash
usage: main.py [-h] [--mode {train,test}] [--hidden-size HIDDEN_SIZE]
               [--memory-size MEMORY_SIZE] [--gamma GAMMA]
               [--eps-start EPS_START] [--eps-decay EPS_DECAY]
               [--eps-end EPS_END] [--clip-grad CLIP_GRAD] [--lr LR]
               [--num-epoch NUM_EPOCH] [--batch-size BATCH_SIZE]
               [--log-step LOG_STEP] [--update-tgt UPDATE_TGT]
               [--render RENDER] [--save-path SAVE_PATH]
               [--checkpoint CHECKPOINT] [--model-path MODEL_PATH]
               [--test-epoch TEST_EPOCH]

DQN for SC2 BuildMarines

optional arguments:
  -h, --help            show this help message and exit
  --mode {train,test}   running mode (default: train)
  --hidden-size         hidden size (default: 256)
  --memory-size         size of replay memory (default: 10000)
  --gamma               discount factor (default: 0.99)
  --eps-start           eps start (default: 0.9)
  --eps-decay           eps decay step (default: 200)
  --eps-end             eps end (default: 0.05)
  --clip-grad           clipping threshold (default: 1.0)
  --lr                  initial learning rate (default: 0.01)
  --num-epoch           number of training epochs (default: 200)
  --batch-size          batch size (default: 64)
  --log-step            logging print step (default: 100)
  --update-tgt          update target net (default: 1)
  --render              whether render (default: 1)
  --save-path           model path for saving (default: model.pt)
  --checkpoint          checkpoint for resuming training (default: )
  --model-path          model path for evaluation (default: model.pt)
  --test-epoch          number of test epochs (default: 3)
```

The testing scores after training for 200 epochs are unstable, but generally **110±30** for each epoch.



## File Structure

### `main.py`

Main code for training and testing DQN agent. See above for details.

### `test_script.py`

Main code for testing scripted agents. See above for details.

### `marine_agent.py`

Implement the

- `BaseBMAgent`, where the 6 actions and checker functions are implemented.
- `ScriptBMAgent`, inherited from `BaseBMAgent`. The scripted strategy is implemented.
- `RandomBMAgent`, inherited from `BaseBMAgent`. It takes random available action.

It also stores game information including

- `BMInfo`, a class that stores locations of minerals, command center, supply depots, etc. Also stores the cost needed to build each units.
- `BMAction`, an enumerator for actions defined above.
- `BMState`, a named-tuple defining the current state information.

### `dqn_model.py`

Implement the DQN network, the replay memory and `DQNAgent` for updating network weights and taking actions.

### `dqn_agent.py`

Implement `DQNBMAgent`, inherited from `BaseBMAgent` and `DQNAgent`.



## DQN Details

In this work, I simply use a one-layer fully-connected neural network as the Q-function approximator. Please refer to `dqn_model.py` for model details.

### State Definition

In order to avoid complex calculation, the state of a frame is defined as a 9-tuple (`BMState` in `marine_agent.py`)

- `n_scv / 10`
- `n_marine / 10`
- `n_depot / 5`
- `n_barracks / 5`
- `mineral / 1000`
- `food_used / 10`
- `food_cap / 10`
- `food_workers / 10`
- `army_count / 10`

The divisors are to normalize raw data to a reasonable range.

### Reward Setting

The original reward is deterred because it takes some time to train marines. In order to force the DQN to learn the pattern, I use customized rewards, which is 

- Receive -0.1 when any of the actions fails in the middle.
- Receive 1 immediately when a making marines action succeeds.
- Receive 0.5 immediately when a making SCV action succeeds.
- Receive -0.1 immediately when a building supply depot action succeeds. This will force the agent to learn to kill marines instead of building many depots.
- Receive 0.1 immediately when a building barracks action succeeds. 
- Receive `(army_count - 3) / 10` when trying to take a killing marines action.
- Receive `min(0, 0.5 - minerals / 1000)` for each action. This will force the agent to spend minerals.

### Performance

The agent performs badly at the beginning phase where a lot of different units shall be built, but generally performs well in the later phases where only making and killing marines actions is needed.