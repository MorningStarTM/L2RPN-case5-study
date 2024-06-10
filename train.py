from agents import ACAgent, KANACAgent, PPOAgent, KANPPOAgent, DQNAgent
from agents import Trainer, PPOTrainer, QTrainer
import gym

from lightsim2grid import LightSimBackend
from grid2op.Action import TopologyChangeAction
import grid2op

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
env_name = "rte_case5_example"  # or any other name.
env = grid2op.make(env_name, test=True, action_class=TopologyChangeAction)


agent = KANPPOAgent(n_actions=132, input_dims=182)
trainer = PPOTrainer(agent=agent, env=env, N=20, n_epochs=50, n_games=100)

trainer.train("result\\ppo-kan.png")
#agentt.save_model("models")