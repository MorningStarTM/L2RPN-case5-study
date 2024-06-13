from tqdm.notebook import tqdm
import numpy as np
from agents import ACAgent, KANACAgent, PPOAgent, KANPPOAgent, DQNAgent
from agents import Trainer, PPOTrainer, QTrainer
import gym
from .converter import Converter
from lightsim2grid import LightSimBackend
from grid2op.Action import TopologyChangeAction
import grid2op




class Evaluator:
    def __init__(self, env_name:str, agent_name:str):
        self.env_name = env_name
        self.agent_name = agent_name
        self.n_actions = 132 if self.env_name=="rte_case5_example" else None
        self.input_dims = 182 if self.env_name == "rte_case5_example" else None
        

    def eval(self):
        env = grid2op.make(self.env_name, test=True, action_class=TopologyChangeAction)
        converter = Converter(env)

        agent = None
        if self.agent_name == "ppo":
            agent = PPOAgent(n_actions=self.n_actions, input_dims=self.input_dims)
            agent.load_models()
        elif self.agent_name == "kanppo":
            agent = KANPPOAgent(n_actions=self.n_actions, input_dims=self.input_dims)
            agent.load_models()



        all_obs = []
        obs = env.reset()
        all_obs.append(obs)
        reward = env.reward_range[0]
        reward_list = []
        done = False
        nb_step = 0
        print(f"{agent.name} PPO Simulation")
        with tqdm(total=env.chronics_handler.max_timestep()) as pbar:
            while True:
                action, _, _ = agent.choose_action(obs.to_vect())
                #action = my_agent.act(obs, reward, done)
                observation_, reward, done, info = env.step(converter.convert_one_hot_encoding_act_to_env_act(converter.int_to_onehot(action)))
                reward_list.append(reward)
                pbar.update(1)
                if done:
                    break
                all_obs.append(obs)
                nb_step += 1

        #reward_list_simple_DQN = np.copy(reward_list)