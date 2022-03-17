from overcooked_ai_py.agents.agent import Agent, AgentPair, RandomAgent, GreedyHumanModel
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, Recipe
from overcooked_ai_py.mdp.actions import Action
#// from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
#// from overcooked_ai_py.agents.benchmarking import LayoutGenerator
from human_aware_rl.rllib.rllib import RlLibAgent, load_agent_pair


class MainAgent(Agent):

    #// mdp_gen_params = {"layout_name": 'leaderboard_collaborative'}
    #// mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
    #// env_params = {"horizon": 1400}
    #// env = OvercookedEnv(mdp_fn, **env_params)

    def __init__(self):
        super().__init__()
        self.agent = RandomAgent()
        #// self.agent = GreedyHumanModel(self.env.mlam)

    def action(self, state):
        # Implement your logic here
        action, action_probs = self.agent.action(state)
        return action, action_probs


class OptionalAgent(Agent):

    def __init__(self):
        super().__init__()
        
    def action(self, state):
        # Implement your logic here
        action, action_probs = Action.STAY, {}
        return action, action_probs


class HacktrickAgent(object):
    # Enable this flag if you are using reinforcement learning from the included ppo ray support library
    RL = False
    # Rplace with the directory for the trained agent
    # Note that `agent_dir` is the full path to the checkpoint FILE, not the checkpoint directory
    agent_dir = ''
    # If you do not plan to use the same agent logic for both agents and use the OptionalAgent set it to False
    # Does not matter if you are using RL as this is controlled by the RL agent
    share_agent_logic = True

    def __init__(self):
        Recipe.configure({})

        if self.RL:
            pass
            agent_pair = load_agent_pair(self.agent_dir)
            self.agent0 = agent_pair.a0
            self.agent1 = agent_pair.a1
        else:
            self.agent0 = MainAgent()#//.agent
            self.agent1 = OptionalAgent()
    
    def set_mode(self, mode):
        self.mode = mode

        if "collaborative" in self.mode:
            if self.share_agent_logic and not self.RL:
                self.agent1 = MainAgent()#//.agent
            self.agent_pair = AgentPair(self.agent0, self.agent1)
        else:
            self.agent1 =None
            self.agent_pair =None
    
    def map_action(self, action):
        action_map = {(0, 0): 'STAY', (0, -1): 'UP', (0, 1): 'DOWN', (1, 0): 'RIGHT', (-1, 0): 'LEFT', 'interact': 'SPACE'}
        action_str = action_map[action[0]]
        return action_str

    def action(self, state_dict):
        state = OvercookedState.from_dict(state_dict['state']['state'])

        if "collaborative" in self.mode:
            (action0, action1) = self.agent_pair.joint_action(state)
            action0 = self.map_action(action0)
            action1 = self.map_action(action1)
            action = [action0, action1]
        else:
            action0 = self.agent0.action(state)
            action0 = self.map_action(action0)
            action = action0

        return action