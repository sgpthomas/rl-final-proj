import numpy as np

from ai_economist.foundation.base.base_env import scenario_registry
from ai_economist.foundation.scenarios.utils import rewards, social_metrics
from ai_economist.foundation.scenarios.simple_wood_and_stone.layout_from_file import LayoutFromFile


@scenario_registry.add
class InitialCoin(LayoutFromFile):

    name = "initial_coin"

    def __init__(
            self,
            *base_env_args,
            initial_coin_distribution=None,
            **base_env_kwargs,
    ):
        super().__init__(*base_env_args, **base_env_kwargs)

        if initial_coin_distribution is None:
            self.coin_distribution = {
                agent.idx: 0 for agent in self.world.agents
            }
        else:
            assert type(initial_coin_distribution) == list
            self.coin_distribution = {
                agent.idx: float(initial_coin_distribution[int(agent.idx)])
                for agent in self.world.agents
            }

    def reset_agent_states(self):
        """
        Part 2/2 of scenario reset. This method handles resetting the state of the
        agents themselves (i.e. inventory, locations, etc.).

        Here, empty inventories and place mobile agents in random, accessible
        locations to start. Note: If using fixed_four_skill_and_loc, the starting
        locations will be overridden in self.additional_reset_steps.
        """
        self.world.clear_agent_locs()
        for agent in self.world.agents:
            agent.state["inventory"] = {k: 0 for k in agent.inventory.keys()}
            agent.state["escrow"] = {k: 0 for k in agent.inventory.keys()}
            agent.state["endogenous"] = {k: 0 for k in agent.endogenous.keys()}
            # Add starting coin
            agent.state["inventory"]["Coin"] = self.coin_distribution[agent.idx]

        self.world.planner.state["inventory"] = {
            k: 0 for k in self.world.planner.inventory.keys()
        }
        self.world.planner.state["escrow"] = {
            k: 0 for k in self.world.planner.escrow.keys()
        }

        for agent in self.world.agents:
            r = np.random.randint(0, self.world_size[0])
            c = np.random.randint(0, self.world_size[1])
            n_tries = 0
            while not self.world.can_agent_occupy(r, c, agent):
                r = np.random.randint(0, self.world_size[0])
                c = np.random.randint(0, self.world_size[1])
                n_tries += 1
                if n_tries > 200:
                    raise TimeoutError
            r, c = self.world.set_agent_loc(agent, r, c)
