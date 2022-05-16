from copy import deepcopy
from pathlib import Path

import numpy as np


from ai_economist.foundation.base.base_env import scenario_registry
from ai_economist.foundation.scenarios.simple_wood_and_stone.layout_from_file import LayoutFromFile
from ai_economist.foundation.scenarios.utils import rewards, social_metrics


def isoelastic_coin_minus_nonlinear_labor(
    coin_endowment, total_labor, isoelastic_eta, labor_coefficient
):
    """Agent utility, concave increasing in coin and linearly decreasing in labor.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        total_labor (float, ndarray): The amount of labor performed by the agent(s).
        isoelastic_eta (float): Constant describing the shape of the utility profile
            with respect to coin endowment. Must be between 0 and 1. 0 yields utility
            that increases linearly with coin. 1 yields utility that increases with
            log(coin). Utility from coin uses:
                https://en.wikipedia.org/wiki/Isoelastic_utility
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor

    Returns:
        Agent utility (float) or utilities (ndarray).
    """
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    assert np.all(coin_endowment >= 0)
    assert 0 <= isoelastic_eta <= 1.0

    # Utility from coin endowment
    if isoelastic_eta == 1.0:  # dangerous
        util_c = np.log(np.max(1, coin_endowment))
    else:  # isoelastic_eta >= 0
        util_c = (coin_endowment ** (1 - isoelastic_eta) - 1) / (1 - isoelastic_eta)

    # disutility from labor
    util_l = total_labor * labor_coefficient
    util_l = np.exp(0.01 * util_l) + util_l

    # Net utility
    util = util_c - util_l

    return util

@scenario_registry.add
class NonLinearDisutility(LayoutFromFile):
    """
    Almost the same world as LayoutFromFile. The only difference is that labor increases
    non-linearly.
    """

    name = "nonlinear_disutility"

    def get_current_optimization_metrics(self):
        """
        Compute optimization metrics based on the current state. Used to compute reward.

        Returns:
            curr_optimization_metric (dict): A dictionary of {agent.idx: metric}
                with an entry for each agent (including the planner) in the env.
        """
        curr_optimization_metric = {}
        # (for agents)
        for agent in self.world.agents:
            curr_optimization_metric[agent.idx] = isoelastic_coin_minus_nonlinear_labor(
                coin_endowment=agent.total_endowment("Coin"),
                total_labor=agent.state["endogenous"]["Labor"],
                isoelastic_eta=self.isoelastic_eta,
                labor_coefficient=self.energy_weight * self.energy_cost,
            )
        # (for the planner)
        if self.planner_reward_type == "coin_eq_times_productivity":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.coin_eq_times_productivity(
                coin_endowments=np.array(
                    [agent.total_endowment("Coin") for agent in self.world.agents]
                ),
                equality_weight=1 - self.mixing_weight_gini_vs_coin,
            )
        elif self.planner_reward_type == "inv_income_weighted_coin_endowments":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.inv_income_weighted_coin_endowments(
                coin_endowments=np.array(
                    [agent.total_endowment("Coin") for agent in self.world.agents]
                )
            )
        elif self.planner_reward_type == "inv_income_weighted_utility":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.inv_income_weighted_utility(
                coin_endowments=np.array(
                    [agent.total_endowment("Coin") for agent in self.world.agents]
                ),
                utilities=np.array(
                    [curr_optimization_metric[agent.idx] for agent in self.world.agents]
                ),
            )
        else:
            print("No valid planner reward selected!")
            raise NotImplementedError
        return curr_optimization_metric
