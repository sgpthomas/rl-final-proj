import numpy as np
from ai_economist.foundation.base.base_component import (
    component_registry
)
from ai_economist.foundation.components.build import Build


@component_registry.add
class LearnToBuild(Build):
    """
    Modifies the Build component to allow skill to increase when building houses.
    """

    name = "LearnToBuild"

    def __init__(
            self,
            *base_component_args,
            skill_increment=0.01,
            initial_skill=0.05,
            **base_component_kwargs,
    ):
        super().__init__(
            *base_component_args,
            skill_dist="none",
            **base_component_kwargs
        )

        self.skill_increment = skill_increment
        self.initial_skill = initial_skill


    def _compute_pay_rate(self, agent):
        PMSM = self.payment_max_skill_multiplier
        return np.minimum(
            PMSM,
            (PMSM - 1.0) * agent.state["build_skill"] + 1.0
        )

        
    def component_step(self):
        """
        See base_component.py for detailed description.

        Override Build.component_step to increase skill and
        recomputed payrate everytime a build is successful.
        """
        world = self.world
        build = []
        # Apply any building actions taken by the mobile agents
        for agent in world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            # This component doesn't apply to this agent!
            if action is None:
                continue

            # NO-OP!
            if action == 0:
                pass

            # Build! (If you can.)
            elif action == 1:
                if self.agent_can_build(agent):
                    # Remove the resources
                    for resource, cost in self.resource_cost.items():
                        agent.state["inventory"][resource] -= cost

                    # Place a house where the agent is standing
                    loc_r, loc_c = agent.loc
                    world.create_landmark("House", loc_r, loc_c, agent.idx)

                    # Receive payment for the house
                    agent.state["inventory"]["Coin"] += agent.state["build_payment"]

                    # Incur the labor cost for building
                    agent.state["endogenous"]["Labor"] += self.build_labor

                    # increase build skill
                    agent.state["build_skill"] += self.skill_increment

                    # increase payrate based on build_skill
                    pay_rate = self._compute_pay_rate(agent)
                    agent.state["build_payment"] = pay_rate * self.payment

                    build.append(
                        {
                            "builder": agent.idx,
                            "loc": np.array(agent.loc),
                            "income": float(agent.state["build_payment"]),
                            "new_skill": agent.state["build_skill"],
                            "new_payrate": pay_rate
                        }
                    )

            else:
                raise ValueError

        self.builds.append(build)

    def additional_reset_steps(self):
        """
        Override Build's default reset behavior. We are not using a skill distribution,
        so just set the build skill to the initial skill.
        """
        world = self.world

        self.sampled_skills = {agent.idx: 1 for agent in world.agents}

        for agent in world.agents:
            agent.state["build_skill"] = float(self.initial_skill)
            pay_rate = self._compute_pay_rate(agent)
            agent.state["build_payment"] = float(pay_rate * self.payment)

            self.sampled_skills[agent.idx] = float(self.initial_skill)

        self.builds = []
