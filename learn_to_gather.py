import numpy as np

from ai_economist.foundation.base.base_component import (
    component_registry,
    BaseComponent
)


@component_registry.add
class MoveGather(BaseComponent):
    """
    Modifies the Gather component to allow skill to increase when gathering.
    """

    name = "MoveGather"
    required_entities = ["Coin", "House", "Labor"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
            self,
            *base_component_args,
            move_labor=1.0,
            collect_labor=1.0,
            initial_skill=1.0,
            skill_increment=0.2,
            **base_component_kwargs
    ):
        super().__init__(
            *base_component_args,
            **base_component_kwargs
        )

        self.move_labor = 1.0
        self.collect_labor = 1.0
        self.initial_skill = 1.0
        self.skill_increment = 1.0
        
        self.gathers = []

        self._aidx = np.arange(self.n_agents)[:, None].repeat(4, axis=1)
        self._roff = np.array([[0, 0, -1, 1]])
        self._coff = np.array([[-1, 1, 0, 0]])

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        Adds 5 actions (move in the four directions + gather) for mobile agents.
        Action meaning:
          0: no-op
          1: move left
          2: move right
          3: move up
          4: move right
          5: gather
        """
        if agent_cls_name == "BasicMobileAgent":
            return 5
        else:
            return None

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        For mobile agents, add state for collection skill.
        """
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicMobileAgent":
            return {"gather_skill": 0.0}
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.

        Try to move to adjacent locations if the action is move. Otherwise
        try to gather to collect resources.
        """
        world = self.world

        gathers = []

        for agent in world.get_random_order_agents():
            # return if this action is not available to the agent
            if self.name not in agent.action:
                return

            action = agent.get_component_action(self.name)

            # get coords of agent
            r, c = [int(x) for x in agent.loc]

            # no-op
            if action == 0:
                # don't move
                new_r, new_c = r, c

            elif 0 < action and action <= 4:
                if action == 1:  # move left
                    new_r, new_c = r, c - 1
                elif action == 2:  # move right
                    new_r, new_c = r, c + 1
                elif action == 3:  # move up
                    new_r, new_c = r - 1, c
                elif action == 4:  # move down
                    new_r, new_c = r + 1, c

                # attempt to move. fails if the new pos is not accessible
                new_r, new_c = world.set_agent_loc(agent, new_r, new_c)

                # charge the agent for moving
                if (new_r != r) or (new_c != c):
                    agent.state["endogenous"]["Labor"] += self.move_labor

            elif action == 5:
                r, c = [int(x) for x in agent.loc]
                for resource, health in world.location_resources(r, c).items():
                    if health >= 1:
                        # sample from from a normal distribution with mean = agent skill
                        n_gathered = np.maximum(
                            0.0,
                            np.round(np.random.normal(agent.state["gather_skill"]))
                        )
                        agent.state["inventory"][resource] += int(n_gathered)
                        world.consume_resource(resource, r, c)
                        # Incur the labor cost of collecting a resource
                        agent.state["endogenous"]["Labor"] += self.collect_labor

                        # increase agent skill
                        agent.state["gather_skill"] += self.skill_increment
                        
                        # Log the gather
                        gathers.append(
                            dict(
                                agent=agent.idx,
                                resource=resource,
                                n=n_gathered,
                                loc=agent.loc,
                            )
                        )
            else:
                raise ValueError

        self.gathers.append(gathers)
                

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Agents observe their own collection skill. The planner does not observe
        antyhing.
        """
        return {
            str(agent.idx): {"gather_skill": agent.state["gather_skill"]}
            for agent in self.world.agents
        }

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Prevent moving to adjacent tiles that are already occupied (or outside the
        boundaries of the world)
        """
        world = self.world

        coords = np.array([agent.loc for agent in world.agents])[:, :, None]
        ris = coords[:, 0] + self._roff + 1
        cis = coords[:, 1] + self._coff + 1

        occ = np.pad(world.maps.unoccupied, ((1, 1), (1, 1)))
        acc = np.pad(world.maps.accessibility, ((0, 0), (1, 1), (1, 1)))
        mask_array = np.logical_and(occ[ris, cis], acc[self._aidx, ris, cis]).astype(
            np.float32
        )

        res_array = np.array([
            len(world.location_resources(agent.loc[0], agent.loc[1])) > 0
            for agent in world.agents
        ])
        masks = {
            agent.idx: np.append(mask_array[i], res_array[i])
            for i, agent in enumerate(world.agents)
        }

        return masks

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Reset agents' collection skills.
        """
        for agent in self.world.agents:
            agent.state["gather_skill"] = self.initial_skill

        self.gathers = []

    def get_dense_log(self):
        return self.gathers
