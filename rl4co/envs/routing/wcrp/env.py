from typing import Optional

import torch
import torch.nn.functional as F

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger

from .generator import WCRPGenerator
from .render import render

log = get_pylogger(__name__)


class WCRPEnv(RL4COEnvBase):
    """Waste Collection Routing Problem (WCRP) environment.
    At each step, the agent chooses a location to visit in order to maximize the collected prize.
    The total length of the path must not exceed a given threshold.

    Observations:
        - location of the depot
        - locations and prize of each bin
        - current location of the vehicle
        - current tour length
        - current total prize
        - the remaining length of the path

    Constraints:
        - the tour starts and ends at the depot
        - not all bins need to be visited

    Finish Condition:
        - the vehicle back to the depot

    Reward:
        - the sum of the prizes of visited nodes

    Args:
        generator: WCRPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "wcrp"

    def __init__(
        self,
        generator: WCRPGenerator = None,
        generator_params: dict = {},
        prize_type: str = "dist",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = WCRPGenerator(**generator_params)
        self.generator = generator
        self.prize_type = prize_type
        assert self.prize_type in [
            "dist",
            "unif",
            "const",
        ], f"Invalid prize_type: {self.prize_type}"
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"][:, None]

        # Update tour length
        previus_loc = gather_by_index(td["locs"], td["current_node"])
        current_loc = gather_by_index(td["locs"], current_node)
        tour_length = td["tour_length"] + (current_loc - previus_loc).norm(p=2, dim=-1)

        # Update prize with collected prize
        current_total_prize = td["current_total_prize"] + gather_by_index(
            td["prize"], current_node, dim=-1
        )

        # If collected bin(s) was(re) overflowing, then reduce number of overflows
        current_number_overflows = td["current_number_overflows"] - torch.sum((td["prize"][:, td["action"]] > td["amount_overflow"]))

        # Set current node as visited
        visited = td["visited"].scatter(-1, current_node, 1)

        # Done if went back to depot (except if it's the first step, since we start at the depot)
        done = (current_node.squeeze(-1) == 0) & (td["i"] > 0)

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "tour_length": tour_length,
                "current_node": current_node,
                "visited": visited,
                "current_total_prize": current_total_prize,
                "current_number_overflows": current_number_overflows,
                "i": td["i"] + 1,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        device = td.device

        # Add depot to locs
        locs_with_depot = torch.cat((td["depot"][:, None, :], td["locs"]), -2)

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": locs_with_depot,
                "prize": F.pad(
                    td["prize"], (1, 0), mode="constant", value=0
                ),  # add 0 for depot
                "tour_length": torch.zeros(*batch_size, device=device),
                "amount_overflow": torch.full(
                    (*batch_size, 1), self.generator.amount_overflow, device=device
                ),
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "visited": torch.zeros(
                    (*batch_size, locs_with_depot.shape[-2]),
                    dtype=torch.bool,
                    device=device,
                ),
                "current_total_prize": torch.zeros(
                    *batch_size, dtype=torch.float, device=device
                ),
                "current_number_overflows": torch.sum((td["prize"] > td["amount_overflow"][:, None]), dim=-1),
                "i": torch.zeros(
                    (*batch_size,), dtype=torch.int64, device=device
                ),  # counter
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """Get action mask with 1 = feasible action, 0 = infeasible action.
        Cannot visit if already visited, if depot has been visited, or if the length exceeds the maximum length.
        """
        mask = td["visited"] | td["visited"][..., 0:1]

        action_mask = ~mask  # 1 = feasible action, 0 = infeasible action

        # Depot can always be visited: we do not hardcode knowledge that this is strictly suboptimal if other options are available
        action_mask[..., 0] = 1
        return action_mask

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Reward is the sum of the prizes of visited nodes"""
        # In case all tours directly return to depot, prevent further problems
        if actions.size(-1) == 1:
            assert (actions == 0).all(), "If all length 1 tours, they should be zero"
            return torch.zeros(actions.size(0), dtype=torch.float, device=actions.device)

        # Prize is the sum of the prizes of the visited nodes. Note that prize is padded with 0 for depot at index 0
        collected_prize = td["prize"].gather(1, actions)

        # Gather locations in order of tour (add depot since we start and end there)
        locs_ordered = torch.cat(
            [
                td["locs"][..., 0:1, :],  # depot
                gather_by_index(td["locs"], actions),  # order locations
            ],
            dim=1,
        )

        # Mask for unvisited bins (excluding the depot)
        unvisited_mask = ~td["visited"]

        # Apply the unvisited mask and check for overflows
        overflow_mask = td["prize"] > td["amount_overflow"]
        unvisited_overflows = overflow_mask & unvisited_mask

        # Compute the number of unvisited bins with overflows
        num_unvisited_overflows = torch.sum(unvisited_overflows, dim=-1)

        return collected_prize.sum(-1) - get_tour_length(locs_ordered) - num_unvisited_overflows

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        """Check that solution is valid: nodes are not visited twice except depot.
        """
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_actions = actions.data.sort(1)[0]
        # Make sure each node visited once at most (except for depot)
        assert (
            (sorted_actions[:, 1:] == 0)
            | (sorted_actions[:, 1:] > sorted_actions[:, :-1])
        ).all(), "Duplicates"

    def _make_spec(self, generator: WCRPGenerator):
        """Make the observation and action specs from the parameters."""
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc + 1, 2),
                dtype=torch.float32,
            ),
            current_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            prize=Unbounded(
                shape=(generator.num_loc,),
                dtype=torch.float32,
            ),
            tour_length=Unbounded(
                shape=(generator.num_loc,),
                dtype=torch.float32,
            ),
            visited=Unbounded(
                shape=(generator.num_loc + 1,),
                dtype=torch.bool,
            ),
            amount_overflow=Unbounded(
                shape=(1,),
                dtype=torch.int64,
            ),
            action_mask=Unbounded(
                shape=(generator.num_loc + 1, 1),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc + 1,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)
