from typing import Callable

import math
import torch
import numpy as np

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

MAX_WASTE_OVERFLOW = 1.


class WCRPGenerator(Generator):
    """Data generator for the Waste Collection Routing Problem (WCRP).

    Args:
        num_loc: number of locations (bins) in the WCRP, without the depot. (e.g. 10 means 10 locs + 1 depot)
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        loc_distribution: distribution for the location coordinates
        depot_distribution: distribution for the depot location. If None, sample the depot from the locations
        min_prize: minimum value for the prize of each bin
        max_prize: maximum value for the prize of each bin
        prize_distribution: distribution for the prize of each bin
        amount_overflow: maximum amount of waste (prize) before bin (node) overflows

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each bin
            depot [batch_size, 2]: location of the depot
            prize [batch_size, num_loc]: prize of each bin
            amount_overflow [batch_size, 1]: maximum amount of waste before each bin overflows
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: int | float | str | type | Callable = Uniform,
        depot_distribution: int | float | str | type | Callable = None,
        min_prize: float = 0.0,
        max_prize: float = 1.0,
        prize_distribution: int | float | type | Callable = Uniform,
        prize_type: str = "dist",
        amount_overflow: float | torch.Tensor = None,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_prize = min_prize
        self.max_prize = max_prize
        self.prize_type = prize_type
        self.amount_overflow = amount_overflow

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

        # Depot distribution
        if kwargs.get("depot_sampler", None) is not None:
            self.depot_sampler = kwargs["depot_sampler"]
        else:
            self.depot_sampler = (
                get_sampler("depot", depot_distribution, min_loc, max_loc, **kwargs)
                if depot_distribution is not None
                else None
            )

        # Prize distribution
        if kwargs.get("prize_sampler", None) is not None:
            self.prize_sampler = kwargs["prize_sampler"]
        elif (
            prize_distribution == "dist"
        ):  # If prize_distribution is 'dist', then the prize is the distance from the depot
            self.prize_sampler = None
        else:
            self.prize_sampler = get_sampler(
                "prize", prize_distribution, min_prize, max_prize, **kwargs
            )

        # Max waste amount before overflow
        if amount_overflow is not None:
            self.amount_overflow = amount_overflow
        else:
            self.amount_overflow = MAX_WASTE_OVERFLOW

    def _generate(self, batch_size) -> TensorDict:
        def __set_gamma_dist(size, th, k):
            th = th * math.ceil(size / 2)
            if size % 2 == 1:
                th = th[:-1]

            k = k * math.ceil(size / 10)
            if size % 10 != 0:
                k = k[:len(k)-size%10]
            return th, k

        # Sample locations: depot and bins
        if self.depot_sampler is not None:
            depot = self.depot_sampler.sample((*batch_size, 2))
            locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))
        else:
            # if depot_sampler is None, sample the depot from the locations
            locs = self.loc_sampler.sample((*batch_size, self.num_loc + 1, 2))
            depot = locs[..., 0, :]
            locs = locs[..., 1:, :]

        locs_with_depot = torch.cat((depot.unsqueeze(1), locs), dim=1)

        # Methods taken from Fischetti et al. (1998) and Kool et al. (2019)
        if self.prize_type == "const":
            prize = torch.ones(*batch_size, self.num_loc, device=self.device)
        elif self.prize_type == "unif":
            prize = (
                1
                + torch.randint(
                    0, 100, (*batch_size, self.num_loc), device=self.device
                ).float()
            ) / 100
        elif self.prize_type == "dist":  # based on the distance to the depot
            prize = (locs_with_depot[..., 0:1, :] - locs_with_depot[..., 1:, :]).norm(
                p=2, dim=-1
            )
            prize = (
                1 + (prize / prize.max(dim=-1, keepdim=True)[0] * 99).int()
            ).float() / 100
        elif self.prize_type == "empty":
            prize = torch.zeros(*batch_size, self.num_loc, device=self.device)
        elif self.prize_type == "gamma":
            th, k = __set_gamma_dist(self.num_loc, th=[5, 2], k=[5, 5, 5, 5, 5, 10, 10, 10, 10, 10])
            prize = torch.from_numpy(np.random.gamma(k, th, size=(*batch_size, self.num_loc))) / 100
        else:
            raise ValueError(f"Invalid prize_type: {self.prize_type}")
        
        # Daily waste filling
        th, k = __set_gamma_dist(self.num_loc, th=[5, 2], k=[5, 5, 5, 5, 5, 10, 10, 10, 10, 10])
        prize += torch.from_numpy(np.random.gamma(k, th, size=(*batch_size, self.num_loc))) / 100

        # Support for heterogeneous max amount of waste before overflow if provided
        if not isinstance(self.amount_overflow, torch.Tensor):
            amount_overflow = torch.full((*batch_size,), self.amount_overflow)
        else:
            amount_overflow = self.amount_overflow

        return TensorDict(
            {
                "locs": locs_with_depot[..., 1:, :],
                "depot": locs_with_depot[..., 0, :],
                "prize": prize,
                "amount_overflow": amount_overflow,
            },
            batch_size=batch_size,
        )
