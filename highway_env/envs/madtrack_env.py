from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle


class MadTrackEnv(AbstractEnv):
    """
    A continuous control environment.

    The agent needs to learn two skills:
    - follow the tracks
    - avoid collisions with other vehicles

    Credits and many thanks to @supperted825 for the idea and initial implementation.
    See https://github.com/eleurent/highway-env/issues/231
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "OccupancyGrid",
                    "features": ["presence", "on_road"],
                    "grid_size": [[-0.75, 0.75], [-0.75, 0.75]],
                    "grid_step": [0.125, 0.125],
                    "as_image": False,
                    "align_to_vehicle_axes": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": False,
                    "lateral": True,
                    "dynamical": False,
                    "target_speeds": [0, 0.5, 1.0], #0, 5, 10],
                    "steering_range": [-np.deg2rad(22), np.deg2rad(22) ]
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300,
                "collision_reward": -1,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "action_reward": -0.3,
                "controlled_vehicles": 1,
                "other_vehicles": 0,
                "screen_width": 1024,
                "screen_height": 768,
                "centering_position": [0, 0],
                "scaling": 1024 / 2.7,
                #"offscreen": True,
            }
        )
        return config

    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward = utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        return {
            "lane_centering_reward": 1
            / (1 + self.config["lane_centering_cost"] * lateral**2),
            "action_reward": np.linalg.norm(action),
            "collision_reward": self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed or self.vehicle.on_road==False

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [1.35, 0.9]  # [m]
        radius = 0.6  # [m]
        alpha = 10  # [deg]
        width = 0.17 # [ m ]
        speed_limit = 0.5 # [ m/s ]
        
        radii = [radius, radius + width ]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane(
                "nw",
                "sw",
                CircularLane(
                    [ 0.9, 0.9 ],
                    radii[lane],
                    np.deg2rad(270),
                    np.deg2rad(90),
                    clockwise=False,
                    line_types=line[lane],
                    width = width,
                    speed_limit = speed_limit,
                ),
            )
            net.add_lane(
                "sw",
                "se",
                    StraightLane(
                    [ 0.9, center[1] + radii[lane] ],
                    [ 1.8, center[1] + radii[lane] ],
                    line_types=line[lane],
                    width = width,
                    speed_limit = speed_limit,
                ),
            )
            net.add_lane(
                "se",
                "ne",
                CircularLane(
                    [ 1.8, 0.9 ],
                    radii[lane],
                    np.deg2rad(90),
                    np.deg2rad(-90),
                    clockwise=False,
                    line_types=line[lane],
                    width = width,
                    speed_limit = speed_limit,
                ),
            )
            net.add_lane(
                "ne",
                "nw",
                    StraightLane(
                    [ 1.8, center[1] - radii[lane] ],
                    [ 0.9, center[1] - radii[lane] ],
                    line_types=line[lane],
                    width = width,
                    speed_limit = speed_limit,
                ),
            )
            
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []

        controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
            self.road, self.road.network.random_lane_index(rng), speed=None, longitudinal=rng.uniform(0, 0.9 )
        )
        #controlled_vehicle.heading += np.pi * rng.integers(0, 2)

        self.controlled_vehicles.append(controlled_vehicle)
        self.road.vehicles.append(controlled_vehicle)

'''
        for i in range(self.config["controlled_vehicles"]):
            lane_index = (
                ("se", "ex", rng.integers(2))
                if i == 0
                else self.road.network.random_lane_index(rng)
            )
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road, lane_index, speed=None, longitudinal=rng.uniform(0, 1)
            )

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        # Front vehicle
        vehicle = IDMVehicle.make_on_lane(
            self.road,
            ("se", "ex", lane_index[-1]),
            longitudinal=rng.uniform(
                low=0, high=self.road.network.get_lane(("se", "ex", 0)).length
            ),
            speed=6 + rng.uniform(high=3),
        )
        self.road.vehicles.append(vehicle)
        
        # Other vehicles
        for i in range(rng.integers(self.config["other_vehicles"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(
                self.road,
                random_lane_index,
                longitudinal=rng.uniform(
                    low=0, high=self.road.network.get_lane(random_lane_index).length
                ),
                speed=6 + rng.uniform(high=3),
            )
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)
'''
