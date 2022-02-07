import Agents
import json
from copy import deepcopy

default_pool_ids = [
    "00001",
    "00002",
    "00003",
    "00004",
    "00005",
    "10001",
    "10002",
    "10003",
    "10004",
    "10005",
]


# A way to interface with our player pool files
class PlayerPool:
    def __init__(self, name, pnr, json_file, pool_ids=None):
        with open(json_file, "r") as f:
            json_vals = json.load(f)

        filtered_vals = {
            key: value
            for key, value in json_vals.items()
            if pool_ids is None or key in pool_ids
        }

        self.player_dict = {
            k: self.from_dict(name, pnr, v) for k, v in filtered_vals.items()
        }
        self.size = len(self.player_dict)

    @property
    def data(self):
        return [self.player_dict[k] for k in sorted(self.player_dict.keys())]

    def get_size(self):
        return self.size

    def get_names(self):
        return [type(a).__name__ for a in self.data]

    def copies(self):
        re = []

        for agent in self.data:
            re.append(deepcopy(agent))

        return re

    def get_agents(self):
        return self.data

    def get_player_dict(self):
        return self.player_dict

    def from_dict(self, name, pnr, json_dict):
        json_dict["name"] = name
        json_dict["pnr"] = pnr
        return getattr(Agents, json_dict["player_class"])(**json_dict)
