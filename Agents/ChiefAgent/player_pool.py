import Agents
import json
from copy import deepcopy


# A way to interface with our player pool files
class PlayerPool:
    def __init__(self, name, pnr, json_file):
        with open(json_file, "r") as f:
            json_vals = json.load(f)
        self.player_dict = {
            k: self.from_dict(name, pnr, v) for k, v in json_vals.items()
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
