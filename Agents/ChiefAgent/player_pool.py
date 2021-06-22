import Agents
import json
from copy import deepcopy

# A way to interface with our player pool files
class PlayerPool:
    def __init__(self, name, pnr, json_file):
        with open(json_file, 'r') as f:
            json_vals = json.load(f)

        self.data = []

        for val in json_vals:
            self.data.append(self.from_dict(name, pnr, json_vals[val]))

        self.size = len(self.data)

    def get_size(self):
        return self.size

    def copies(self):
        re = []

        for agent in self.data:
            re.append(deepcopy(agent))

        return re

    def get_agents(self):
        return self.data

    def from_dict(self, name, pnr, json_dict):
        json_dict["name"] = name
        json_dict["pnr"] = pnr
        return getattr(Agents, json_dict["player_class"])(**json_dict)