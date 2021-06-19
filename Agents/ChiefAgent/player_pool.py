import Agents

# A way to interface with our player pool files
class PlayerPool:
    def __init__(self, json_file):
        self.size =
        self.data = 

    def get_size(self):
        return self.size

    def copies(self):
        pass

    @staticmethod
    def from_dict(name, pnr, json_dict):
        json_dict["name"] = name
        json_dict["pnr"] = pnr
        return getattr(Agents, json_dict["player_class"])(**json_dict)