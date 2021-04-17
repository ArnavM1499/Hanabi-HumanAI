import Agents


# A way to interface with our player pool files
class PlayerPool:
    @staticmethod
    def from_dict(name, pnr, json_dict):
        json_dict["name"] = name
        json_dict["pnr"] = pnr
        return getattr(Agents, json_dict["player_class"])(**json_dict)
