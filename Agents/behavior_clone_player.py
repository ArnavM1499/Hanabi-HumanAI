from Agents.player import Player
from game_net.behavior_clone import BehaviorClone


class BehaviorPlayer(Player):
    def __init__(self, name, pnr, **kwargs):
        assert "agent_id" in kwargs.keys()
        self.name = name
        self.pnr = pnr
        self.agent_id = kwargs["agent_id"]
        self.history_states = []
        self.history_models = []

    def get_action(self, game_state, player_model):
        self.history_states.append(game_state)
        self.history_models.append(player_model)
        return BehaviorClone.sequential_predict(
            self.agent_id, self.history_states, self.history_models, return_dict=False
        )
