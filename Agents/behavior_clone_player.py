from Agents.player import Player
from game_net.behavior_clone import BehaviorClone
from copy import deepcopy


class BehaviorPlayer(Player):
    def __init__(self, name, pnr, **kwargs):
        assert "agent_id" in kwargs.keys()
        self.name = name
        self.pnr = pnr
        self.agent_id = kwargs["agent_id"]
        self.history_states = []
        self.history_models = []
        self.history_partner_knowledge_model = []
        self.is_behavior_clone = True

    def get_action(self, game_state, player_model, partner_knowledge_model):
        self.history_states.append(deepcopy(game_state))
        self.history_models.append(deepcopy(player_model))
        self.history_partner_knowledge_model.append(deepcopy(partner_knowledge_model))

        return BehaviorClone.sequential_predict(
            self.agent_id,
            self.history_states,
            self.history_models,
            self.history_partner_knowledge_model,
            return_dict=False,
        )
