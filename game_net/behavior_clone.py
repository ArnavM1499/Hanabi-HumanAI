import numpy as np
import os
import torch
import common_game_functions as cgf
import Agents.common_player_functions as cpf

from lstm_net import LSTMNet, default_config
from Agents.player import Action


MODEL_DIR = "game_net/models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BehaviorCloneBase:
    def __init__(self):
        self.models = {}

    def _load_model(self, agent_id):
        model = LSTMNet(**default_config)
        model_path = os.path.join(MODEL_DIR, "model_lstm_{}.pth".format(agent_id))
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=DEVICE,
            )
        )
        self.models[agent_id] = model

    def predict(self, agent_id, game_state, player_model, return_dict=True):
        return self.sequential_predict(
            agent_id, [game_state], [player_model], return_dict
        )

    def sequential_predict(
        self, agent_id, game_states, player_models, return_dict=True
    ) -> Action:
        if agent_id not in self.models.keys():
            self._load_model(agent_id)

        state_list = []

        for game_state, player_model in zip(game_states, player_models):
            current_player, encoded_state = self._convert_game_state(
                game_state, player_model
            )
            state_list.append(encoded_state)

        game_net_input = (
            torch.tensor(np.array([state_list]), dtype=torch.float32) * 0.333
        )  # with rough normalization
        try:
            pred = self.models[agent_id](
                torch.transpose(game_net_input, 0, 1), [len(state_list)]
            ).data[-1]
        except RuntimeError:
            import pdb

            pdb.set_trace()
        # action = int(tf.math.argmax(pred, axis=1).numpy()[-1])
        ret = dict()
        pred = torch.nn.functional.softmax(pred, dim=0)

        for i, p in enumerate(pred):
            ret[Action.from_encoded(i, pnr=current_player)] = p

        if return_dict:
            return ret
        else:
            return max(ret.keys(), key=lambda x: ret[x])

    def _convert_game_state(
        self, game_state: cgf.GameState, player_model: cgf.BasePlayerModel
    ) -> list:
        current_player = game_state.get_current_player()
        partner_player = 1 - current_player
        try:
            last_action = player_model.get_actions()[partner_player][-1]
        except IndexError:
            last_action = None
        knowledge = game_state.get_all_knowledge()
        board = game_state.get_board()
        trash = game_state.get_trash()
        extra = []
        for k in knowledge[partner_player]:
            extra.append(cpf.slot_playable_pct(k, board))
        while len(extra) < 5:
            extra.append(0)
        for k in knowledge[current_player]:
            extra.append(cpf.slot_playable_pct(k, board))
        while len(extra) < 10:
            extra.append(0)
        for k in knowledge[partner_player]:
            extra.append(cpf.slot_discardable_pct(k, board, trash))
        while len(extra) < 15:
            extra.append(0)
        for k in knowledge[current_player]:
            extra.append(cpf.slot_discardable_pct(k, board, trash))
        while len(extra) < 20:
            extra.append(0)
        encoded = cgf.encode_state(
            game_state.get_hands()[partner_player],
            knowledge[partner_player],
            knowledge[current_player],
            board,
            trash,
            game_state.get_hits(),
            game_state.get_num_hints(),
            last_action,
            Action(cgf.PLAY, cnr=0),
            current_player,
            extra,
        )
        pnr, _, encoded_state = cgf.decode_state(encoded)
        return current_player, encoded_state


BehaviorClone = BehaviorCloneBase()
