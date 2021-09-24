import numpy as np
import os
import torch
import common_game_functions as cgf

from lstm_net import LSTMNet, default_config
from Agents.player import Action

MODEL_DIR = "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BehaviorCloneBase:
    def __init__(self):
        self.models = {}

    def _load_model(self, agent_id):
        model = LSTMNet(**default_config)
        model.load_state_dict(
            torch.load(
                os.path.join(MODEL_DIR, "model_lstm_{}.pth".format(agent_id)),
                map_location=DEVICE,
            )
        )
        self.models[agent_id] = model

    def sequential_predict(
        self,
        agent_id,
        game_states,
        player_models,
    ) -> Action:
        if agent_id not in self.models.keys():
            try:
                self._load_model(agent_id)
            except:  # noqa E722
                raise FileNotFoundError

        state_list = []

        for game_state, player_model in zip(game_states, player_models):
            current_player, encoded_state = self._convert_game_state(
                game_state, player_model
            )
            state_list.append(encoded_state)

        game_net_input = (
            torch.tensor(np.array(state_list), dtype=torch.float32) * 0.333
        )  # with rough normalization
        pred = self.models[agent_id](game_net_input, training=False)[-1]
        # action = int(tf.math.argmax(pred, axis=1).numpy()[-1])
        ret = dict()

        for i, p in enumerate(pred):
            ret[Action.from_encoded(i, pnr=current_player)] = p

        return ret

    def _convert_game_state(
        self, game_state: cgf.GameState, player_model: cgf.BasePlayerModel
    ) -> list:
        current_player = game_state.get_current_player()
        partner_player = 1 - current_player
        try:
            last_action = player_model.get_actions()[partner_player][-1]
        except IndexError:
            last_action = None
        encoded = cgf.encode_state(
            game_state.get_hands()[partner_player],
            game_state.get_all_knowledge()[partner_player],
            game_state.get_all_knowledge()[current_player],
            game_state.get_board(),
            game_state.get_trash(),
            game_state.get_hits(),
            game_state.get_num_hints(),
            last_action,
            Action(cgf.PLAY, cnr=0),
            current_player,
        )
        pnr, _, encoded_state = cgf.decode_state(encoded)
        return current_player, encoded_state


BehaviorClone = BehaviorCloneBase()
