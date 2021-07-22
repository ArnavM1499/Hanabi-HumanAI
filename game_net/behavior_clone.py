from copy import deepcopy
import os
import tensorflow as tf

import common_game_functions as cgf
from Agents.player import Action
from game_net.settings import model, classification_head

MODEL_DIR = "models"


class BehaviorCloneBase:
    def __init__(self):
        self.model = model
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.model.load_weights(os.path.join(self.current_dir, MODEL_DIR, "model"))
        self.heads = {}

    def predict(
        self,
        agent_id: str,
        game_state: cgf.GameState,
        player_model: cgf.BasePlayerModel,
    ) -> Action:
        if agent_id not in self.heads.keys():
            try:
                self.heads[agent_id] = deepcopy(classification_head)
                self.heads[agent_id].load_weights(
                    os.path.join(self.current_dir, MODEL_DIR, "model_head_" + agent_id)
                )
            except:  # noqa E722
                raise FileNotFoundError
        current_player, encoded_state = self._convert_game_state(
            game_state, player_model
        )
        game_net_input = (
            tf.constant([encoded_state], dtype=tf.float32) * 0.333
        )  # with rough normalization
        features = self.model(game_net_input, training=False)
        pred = self.heads[agent_id](features, training=False)
        action = int(tf.math.argmax(pred, axis=1).numpy())
        return Action.from_encoded(action, pnr=current_player)

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

    def create_new_head(self, new_id: str, copy_from_id=""):
        if copy_from_id in self.heads.keys():
            self.heads[new_id] = deepcopy(self.heads[copy_from_id])
        else:
            self.heads[new_id] = deepcopy(classification_head)


BehaviorClone = BehaviorCloneBase()
