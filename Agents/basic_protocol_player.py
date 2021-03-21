from Agents.common_player_functions import *
from Agents.player import *


class BasicProtocolPlayer(Player):
    # strategies is a dictionary with all strategies being used; can be modified as we use more
    def __init__(self, name, pnr, strategies):
        super().__init__(name, pnr)
        self.strategies = strategies
        self.explanation = []
        self.personal_knowledge = None

    # not currently used
    def update_new_card_knowledge(self, other_cards):
        if self.strategies["count_cards"]:
            for hand in self.personal_knowledge:
                for (col, num) in other_cards:
                    hand[col][num - 1] = max(0, hand[col][num - 1])

    def get_action(self, game_state, base_player_model):
        # this should only happen on the 1st turn
        if self.personal_knowledge is None:
            self.personal_knowledge = base_player_model.get_knowledge().copy()

        # temporary until we implement knowledge
        self.personal_knowledge = base_player_model.get_knowledge().copy()

        # high confidence play: 95%+ chance of success
        max_play_success_rate = 0.0
        max_play_index = -1
        # note that this implementation means that we will play the leftmost if probabilities equal
        for slot_index in range(len(self.personal_knowledge)):
            card_slot = self.personal_knowledge[slot_index]
            success_rate = slot_playable_pct(card_slot, game_state.get_board())
            if success_rate > max_play_success_rate:
                max_play_index = slot_index
                max_play_success_rate = success_rate
        #print(max_play_success_rate)
        if max_play_success_rate >= 0.95:
            return Action(PLAY, cnr=max_play_index)

        # high confidence discard: 95%+ chance of success
        max_discard_success_rate = 0.0
        max_discard_index = -1
        # note that this implementation means that we will play the leftmost if probabilities equal
        for slot_index in range(len(self.personal_knowledge)):
            card_slot = self.personal_knowledge[slot_index]
            success_rate = slot_discardable_pct(card_slot, game_state.get_board())
            if success_rate > max_discard_success_rate:
                max_discard_index = slot_index
                max_discard_success_rate = success_rate
        #print(max_discard_success_rate)
        if max_discard_success_rate >= 0.95:
            return Action(DISCARD, cnr=max_discard_index)

        return random.choice(game_state.get_valid_actions())

    def inform(self, action, player, game):
        # we already know our own move
        if player == self.pnr:
            pass

        pass
