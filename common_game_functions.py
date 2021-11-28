from copy import deepcopy
import random

global HINT_COLOR, HINT_NUMBER, PLAY, DISCARD, CANDISCARD, GREEN, YELLOW, WHITE, BLUE, RED, ALL_COLORS, COLORNAMES, COUNTS  # noqa E501

HINT_COLOR = 0
HINT_NUMBER = 1
PLAY = 2
DISCARD = 3
CANDISCARD = 128

GREEN = 0
YELLOW = 1
WHITE = 2
BLUE = 3
RED = 4
ALL_COLORS = [GREEN, YELLOW, WHITE, BLUE, RED]
COLORNAMES = ["green", "yellow", "white", "blue", "red"]

COUNTS = [3, 2, 2, 2, 1]


class StartOfGame(Exception):
    pass


def f(something):
    if type(something) == list:
        return map(f, something)
    elif type(something) == dict:
        return {k: something(v) for (k, v) in something.iteritems()}
    elif type(something) == tuple and len(something) == 2:
        return (COLORNAMES[something[0]], something[1])
    return something


def make_deck():
    deck = []
    for col in ALL_COLORS:
        for num, cnt in enumerate(COUNTS):
            for i in range(cnt):
                deck.append((col, num + 1))
    random.shuffle(deck)
    return deck


def initial_knowledge():
    knowledge = []
    for col in ALL_COLORS:
        knowledge.append(COUNTS[:])
    return knowledge


class BasePlayerModel(object):
    def __init__(self, nr, knowledge, hints, actions):
        self.nr = nr
        self.knowledge = knowledge  # This is the knowledge matrix based only on updates in game engine
        self.hints = hints  # These are the hints that this player has received (Format: List of (P,Hint) if recieved from player P)
        self.actions = actions  # These are the actions taken by all players in the past (Format: Dictionary with player as keys and actions as values)

    def get_hints(self):
        return self.hints

    def get_knowledge(self):
        return self.knowledge

    def get_actions(self):
        return self.actions

    def get_hints_from_player(self, p):
        filtered_hints = []

        for player, hint in self.hints:
            if p == player:
                filtered_hints.append(hint)

        return filtered_hints


class GameState(object):
    def __init__(
        self,
        current_player,
        hands,
        trash,
        played,
        board,
        hits,
        valid_actions,
        num_hints,
        all_knowledge,
        hinted_indices=[],
        card_changed=None,
    ):
        self.current_player = current_player
        self.hands = hands
        self.trash = trash
        self.played = played
        self.board = board
        self.hits = hits
        self.valid_actions = valid_actions
        self.num_hints = num_hints
        self.all_knowledge = all_knowledge
        self.hinted_indices = hinted_indices
        self.card_changed = card_changed

    def get_current_player(self):
        return self.current_player

    def get_hands(self):
        return self.hands

    def get_trash(self):
        return self.trash

    def get_played(self):
        return self.played

    def get_board(self):
        return self.board

    def get_hits(self):
        return self.hits

    def get_valid_actions(self):
        return self.valid_actions

    def get_num_hints(self):
        return self.num_hints

    def get_hinted_indices(self):
        return self.hinted_indices

    def get_card_changed(self):
        return self.card_changed

    def get_all_knowledge(self):
        return self.all_knowledge

    def get_common_visible_cards(self):
        cards = deepcopy(self.trash)
        for col, num in self.board:
            cards.extend([(col, i + 1) for i in range(num)])
        return sorted(cards)


def checkpoint(passed):
    assert passed
    if not passed:
        import pdb

        pdb.set_trace()


def apply_hint_to_knowledge(action, hands, knowledges):
    return_knowledge = deepcopy(knowledges)
    if action.type == HINT_COLOR:
        for (col, num), knowledge in zip(
                hands[action.pnr], return_knowledge[action.pnr]
        ):
            if col == action.col:
                for i, k in enumerate(knowledge):
                    if i != col:
                        for i in range(len(k)):
                            k[i] = 0
            else:
                for i in range(len(knowledge[action.col])):
                    knowledge[action.col][i] = 0
    else:
        assert action.type == HINT_NUMBER
        for (col, num), knowledge in zip(
                hands[action.pnr], return_knowledge[action.pnr]
        ):
            if num == action.num:
                for k in knowledge:
                    for i in range(len(COUNTS)):
                        if i + 1 != num:
                            k[i] = 0
            else:
                for k in knowledge:
                    k[action.num - 1] = 0
    return return_knowledge[action.pnr]


def encode_action_values(value_dict):
    values = [0] * 20
    for action in value_dict.keys():
        values[action.encode()] = value_dict[action]
    return values


def encode_new_knowledge_models(knowledge_models):
    values = [[] for i in range(10)]
    for action in knowledge_models.keys():
        encoding = action.encode()
        value = knowledge_models[action]
        values[encoding].extend(value)
        if len(value) < 5:
            values[encoding].append([[0, 0, 0, 0, 0] for _ in range(5)])
    for i in range(10):
        if not values[i]:
            values[i] = [[[0, 0, 0, 0, 0] for _ in range(5)] for _ in range(5)]
    ans = []
    for knowledge in values:
        for single in knowledge:
            ans.extend(sum(single, []))
    assert(len(ans) == 1250)
    return ans


def encode_state(
    partner_hand,
    partner_knowledge,
    self_knowledge,
    board,
    trash,
    hits,
    hints,
    last_action,
    action,
    partner_knowledge_model,
    pnr,
    extras=[],
):
    state = []
    state.extend([(col * 6 + num) for col, num in sorted(board)])
    for (col, num) in partner_hand:
        state.append(col * 5 + num - 1)
    state.extend([25] * (5 - len(partner_hand)))
    checkpoint(len(state) == 10)
    knowledges = []
    knowledges.extend(partner_knowledge)
    if len(partner_knowledge) < 5:
        knowledges.append([[0, 0, 0, 0, 0] for _ in range(5)])
    knowledges.extend(self_knowledge)
    if len(self_knowledge) < 5:
        knowledges.append([[0, 0, 0, 0, 0] for _ in range(5)])
    for knowledge in knowledges:
        state.extend(sum(knowledge, []))
    trash_reformat = [[0] * 5 for _ in range(5)]
    for (col, num) in trash:
        trash_reformat[col][num - 1] += 1
    state.extend(sum(trash_reformat, []))
    checkpoint(len(state) == 285)
    state.extend(encode_new_knowledge_models(partner_knowledge_model))
    state.extend(
        [3 * x for x in extras]
    )  # 3 is a magic number, extras should be normalized to 0-1
    state.append(3 - hits)
    state.append(hints)
    if last_action:
        state.append(last_action.encode())
    else:
        state.append(20)
    state.append(action.encode())
    state.append(pnr)
    return state


def decode_state(state):
    # convert hand, hint, hit, last action to one hot
    # player {0, 1}, action [0-19], game state
    if state == []:
        raise StartOfGame
    expanded = []
    hand = [0] * 30
    for i in range(5):  # board (5)
        h = hand.copy()
        if state[i] < 25:
            h[state[i]] = 1
        expanded.extend(h)
    hand = [0] * 25
    for i in range(5, 9):  # first four cards
        h = hand.copy()
        if state[i] < 25:
            h[state[i]] = 1
        expanded.extend(h)
    hand.append(0)  # include empty card
    hand[state[10]] = 1
    expanded.extend(hand)
    expanded.extend(state[10:-5])
    # action_values = state[-25:-5]
    hits, hints, last_action, action, pnr = state[-5:]
    expanded.append(hits % 2)
    expanded.append(hits // 2)
    hints_one_hot = [0] * 9
    hints_one_hot[hints] = 1
    expanded.extend(hints_one_hot)
    action_one_hot = [0] * 21  # include empty last_action
    action_one_hot[last_action] = 1
    expanded.extend(action_one_hot)
    return pnr, action, expanded
