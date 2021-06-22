from copy import deepcopy
import random

global HINT_COLOR, HINT_NUMBER, PLAY, DISCARD, CANDISCARD, GREEN, YELLOW, WHITE, BLUE, RED, ALL_COLORS, COLORNAMES, COUNTS

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


ENCODING_MAX = (
    [25, 25, 25, 25, 26]  # partner hand, 26 includes empty card
    + [4, 3, 3, 3, 2] * 50  # both knowledges
    + [6] * 5  # board
    + [4, 3, 3, 3, 2] * 5  # trash
    + [3]  # hits
    + [9]  # hints
    + [4, 5]  # action
    + [2]  # pnr
)


def encode_state(
    partner_hand,
    partner_knowledge,
    self_knowledge,
    board,
    trash,
    hits,
    hints,
    action,
    pnr,
):
    """compress the game state in favor of saving storage space"""

    state = []
    for (col, num) in partner_hand:
        state.append(col * 5 + num - 1)
    if len(state) < 5:
        state.append(25)
    knowledges = []
    knowledges.extend(partner_knowledge)
    if len(partner_knowledge) < 5:
        knowledges.append([[0, 0, 0, 0, 0] for _ in range(5)])
    knowledges.extend(self_knowledge)
    if len(self_knowledge) < 5:
        knowledges.append([[0, 0, 0, 0, 0] for _ in range(5)])
    for knowledge in knowledges:
        for row in knowledge:
            state.extend(row)
    state.extend([num for col, num in sorted(board)])
    trash_reformat = [[0] * 5 for _ in range(5)]
    for (col, num) in trash:
        trash_reformat[col][num - 1] += 1
    for row in trash_reformat:
        state.extend(row)
    state.append(3 - hits)
    state.append(hints)
    state.extend(action.encode())
    state.append(pnr)
    encoded = 0
    for i, (v, m) in enumerate(zip(state, ENCODING_MAX)):
        encoded += v
        encoded *= m
    encoded = encoded // ENCODING_MAX[-1]
    return hex(encoded)[2:] + "\n"


def decode_state(code):
    if isinstance(code, str):
        code = int(code, 16)
    res = []
    for m in ENCODING_MAX[::-1]:
        res.append(code % m)
        code = code // m
    # player {0, 1}, action [0-19], game state
    return res[0], res[1] * res[2], res[:2:-1]
