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
        return self.knowledge[self.nr]

    def get_actions(self):
        return self.actions

    def get_hints_from_player(self, p):
        filtered_hints = []

        for player, hint in self.hints:
            if p == player:
                filtered_hints.append(hint)

        return filtered_hints

    def get_all_knowledge(self):
        return self.knowledge


class GameState(object):
    def __init__(
        self,
        current_player,
        hands,
        trash,
        played,
        board,
        valid_actions,
        num_hints,
        hinted_indices=[],
        card_changed=None,
    ):
        self.current_player = current_player
        self.hands = hands
        self.trash = trash
        self.played = played
        self.board = board
        self.valid_actions = valid_actions
        self.num_hints = num_hints
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

    def get_valid_actions(self):
        return self.valid_actions

    def get_num_hints(self):
        return self.num_hints

    def get_hinted_indices(self):
        return self.hinted_indices

    def get_card_changed(self):
        return self.card_changed