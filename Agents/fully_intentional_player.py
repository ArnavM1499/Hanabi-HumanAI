from Agents.common_player_functions import *
from Agents.player import *


class FullyIntentionalPlayer(Player):
    def __init__(self, name, pnr):
        self.name = name
        self.hints = {}
        self.pnr = pnr
        self.gothint = None
        self.last_knowledge = []
        self.last_played = []
        self.last_board = []

    def get_action(
        self, game_state, player_model#nr, hands, knowledge, trash, played, board, valid_actions, hints
    ):
        nr = game_state.get_current_player()
        hands = game_state.get_hands()
        trash = game_state.get_trash()
        played = game_state.get_played()
        board = game_state.get_board()
        valid_actions = game_state.get_valid_actions()
        hints = game_state.get_num_hints()
        knowledge = player_model.get_all_knowledge()

        handsize = len(knowledge[0])
        possible = []

        self.gothint = None
        for k in knowledge[nr]:
            possible.append(get_possible(k))

        discards = []
        plays = []
        duplicates = []
        for i, p in enumerate(possible):
            if playable(p, board):
                plays.append(i)
            if discardable(p, board):
                discards.append(i)

        playables = []
        useless = []
        discardables = []
        othercards = trash + board
        intentions = [None for i in range(handsize)]
        for i, h in enumerate(hands):
            if i != nr:
                for j, (col, n) in enumerate(h):
                    if board[col][1] + 1 == n:
                        playables.append((i, j))
                        intentions[j] = PLAY
                    if board[col][1] <= n:
                        useless.append((i, j))
                        if not intentions[j]:
                            intentions[j] = DISCARD
                    if n < 5 and (col, n) not in othercards:
                        discardables.append((i, j))
                        if not intentions[j]:
                            intentions[j] = CANDISCARD

        if hints > 0:
            valid = []
            for c in ALL_COLORS:
                action = (HINT_COLOR, c)
                # print "HINT", COLORNAMES[c],
                (isvalid, score, _) = pretend(
                    action, knowledge[1 - nr], intentions, hands[1 - nr], board
                )
                # print isvalid, score
                if isvalid:
                    valid.append((action, score))

            for r in range(5):
                r += 1
                action = (HINT_NUMBER, r)
                # print "HINT", r,
                (isvalid, score, _) = pretend(
                    action, knowledge[1 - nr], intentions, hands[1 - nr], board
                )
                # print isvalid, score
                if isvalid:
                    valid.append((action, score))
            if valid:
                valid.sort(key=lambda a_s: -a_s[1])
                # print valid
                (a, s) = valid[0]
                if a[0] == HINT_COLOR:
                    return Action(HINT_COLOR, pnr=1 - nr, col=a[1])
                else:
                    return Action(HINT_NUMBER, pnr=1 - nr, num=a[1])

        for i, k in enumerate(knowledge):
            if i == nr or True:
                continue
            cards = list(range(len(k)))
            random.shuffle(cards)
            c = cards[0]
            (col, num) = hands[i][c]
            hinttype = [HINT_COLOR, HINT_NUMBER]
            if (c, i) not in self.hints:
                self.hints[(c, i)] = []
            for h in self.hints[(c, i)]:
                hinttype.remove(h)
            if hinttype and hints > 0:
                if random.choice(hinttype) == HINT_COLOR:
                    self.hints[(c, i)].append(HINT_COLOR)
                    return Action(HINT_COLOR, pnr=i, col=col)
                else:
                    self.hints[(c, i)].append(HINT_NUMBER)
                    return Action(HINT_NUMBER, pnr=i, num=num)

        return random.choice([Action(DISCARD, cnr=i) for i in range(handsize)])

    def inform(self, action, player, game, x):
        if action.type in [PLAY, DISCARD]:
            x = str(action)
            if (action.cnr, player) in self.hints:
                self.hints[(action.cnr, player)] = []
            for i in range(10):
                if (action.cnr + i + 1, player) in self.hints:
                    self.hints[(action.cnr + i, player)] = self.hints[
                        (action.cnr + i + 1, player)
                    ]
                    self.hints[(action.cnr + i + 1, player)] = []
        elif action.pnr == self.pnr:
            self.gothint = (action, player)
            self.last_knowledge = x.knowledge[:]
            self.last_board = game.board[:]
            self.last_trash = game.trash[:]
            self.played = game.played[:]
