from Agents.common_player_functions import *
from Agents.player import *


class InnerStatePlayer(Player):
    def __init__(self, name, pnr):
        self.name = name
        self.explanation = []

    def get_action(
        self, nr, hands, knowledge, trash, played, board, valid_actions, hints
    ):
        handsize = len(knowledge[0])
        possible = []
        for k in knowledge[nr]:
            possible.append(get_possible(k))

        discards = []
        duplicates = []
        for i, p in enumerate(possible):
            if playable(p, board):
                return Action(PLAY, cnr=i)
            if discardable(p, board):
                discards.append(i)

        if discards:
            return Action(DISCARD, cnr=random.choice(discards))

        playables = []
        for i, h in enumerate(hands):
            if i != nr:
                for j, (col, n) in enumerate(h):
                    if board[col][1] + 1 == n:
                        playables.append((i, j))

        if playables and hints > 0:
            i, j = playables[0]
            if random.random() < 0.5:
                return Action(HINT_COLOR, pnr=i, col=hands[i][j][0])
            return Action(HINT_NUMBER, pnr=i, num=hands[i][j][1])

        for i, k in enumerate(knowledge):
            if i == nr:
                continue
            cards = list(range(len(k)))
            random.shuffle(cards)
            c = cards[0]
            (col, num) = hands[i][c]
            hinttype = [HINT_COLOR, HINT_NUMBER]
            if hinttype and hints > 0:
                if random.choice(hinttype) == HINT_COLOR:
                    return Action(HINT_COLOR, pnr=i, col=col)
                else:
                    return Action(HINT_NUMBER, pnr=i, num=num)

        prefer = []
        for v in valid_actions:
            if v.type in [HINT_COLOR, HINT_NUMBER]:
                prefer.append(v)
        prefer = []
        if prefer and hints > 0:
            return random.choice(prefer)
        return random.choice([Action(DISCARD, cnr=i) for i in range(len(knowledge[0]))])

    def inform(self, action, player, game):
        pass
