from Agents.common_player_functions import *
from Agents.player import *


class CardCountingLeftRightPlayer(Player):
    def __init__(self, name, pnr, count_cards, play_left):
        self.name = name
        self.pnr = pnr
        self.count_cards = count_cards
        self.play_left = play_left
        self.explanation = []
        self.personal_knowledge = None

    def update_new_card_knowledge(self, other_cards):
        if not self.count_cards:
            return

        for hand in self.personal_knowledge:
            for (col, num) in other_cards:
                hand[col][num - 1] = max(0, hand[col][num - 1])

    def get_action(
        self, game_state, player_model#, nr, hands, knowledge, trash, played, board, valid_actions, hints
    ):
        nr = game_state.get_current_player()
        hands = game_state.get_hands()
        trash = game_state.get_trash()
        played = game_state.get_played()
        board = game_state.get_board()
        valid_actions = game_state.get_valid_actions()
        hints = game_state.get_num_hints()
        knowledge = player_model.get_all_knowledge()

        if self.personal_knowledge == None:
            self.personal_knowledge = knowledge[nr].copy()

        other_cards = []

        for i in hands:
            for h in i:
                other_cards.append(h)

        self.update_new_card_knowledge(trash + played + other_cards)

        handsize = len(knowledge[0])
        possible = []

        for k in self.personal_knowledge:
            possible.append(get_possible(k))

        discards = []
        duplicates = []
        playables = []

        for i, p in enumerate(possible):
            if percent_playable(p, board) >= 0.8:
                playables.append(i)
            if discardable(p, board):
                discards.append(i)

        if playables and self.play_left:
            return Action(PLAY, cnr=playables[0])
        elif playables and not self.play_left:
            return Action(PLAY, cnr=playables[-1])

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

        if prefer and hints > 0:
            return random.choice(prefer)
        return random.choice([Action(DISCARD, cnr=i) for i in range(len(knowledge[0]))])
        return [Action(DISCARD, cnr=i) for i in range(len(knowledge[0]))][0]

    def inform(self, action, player, game, model):
        pass
