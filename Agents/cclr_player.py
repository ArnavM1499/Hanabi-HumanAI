from Agents.common_player_functions import *
from Agents.player import *


class CardCountingLeftRightPlayer(Player):
    def __init__(self, name, pnr, count_cards, play_left):
        self.name = name
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

    def get_action(self, game_state, player_model):
        # game_state contains current_player, hands, trash, played, board, valid_actions, num_hints, and all_knowledge
        # player_model contains knowledge, hints, and actions

        if self.personal_knowledge == None:
            self.personal_knowledge = player_model.knowledge.copy()

        other_cards = []

        for i in game_state.hands:
            for h in i:
                other_cards.append(h)

        self.update_new_card_knowledge(game_state.trash + game_state.played + other_cards)

        handsize = len(self.personal_knowledge)
        possible = []

        for k in self.personal_knowledge:
            possible.append(get_possible(k))

        discards = []
        duplicates = []
        playables = []

        for i, p in enumerate(possible):
            if percent_playable(p, game_state.board) >= 0.8:
                playables.append(i)
            if discardable(p, game_state.board):
                discards.append(i)

        if playables and self.play_left:
            return Action(PLAY, cnr=playables[0])
        elif playables and not self.play_left:
            return Action(PLAY, cnr=playables[-1])

        if discards:
            return Action(DISCARD, cnr=random.choice(discards))

        playables = []
        for i, h in enumerate(game_state.hands):
            if i != game_state.current_player:
                for j, (col, n) in enumerate(h):
                    if game_state.board[col][1] + 1 == n:
                        playables.append((i, j))

        if playables and game_state.num_hints > 0:
            i, j = playables[0]
            if random.random() < 0.5:
                return Action(HINT_COLOR, pnr=i, col=game_state.hands[i][j][0])
            return Action(HINT_NUMBER, pnr=i, num=game_state.hands[i][j][1])

        for i, k in enumerate(game_state.all_knowledge):
            if i == game_state.current_player:
                continue
            cards = list(range(len(k)))
            random.shuffle(cards)
            c = cards[0]
            (col, num) = game_state.hands[i][c]
            hinttype = [HINT_COLOR, HINT_NUMBER]
            if hinttype and game_state.num_hints > 0:
                if random.choice(hinttype) == HINT_COLOR:
                    return Action(HINT_COLOR, pnr=i, col=col)
                else:
                    return Action(HINT_NUMBER, pnr=i, num=num)

        prefer = []
        for v in game_state.valid_actions:
            if v.type in [HINT_COLOR, HINT_NUMBER]:
                prefer.append(v)

        if prefer and game_state.num_hints > 0:
            return random.choice(prefer)
        return random.choice([Action(DISCARD, cnr=i) for i in range(len(self.personal_knowledge))])

    def inform(self, action, player, game):
        pass
