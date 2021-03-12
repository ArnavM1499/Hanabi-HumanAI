from Agents.common_player_functions import *
from Agents.player import *

class IntentionalPlayer(Player):
    def __init__(self, name, pnr):
        self.name = name
        self.hints = {}
        self.pnr = pnr
        self.gothint = None
        self.last_knowledge = []
        self.last_played = []
        self.last_board = []
        self.explanation = []
    def get_action(self, nr, hands, knowledge, trash, played, board, valid_actions, hints):
        handsize = len(knowledge[0])
        possible = []
        result = None
        self.explanation = []
        self.explanation.append(["Your Hand:"] + list(map(f, hands[1-nr])))

        self.gothint = None
        for k in knowledge[nr]:
            possible.append(get_possible(k))

        discards = []
        duplicates = []
        for i,p in enumerate(possible):
            if playable(p,board):
                result = Action(PLAY, cnr=i)
            if discardable(p,board):
                discards.append(i)

        if discards and hints < 8 and not result:
            result =  Action(DISCARD, cnr=random.choice(discards))

        playables = []
        useless = []
        discardables = []
        othercards = trash + board
        intentions = [None for i in range(handsize)]
        for i,h in enumerate(hands):
            if i != nr:
                for j,(col,n) in enumerate(h):
                    if board[col][1] + 1 == n:
                        playables.append((i,j))
                        intentions[j] = PLAY
                    if board[col][1] >= n:
                        useless.append((i,j))
                        if not intentions[j]:
                            intentions[j] = DISCARD
                    if n < 5 and (col,n) not in othercards:
                        discardables.append((i,j))
                        if not intentions[j]:
                            intentions[j] = CANDISCARD

        self.explanation.append(["Intentions"] + list(map(format_intention, intentions)))



        if hints > 0:
            valid = []
            for c in ALL_COLORS:
                action = (HINT_COLOR, c)
                #print "HINT", COLORNAMES[c],
                (isvalid,score,expl) = pretend(action, knowledge[1-nr], intentions, hands[1-nr], board)
                self.explanation.append(["Prediction for: Hint Color " + COLORNAMES[c]] + list(map(format_intention, expl)))
                #print isvalid, score
                if isvalid:
                    valid.append((action,score))

            for r in range(5):
                r += 1
                action = (HINT_NUMBER, r)
                #print "HINT", r,

                (isvalid,score, expl) = pretend(action, knowledge[1-nr], intentions, hands[1-nr], board)
                self.explanation.append(["Prediction for: Hint Rank " + str(r)] + list(map(format_intention, expl)))
                #print isvalid, score
                if isvalid:
                    valid.append((action,score))

            if valid and not result:
                valid.sort(key=lambda a_s: -a_s[1])
                #print valid
                (a,s) = valid[0]
                if a[0] == HINT_COLOR:
                    result = Action(HINT_COLOR, pnr=1-nr, col=a[1])
                else:
                    result = Action(HINT_NUMBER, pnr=1-nr, num=a[1])

        self.explanation.append(["My Knowledge"] + list(map(format_knowledge, knowledge[nr])))
        possible = [ Action(DISCARD, cnr=i) for i in range(handsize) ]

        scores = [pretend_discard(p, knowledge[nr], board, trash) for p in possible]
        def format_term(term):
            (col,rank,n,prob,val) = term
            return COLORNAMES[col] + " " + str(rank) + " (%.2f%%): %.2f"%(prob*100, val)

        self.explanation.append(["Discard Scores"] + ["\n".join(map(format_term, a_s_t[2])) + "\n%.2f"%(a_s_t[1]) for a_s_t in scores])
        scores.sort(key=lambda a_s_t1: -a_s_t1[1])
        if result:
            return result
        return scores[0][0]

        return random.choice([Action(DISCARD, cnr=i) for i in range(handsize)])
    def inform(self, action, player, game):
        if action.type in [PLAY, DISCARD]:
            x = str(action)
            if (action.cnr,player) in self.hints:
                self.hints[(action.cnr,player)] = []
            for i in range(10):
                if (action.cnr+i+1,player) in self.hints:
                    self.hints[(action.cnr+i,player)] = self.hints[(action.cnr+i+1,player)]
                    self.hints[(action.cnr+i+1,player)] = []
        elif action.pnr == self.pnr:
            self.gothint = (action,player)
            self.last_knowledge = game.knowledge[:]
            self.last_board = game.board[:]
            self.last_trash = game.trash[:]
            self.played = game.played[:]
