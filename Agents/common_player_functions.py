import copy
from Agents.player import Action
from common_game_functions import *


def hint_color(knowledge, color, truth):
    result = []
    for col in ALL_COLORS:
        if truth == (col == color):
            result.append(knowledge[col][:])
        else:
            result.append([0 for i in knowledge[col]])
    return result


def hint_rank(knowledge, rank, truth):
    result = []
    for col in ALL_COLORS:
        colknow = []
        for i, k in enumerate(knowledge[col]):
            if truth == (i + 1 == rank):
                colknow.append(k)
            else:
                colknow.append(0)
        result.append(colknow)
    return result


def iscard(colnum):
    (c, n) = colnum
    knowledge = []
    for col in ALL_COLORS:
        knowledge.append(COUNTS[:])
        for i in range(len(knowledge[-1])):
            if col != c or i + 1 != n:
                knowledge[-1][i] = 0
            else:
                knowledge[-1][i] = 1

    return knowledge


def get_possible(knowledge):
    result = []
    for col in ALL_COLORS:
        for i, cnt in enumerate(knowledge[col]):
            if cnt > 0:
                result.append((col, i + 1))
    return result


# card = (col, nr)
def card_playable(card, board):
    return board[card[0]][1] + 1 == card[1]


def card_discardable(card, board, trash=None):
    col, nr = card
    if board[col][1] >= card[1]:
        return True
    if trash:
        for i in range(1, nr):
            if trash.count((col, i)) == COUNTS[i]:
                return True
    return False


# slot is one of the entries in the knowledge list: a 2D list w/ #'s of possible cards
# ie slot[1][3] = 2 means the card in the slot could be color 1 and number 3
# and there are 2 (1, 3) cards unseen so far
def slot_playable_pct(slot, board):
    total_combos = 0.0
    playable_combos = 0.0
    for col in range(len(slot)):
        # there are 5 possible numbers
        for num in range(5):
            total_combos += slot[col][num]
            if card_playable((col, num + 1), board):
                playable_combos += slot[col][num]

    if total_combos < 1:
        total_combos = 1

    return playable_combos / total_combos


def slot_discardable_pct(slot, board, trash=None):
    total_combos = 0
    discardable_combos = 0
    for col in range(len(slot)):
        # there are 5 possible numbers
        for num in range(5):
            total_combos += slot[col][num]
            if card_discardable((col, num + 1), board, trash):
                discardable_combos += slot[col][num]
    if total_combos < 1:
        total_combos = 1
    return discardable_combos / total_combos


def target_possible(hint, target, knowledge, board):
    slot = copy.deepcopy(knowledge[target])
    # print(slot)
    if hint.type == HINT_COLOR:
        for i in range(len(slot)):
            if i != hint.col:
                slot[i] = [0, 0, 0, 0, 0]
    elif hint.type == HINT_NUMBER:
        # print(hint.num)
        for col in slot:
            for i in range(5):
                if i != hint.num - 1:
                    col[i] = 0
    # print(slot)
    if slot_playable_pct(slot, board) > 0.001:
        return True
    return False


def get_target(hint, hand, exl=None):
    if exl is None:
        exl = []
    target = -1
    for i in range(len(hand)):
        if hint.type == HINT_COLOR and hand[i][0] == hint.col and i not in exl:
            target = i
        elif hint.type == HINT_NUMBER and hand[i][1] == hint.num and i not in exl:
            target = i
    return target


def get_multi_target(hint, hand, knowledge, board, play_threshold, disc_threshold):
    exl = []
    for i in range(len(hand)):
        if slot_playable_pct(knowledge[i], board) >= play_threshold:
            exl.append(i)
    return get_target(hint, hand, exl)


def hint_ambiguous(hint, hand, knowledge, board):
    target = get_target(hint, hand)
    if target == -1:
        return False
    return target_possible(hint, target, knowledge, board)


# returns the # of combos of cards removed from a hint
# if a hint
def targeted_info_gain(hint, hand, target, knowledge, board):
    combos_removed = 0
    if hint.type == HINT_COLOR:
        for slot in knowledge:
            for nr in slot[hint.col]:
                combos_removed += nr
        for i in range(target + 1, len(hand)):
            if hint.col == hand[i][0]:
                # and hint_ambiguous(hint, i, knowledge, board):
                return -1
    elif hint.type == HINT_NUMBER:
        for slot in knowledge:
            for col in slot:
                combos_removed += col[hint.num - 1]
        for i in range(target + 1, len(hand)):
            if hint.num == hand[i][1]:
                # and hint_ambiguous(hint, i, knowledge, board):
                return -1

    return combos_removed


def best_hint_type(hand, target, knowledge, board):
    card = hand[target]
    color_info_gain = targeted_info_gain(
        Action(HINT_COLOR, 0, col=card[0]), hand, target, knowledge, board
    )
    num_info_gain = targeted_info_gain(
        Action(HINT_NUMBER, 0, num=card[1]), hand, target, knowledge, board
    )
    if color_info_gain <= 0 and num_info_gain <= 0:
        return None
    elif color_info_gain > num_info_gain:
        return HINT_COLOR
    else:
        return HINT_NUMBER


def best_discard_hint_type(hand, target, knowledge, board):
    # print(knowledge)
    # print(target)
    # print(hand)
    # print(board)
    card = hand[target]
    color_info_gain = targeted_info_gain(
        Action(HINT_COLOR, 0, col=card[0]), hand, target, knowledge, board
    )
    num_info_gain = targeted_info_gain(
        Action(HINT_NUMBER, 0, num=card[1]), hand, target, knowledge, board
    )
    color_ambiguous = target_possible(
        Action(HINT_COLOR, 0, col=card[0]), target, knowledge, board
    )
    num_ambiguous = target_possible(
        Action(HINT_NUMBER, 0, num=card[1]), target, knowledge, board
    )
    if color_ambiguous:
        color_info_gain -= 100
    if num_ambiguous:
        num_info_gain -= 100
    if color_info_gain <= 0 and num_info_gain <= 0:
        return None
    elif color_info_gain > num_info_gain:
        return HINT_COLOR
    else:
        return HINT_NUMBER


def playable(possible, board):
    for (col, nr) in possible:
        if board[col][1] + 1 != nr:
            return False
    return True


def percent_playable(possible, board):
    num = 0

    for (col, nr) in possible:
        if board[col][1] + 1 == nr:
            num += 1

    return num / len(possible)


def potentially_playable(possible, board):
    for (col, nr) in possible:
        if board[col][1] + 1 == nr:
            return True
    return False


def discardable(possible, board):
    for (col, nr) in possible:
        if board[col][1] < nr:
            return False
    return True


def potentially_discardable(possible, board):
    for (col, nr) in possible:
        if board[col][1] >= nr:
            return True
    return False


def update_knowledge(knowledge, used):
    result = copy.deepcopy(knowledge)
    for r in result:
        for (c, nr) in used:
            r[c][nr - 1] = max(r[c][nr - 1] - used[c, nr], 0)
    return result


def generate_hands(knowledge, used={}):
    if len(knowledge) == 0:
        yield []
        return

    for other in generate_hands(knowledge[1:], used):
        for col in ALL_COLORS:
            for i, cnt in enumerate(knowledge[0][col]):
                if cnt > 0:

                    result = [(col, i + 1)] + other
                    ok = True
                    thishand = {}
                    for (c, n) in result:
                        if (c, n) not in thishand:
                            thishand[(c, n)] = 0
                        thishand[(c, n)] += 1
                    for (c, n) in thishand:
                        if used[(c, n)] + thishand[(c, n)] > COUNTS[n - 1]:
                            ok = False
                    if ok:
                        yield result


def generate_hands_simple(knowledge, used={}):
    if len(knowledge) == 0:
        yield []
        return
    for other in generate_hands_simple(knowledge[1:]):
        for col in ALL_COLORS:
            for i, cnt in enumerate(knowledge[0][col]):
                if cnt > 0:
                    yield [(col, i + 1)] + other


def format_intention(i):
    if isinstance(i, str):
        return i
    if i == PLAY:
        return "Play"
    elif i == DISCARD:
        return "Discard"
    elif i == CANDISCARD:
        return "Can Discard"
    return "Keep"


def whattodo(knowledge, pointed, board):
    possible = get_possible(knowledge)
    play = potentially_playable(possible, board)
    discard = potentially_discardable(possible, board)

    if play and pointed:
        return PLAY
    if discard and pointed:
        return DISCARD
    return None


def pretend(action, knowledge, intentions, hand, board):
    (type, value) = action
    positive = []
    haspositive = False
    change = False
    if type == HINT_COLOR:
        newknowledge = []
        for i, (col, num) in enumerate(hand):
            positive.append(value == col)
            newknowledge.append(hint_color(knowledge[i], value, value == col))
            if value == col:
                haspositive = True
                if newknowledge[-1] != knowledge[i]:
                    change = True
    else:
        newknowledge = []
        for i, (col, num) in enumerate(hand):
            positive.append(value == num)

            newknowledge.append(hint_rank(knowledge[i], value, value == num))
            if value == num:
                haspositive = True
                if newknowledge[-1] != knowledge[i]:
                    change = True
    if not haspositive:
        return False, 0, ["Invalid hint"]
    if not change:
        return False, 0, ["No new information"]
    score = 0
    predictions = []
    pos = False
    for i, c, k, p in zip(intentions, hand, newknowledge, positive):

        action = whattodo(k, p, board)

        if action == PLAY and i != PLAY:
            # print "would cause them to play", f(c)
            return False, 0, predictions + [PLAY]

        if action == DISCARD and i not in [DISCARD, CANDISCARD]:
            # print "would cause them to discard", f(c)
            return False, 0, predictions + [DISCARD]

        if action == PLAY and i == PLAY:
            pos = True
            predictions.append(PLAY)
            score += 3
        elif action == DISCARD and i in [DISCARD, CANDISCARD]:
            pos = True
            predictions.append(DISCARD)
            if i == DISCARD:
                score += 2
            else:
                score += 1
        else:
            predictions.append(None)
    if not pos:
        return False, score, predictions
    return True, score, predictions


HINT_VALUE = 0.5


def pretend_discard(act, knowledge, board, trash):
    which = copy.deepcopy(knowledge[act.cnr])
    for (col, num) in trash:
        if which[col][num - 1]:
            which[col][num - 1] -= 1
    for col in ALL_COLORS:
        for i in range(board[col][1]):
            if which[col][i]:
                which[col][i] -= 1
    possibilities = sum(map(sum, which))
    expected = 0
    terms = []
    for col in ALL_COLORS:
        for i, cnt in enumerate(which[col]):
            rank = i + 1
            if cnt > 0:
                prob = cnt * 1.0 / possibilities
                if board[col][1] >= rank:
                    expected += prob * HINT_VALUE
                    terms.append((col, rank, cnt, prob, prob * HINT_VALUE))
                else:
                    dist = rank - board[col][1]
                    if cnt > 1:
                        value = prob * (6 - rank) / (dist * dist)
                    else:
                        value = 6 - rank
                    if rank == 5:
                        value += HINT_VALUE
                    value *= prob
                    expected -= value
                    terms.append((col, rank, cnt, prob, -value))
    return (act, expected, terms)


def format_knowledge(k):
    result = ""
    for col in ALL_COLORS:
        for i, cnt in enumerate(k[col]):
            if cnt > 0:
                result += COLORNAMES[col] + " " + str(i + 1) + ": " + str(cnt) + "\n"
    return result
