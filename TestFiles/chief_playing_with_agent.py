import Agents
import pickle
import hanabi
from common_game_functions import *
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import random

args = sys.argv[1:]

if len(args) == 2 and args[0] == '--seed':
    print("setting seed", args[1])
    random.seed(int(args[1]))
    np.random.seed(int(args[1]))

file_name = "blank.csv"
pickle_file_name = "chief_testing"

pool_ids = ["00001","00002","00003","00004","00005","10001","10002","10003","10004","10005"]
# new_chief = ChiefPlayer("CHIEF", 0, "Agents/configs/players.json", pool_ids)

with open("Agents/configs/players.json", "r") as f:
    json_vals = json.load(f)

def try_pickle(file):
    try:
        return pickle.load(file)
    except:
        return None

def from_dict(name, pnr, json_dict):
    json_dict["name"] = name
    json_dict["pnr"] = pnr
    return getattr(Agents, json_dict["player_class"])(**json_dict)

L = {}

for i in range(100):
    pickle_file = open(pickle_file_name, "wb")
    l = []

    id_string = "00004"
    print("CHOSE AGENT: ", json_vals[id_string])

    # P1 = Agents.behavior_clone_player.BehaviorPlayer("BC", 0, agent_id="99994")
    P1 = from_dict("P1", 0, json_vals[id_string])
    P2 = from_dict("P2", 1, json_vals[id_string])
    pickle.dump(["NEW"], pickle_file)
    G = hanabi.Game([P1, P2], file_name, pickle_file)
    Result = G.run(100)

    l.append(Result)

    pickle_file.close()

    bc_hits = 0
    agent_hits = 0
    action_mismatches = 0
    action_count = 0

    dagger_yes_bc_no = 0
    dagger_no_bc_yes = 0
    dagger_no_bc_no = 0
    daggermis=0

    Agent_Ref = from_dict("a", 0, json_vals[id_string])
    Dagger_Ref = Agents.behavior_clone_player.BehaviorPlayer("d", 0, agent_id="99984")

    with open(pickle_file_name, 'rb') as f:
        row = try_pickle(f)

        while(row != None):
            if row[0] == "Action" and row[1].get_current_player() == 0:
                a = Agent_Ref.get_action(row[1], row[2])
                a2 = Dagger_Ref.get_action(row[1], row[2], row[4])
                
                action_mismatches += int(row[3] != a)
                action_count += 1

                if (row[3] != a2):
                    print("THISSHOULDNTHAPPEN")

                dagger_yes_bc_no += int(row[3] != a and a2 == a)
                dagger_no_bc_yes += int(row[3] == a and a2 != a)
                dagger_no_bc_no += int(row[3] != a and a2 != a)

                print()
                print("GREEN = 0 YELLOW = 1 WHITE = 2 BLUE = 3 RED = 4")
                print("Source would've done", a)
                print("Dagger would've done", a2)

            if row[0] == "Inform" and row[4] == 1:
                game_state = row[1]
                player_model = row[2]
                action = row[3]
                curr_player = row[5]

                print("P1 hand:", game_state.hands[0])

            if row[0] == "Inform" and row[4] == 0:
                game_state = row[1]
                player_model = row[2]
                action = row[3]
                curr_player = row[5]

                print("Player", curr_player, "did", action)
                print(game_state.board)
                print("P2 hand:", game_state.hands[1])

                Agent_Ref.inform(row[3], row[5], row[1], row[2])
                Dagger_Ref.inform(row[3], row[5], row[1], row[2])

                if curr_player == 0: # Behavior clone did action
                    if action.type == PLAY and len(game_state.trash) > 0 and game_state.card_changed == game_state.trash[-1]:
                        bc_hits += 1
                else:
                    if action.type == PLAY and len(game_state.trash) > 0 and game_state.card_changed == game_state.trash[-1]:
                        agent_hits += 1

            row = try_pickle(f)

    l.append(bc_hits)
    l.append(agent_hits)
    l.append(action_mismatches/action_count)
    l.append(dagger_yes_bc_no/action_count)
    l.append(dagger_no_bc_yes/action_count)
    l.append(dagger_no_bc_no/action_count)

    if id_string in L:
        L[id_string].append(l)
    else:
        L[id_string] = [l]

print(np.array(L["00004"]).mean(axis=0))
print(daggermis)

with open(pickle_file_name, "wb") as f:
    pickle.dump(L, f)