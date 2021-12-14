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

for i in range(400):
	pickle_file = open(pickle_file_name, "wb")
	l = []

	id_string = np.random.choice(pool_ids)
	print("CHOSE AGENT: ", json_vals[id_string])

	# P1 = from_dict("P1", 0, json_vals[id_string])
	# P2 = from_dict("P2", 1, json_vals[id_string])
	# pickle.dump(["NEW"], pickle_file)
	# G = hanabi.Game([P1, P2], file_name, pickle_file)
	# Result = G.run(100)

	# l.append(Result)

	P1 = from_dict("BC", 0, json_vals[id_string[0] + "9" + id_string[2:]])
	P2 = from_dict("P2", 1, json_vals[id_string])
	pickle.dump(["NEW"], pickle_file)
	G = hanabi.Game([P1, P2], file_name, pickle_file)
	Result = G.run(100)

	l.append(Result)

	pickle_file.close()

	bc_hits = 0
	agent_hits = 0

	with open(pickle_file_name, 'rb') as f:
		row = try_pickle(f)

		while(row != None):
			if row[0] == "Inform" and row[4] == 0:
				game_state = row[1]
				player_model = row[2]
				action = row[3]
				curr_player = row[5]

				if curr_player == 0: # Behavior clone did action
					if action.type == PLAY and len(game_state.trash) > 0 and game_state.card_changed == game_state.trash[-1]:
						bc_hits += 1
				else:
					if action.type == PLAY and len(game_state.trash) > 0 and game_state.card_changed == game_state.trash[-1]:
						agent_hits += 1

			row = try_pickle(f)

	l.append(bc_hits)
	l.append(agent_hits)

	if id_string in L:
		L[id_string].append(l)
	else:
		L[id_string] = [l]

with open(pickle_file_name, "wb") as f:
	pickle.dump(L, f)