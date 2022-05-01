import pyximport; pyximport.install(language_level=3)
from Agents.ChiefAgent.chief_player import ChiefPlayer
from Agents.behavior_clone_player import BehaviorPlayer
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

if len(args) == 2 and args[0] == "-seed":
    random.seed(int(args[1]))
    np.random.seed(int(args[1]))

file_name = "blank.csv"
pickle_file_name = "chief_testing"

pool_ids = ["00001","00002","00003","00004","00005","10001","10002","10003","10004","10005"]


def from_dict(name, pnr, json_dict):
    json_dict["name"] = name
    json_dict["pnr"] = pnr
    return getattr(Agents, json_dict["player_class"])(**json_dict)


with open("Agents/configs/players.json", "r") as f:
    json_vals = json.load(f)

def try_pickle(file):
	try:
		return pickle.load(file)
	except:
		return None

with open("resultlog", "a") as f:
	print("clearing resultlog", file=sys.stderr)

def run_n_games(n, mirror, general, id_strings):
	thesis_log = []

	for i in range(n):
		id_string = id_strings[i]

		if (i%5 == 0):
			print(i, file=sys.stderr)
		
		if not mirror:
			if np.random.rand() < 0.5:
				chief_idx = 0
				P1 = ChiefPlayer("CHIEF", 0, pool_ids, general=general)
				P2 = from_dict("Teammate", 1, json_vals[id_string])
			else:
				chief_idx = 1
				P1 = from_dict("Teammate", 0, json_vals[id_string])
				P2 = ChiefPlayer("CHIEF", 1, pool_ids, general=general)
		else:
			P1 = from_dict("Player 1", 0, json_vals[id_string])
			P2 = from_dict("Player 2", 1, json_vals[id_string])
		
		pickle_file = open(pickle_file_name, "wb")
		pickle.dump(["NEW"], pickle_file)
		G = hanabi.Game([P1, P2], file_name, pickle_file)
		Result = G.run(100)
		pickle_file.close()

		if mirror:
			thesis_log.append({"score":Result, "teammate": id_string})
		else:
			thesis_log.append({"details":[P1,P2][chief_idx].get_result_log(), "score":Result, "teammate": id_string})

	return thesis_log

id_strings = [np.random.choice(pool_ids) for x in range(50)]
print(id_strings, file=sys.stderr)
PER_PARAM_LOG = run_n_games(50, False, False, id_strings)

with open("thesis_results/chief_results_chief", "wb") as f:
	pickle.dump("CHIEF", f)
	pickle.dump(PER_PARAM_LOG, f)
	print("thesis_results/chief_results_chief generated", file=sys.stderr)


PER_PARAM_LOG = run_n_games(50, False, True, id_strings)

with open("thesis_results/chief_results_general", "wb") as f:
	pickle.dump("General", f)
	pickle.dump(PER_PARAM_LOG, f)
	print("thesis_results/chief_results_general generated", file=sys.stderr)


PER_PARAM_LOG = run_n_games(50, True, False, id_strings)

with open("thesis_results/chief_results_mirror", "wb") as f:
	pickle.dump("Mirror", f)
	pickle.dump(PER_PARAM_LOG, f)
	print("thesis_results/chief_results_mirror generated", file=sys.stderr)

