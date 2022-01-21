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

L = []

for i in range(50):
	id_string = np.random.choice(pool_ids)
	P1 = ChiefPlayer("CHIEF", 0, pool_ids)
	P2 = from_dict("Teammate", 1, json_vals[id_string])
	
	pickle_file = open(pickle_file_name, "wb")
	pickle.dump(["NEW"], pickle_file)
	G = hanabi.Game([P1, P2], file_name, pickle_file)
	Result = G.run(100)
	pickle_file.close()

	L.append((id_string, json_vals[id_string]["player_class"], Result))
	print(L[-1], file=sys.stderr)

print(np.mean([a[1] for a in L]), file=sys.stderr)
