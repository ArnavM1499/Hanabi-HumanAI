from Agents.ChiefAgent.chief_player import ChiefPlayer
from Agents.behavior_clone_player import BehaviorPlayer
import Agents
import pickle
import hanabi
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import random

args = sys.argv[1:]

if len(args) == 2 and args[0] == '-seed':
	random.seed(int(args[1]))
	np.random.seed(int(args[1]))

file_name = "blank.csv"
pickle_file_name = "chief_testing"

# pool_ids = ["00001","00002","00003","00004","00005","10001","10002","10003","10004","10005"]
pool_ids = ["10001"]
id_string = np.random.choice(pool_ids)
new_chief = ChiefPlayer("CHIEF", 0, pool_ids)

def from_dict(name, pnr, json_dict):
    json_dict["name"] = name
    json_dict["pnr"] = pnr
    return getattr(Agents, json_dict["player_class"])(**json_dict)

with open("Agents/configs/players.json", "r") as f:
    json_vals = json.load(f)

print("CHOSE AGENT: ", json_vals[id_string])

L = []

for i in range(10):
	# P1 = new_chief
	P1 = BehaviorPlayer("BC",0,agent_id=id_string)
	P2 = from_dict("Teammate", 1, json_vals[id_string])
	
	pickle_file = open(pickle_file_name, "wb")
	pickle.dump(["NEW"], pickle_file)
	G = hanabi.Game([P1, P2], file_name, pickle_file)
	Result = G.run(100)
	pickle_file.close()

	L.append(Result)

print(np.mean(L), file=sys.stderr)