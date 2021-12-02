from Agents.hardcode_player import HardcodePlayer2
from Agents.value_player import ValuePlayer
from Agents.experimental_player import ExperimentalPlayer
from Agents.ChiefAgent.chief_player import ChiefPlayer
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
pickle_file = open(pickle_file_name, "wb")

pool_ids = ["00001","00002","00003","00004","00005","10001","10002","10003","10004","10005"]
id_string = np.random.choice(pool_ids)
new_chief = ChiefPlayer("CHIEF", 0, "Agents/configs/players.json", pool_ids)

with open("Agents/configs/players.json", "r") as f:
    json_vals = json.load(f)

print("CHOSE AGENT: ", json_vals[id_string])

for i in range(1):
	P1 = new_chief
	P2 = new_chief.player_pool.from_dict("Teammate", 1, json_vals[id_string])
	pickle.dump(["NEW"], pickle_file)
	G = hanabi.Game([P1, P2], file_name, pickle_file)
	Result = G.run(100)

pickle_file.close()