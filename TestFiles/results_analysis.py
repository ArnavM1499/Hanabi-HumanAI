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

pool_ids = ["00001","00002","00003","00004","00005","10001","10002","10003","10004","10005"]

id_to_idxs = {}

for idx, id_ in enumerate(sorted(pool_ids)):
	id_to_idxs[id_] = idx

print(id_to_idxs)

PLOT = np.zeros(shape=(6,40))

for result_idx in range(1,7):
	with open("thesis_results/full_info" + str(result_idx), "rb") as f:
		print(pickle.load(f))
		logs = pickle.load(f)

	L = [[] for num in range(40)]

	for game in logs: # For a single game
		teamidx = id_to_idxs[game["teammate"]]

		for i in range(len(game["details"])):
			L[i].append(game["details"][i]["confidences"][i][teamidx])

	for k in range(40):
		if len(L[k]) < 15:
			PLOT[result_idx-1][k] = None
			continue

		PLOT[result_idx - 1][k] = np.mean(L[k])


for plot_idx in range(4,6):
	y_axis = PLOT[plot_idx][~np.isnan(PLOT[plot_idx])]
	x_axis = [i for i in range(len(y_axis))]
	plt.plot(x_axis, y_axis, label=str(plot_idx))

plt.legend()
plt.show()
