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


### Results for main section

# xL = ["low (< .25)", "medium (.25 - .75)", "high (> .75)"]
# yL = [[],[],[]]
# ind = ["chief"]

# for filename in ind:

# 	with open("thesis_results/chief_results_" + filename, "rb") as f:
# 		print(pickle.load(f))
# 		logs = pickle.load(f)

# 	for game in logs:
# 		num_turns = len(game["details"])
# 		conf = game["details"][num_turns//2]["confidences"][num_turns//2][id_to_idxs[game["teammate"]]]

# 		if conf < 0.25:
# 			yL[0].append(game["score"])
# 		elif conf > 0.75:
# 			yL[2].append(game["score"])
# 		else:
# 			yL[1].append(game["score"])

# plt.bar(xL, [len(y) for y in yL])
# plt.ylabel("Number of games")
# plt.title("Number of games per halfway confidence in correct model")
# plt.show()



### Results for smaller sections

print(id_to_idxs)

PLOT = np.zeros(shape=(6,40))
PLOT_err = np.zeros(shape=(6,40))

for result_idx in range(1,7):
	with open("thesis_results/fixed_teammate_" + str(result_idx), "rb") as f:
		print(pickle.load(f))
		logs = pickle.load(f)

	L = [[] for num in range(40)]

	for game in logs: # For a single game
		teamidx = id_to_idxs[game["teammate"]]

		for i in range(len(game["details"])):
			L[i].append(game["details"][-1]["confidences"][i][teamidx])

	for k in range(40):
		if len(L[k]) < 15:
			PLOT[result_idx-1][k] = None
			PLOT_err[result_idx-1][k] = None
			continue

		PLOT[result_idx - 1][k] = np.mean(L[k])
		PLOT_err[result_idx-1][k] = np.var(L[k])
		print(np.vectorize(lambda a : round(a,1))(L[k]), round(np.var(L[k]),1), round(np.mean(L[k]),1))

LABELS = ["1 sample", "2 samples", "5 samples", "10 samples", "With knowledge rollbacks", "Without knowledge rollbacks"]


for plot_idx in range(4,6):
	y_axis = PLOT[plot_idx][~np.isnan(PLOT[plot_idx])]
	x_axis = [i for i in range(len(y_axis))]
	y_err = PLOT_err[plot_idx][~np.isnan(PLOT[plot_idx])]
	plt.fill_between(x_axis, y_axis - y_err, y_axis + y_err, alpha=0.3)
	plt.plot(x_axis, y_axis, label=LABELS[plot_idx])

plt.legend()
plt.ylim(0,1)
plt.xlabel("Move")
plt.ylabel("Confidence in correct model (probability)")
plt.title("Immediate confidence over time")
plt.show()
