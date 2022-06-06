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

xL = ["low (< .25)", "medium (.25 - .75)", "high (> .75)"]
yL = [[],[],[]]
ind = ["chief"]

for filename in ind:

	with open("thesis_results/chief_results_" + filename, "rb") as f:
		print(pickle.load(f))
		logs = pickle.load(f)

	for game in logs:
		num_turns = len(game["details"])
		conf = game["details"][num_turns-1]["confidences"][num_turns-1][id_to_idxs[game["teammate"]]]

		if conf < 0.25:
			yL[0].append(game["score"])
		elif conf > 0.75:
			yL[2].append(game["score"])
		else:
			yL[1].append(game["score"])

plt.bar(xL, [np.mean(y) for y in yL], yerr=[np.std(y) for y in yL], capsize=5)
plt.ylabel("Average score")
plt.title("Average score per final confidence in correct model")
plt.show()



### Results for smaller sections

###
# Environment Knowledge Management
###

# print(id_to_idxs)

# PLOT = np.zeros(shape=(6,40))
# PLOT_err = np.zeros(shape=(6,40))

# for agent_teammate_ in ["00001", "00004", "10001"]:
# 	for i in range(1,7):
# 		with open("thesis_results/knowledge_results_" + str(i) + "_" + agent_teammate_, "rb") as f:
# 			print(pickle.load(f))
# 			logs = pickle.load(f)

# 		L = [[] for num in range(40)]

# 		for game in logs: # For a single game
# 			teamidx = id_to_idxs[game["teammate"]]

# 			for g in range(len(game["details"])):
# 				if i < 5: # samples comparison
# 					L[g].append(game["details"][g]["conditionals"][g][teamidx])
# 				else: # rollback comparison
# 					L[g].append(game["details"][-1]["conditionals"][g][teamidx])

# 		for k in range(40):
# 			if len(L[k]) < 40:
# 				PLOT[i-1][k] = None
# 				PLOT_err[i-1][k] = None
# 				continue

# 			PLOT[i - 1][k] = np.mean(L[k])
# 			PLOT_err[i-1][k] = np.var(L[k])
# 			# print(np.vectorize(lambda a : round(a,1))(L[k]), round(np.var(L[k]),1), round(np.mean(L[k]),1))

# 	LABELS = ["1 sample", "2 samples", "5 samples", "10 samples", "With knowledge rollbacks", "Without knowledge rollbacks"]


# 	plt.figure(figsize=(14,8))

# 	for plot_idx in range(4):
# 		y_axis = PLOT[plot_idx][~np.isnan(PLOT[plot_idx])]
# 		x_axis = [i for i in range(len(y_axis))]
# 		y_err = PLOT_err[plot_idx][~np.isnan(PLOT[plot_idx])]
# 		plt.fill_between(x_axis, y_axis - y_err, y_axis + y_err, alpha=0.3)
# 		plt.plot(x_axis, y_axis, label=LABELS[plot_idx])

# 	plt.legend()
# 	plt.ylim(0,1)
# 	plt.xlabel("Move")
# 	plt.ylabel("Likelihood of picking same action")
# 	plt.title(agent_teammate_ + "- likelihood of correct model picking same immediate action at each turn")
# 	plt.savefig("knowledge_results_" + agent_teammate_ + "_samples", dpi=96)

# 	plt.figure(figsize=(14,8))

# 	for plot_idx in range(4,6):
# 		y_axis = PLOT[plot_idx][~np.isnan(PLOT[plot_idx])]
# 		x_axis = [i for i in range(len(y_axis))]
# 		y_err = PLOT_err[plot_idx][~np.isnan(PLOT[plot_idx])]
# 		plt.fill_between(x_axis, y_axis - y_err, y_axis + y_err, alpha=0.3)
# 		plt.plot(x_axis, y_axis, label=LABELS[plot_idx])

# 	plt.legend()
# 	plt.ylim(0,1)
# 	plt.xlabel("Move")
# 	plt.ylabel("Likelihood of picking same action")
# 	plt.title(agent_teammate_ + "- likelihood of correct model picking same action given final knowledge")
# 	plt.savefig("knowledge_results_" + agent_teammate_ + "_rollback", dpi=96)


###
# Teammate Belief Processing results
###

# print(id_to_idxs)

# PLOT = np.zeros(shape=(10,40))
# PLOT_err = np.zeros(shape=(10,40))

# for agent_teammate_ in ["00001", "00004", "10001"]:
# 	with open("thesis_results/knowledge_results_3" + "_" + agent_teammate_, "rb") as f: # Using the runs with 5 samples and knowledge rollback on
# 		print(pickle.load(f))
# 		logs = pickle.load(f)

# 	L = {}

# 	for id_ in pool_ids:
# 		L[id_] = [[] for num in range(40)]

# 	for game in logs: # For a single game
# 		teamidx = id_to_idxs[game["teammate"]]

# 		for g in range(len(game["details"])):
# 			for id_ in pool_ids:
# 				L[id_][g].append(game["details"][g]["confidences"][g][id_to_idxs[id_]])

# 			# if i < 5: # samples comparison
# 			# 	L[g].append(game["details"][g]["conditionals"][g][teamidx])
# 			# else: # rollback comparison
# 			# 	L[g].append(game["details"][-1]["conditionals"][g][teamidx])

# 	for id_ in pool_ids:
# 		for k in range(40):
# 			if len(L["00001"][k]) < 40:
# 				PLOT[id_to_idxs[id_]][k] = None
# 				PLOT_err[id_to_idxs[id_]][k] = None
# 				continue

# 			PLOT[id_to_idxs[id_]][k] = np.mean(L[id_][k])
# 			PLOT_err[id_to_idxs[id_]][k] = np.var(L[id_][k])
# 			# print(np.vectorize(lambda a : round(a,1))(L[k]), round(np.var(L[k]),1), round(np.mean(L[k]),1))


# 	plt.figure(figsize=(14,8))

# 	for model_idx in range(10):
# 		y_axis = PLOT[model_idx][~np.isnan(PLOT[model_idx])]
# 		x_axis = [i for i in range(len(y_axis))]
# 		y_err = PLOT_err[model_idx][~np.isnan(PLOT[model_idx])]
# 		plt.fill_between(x_axis, y_axis - y_err, y_axis + y_err, alpha=0.3)
# 		plt.plot(x_axis, y_axis, label=pool_ids[model_idx])

# 	plt.legend()
# 	plt.ylim(0,1)
# 	plt.xlabel("Move")
# 	plt.ylabel("Model confidence")
# 	plt.title("Actual teammate id: " + agent_teammate_ + "- confidence in each model")
# 	plt.savefig("belief_results_" + agent_teammate_, dpi=96)


###
# Pool of Models
###

# for id_ in pool_ids:
# 	with open("thesis_results/behavior_clone_results_" + id_, "rb") as f:
# 		x = pickle.load(f)
# 		L = []
# 		L.append(pickle.load(f))
# 		L.append(pickle.load(f))

# 	# Computing average BC accuracy when BC is driving
# 	# print(L[0][0])
# 	BC_data = L[0][1]
# 	BC_analysis = []

# 	for game in BC_data:
# 		for move in game["details"]:
# 			BC_analysis.append(move[0] == move[1])

# 	BC_driving_score = np.mean(BC_analysis)

# 	# print("BC driving -- accuracy:", np.mean(BC_analysis))

# 	# Computing average BC accuracy when Agent is driving
# 	# print(L[1][0])
# 	BC_data = L[1][1]
# 	BC_analysis = []

# 	for game in BC_data:
# 		for move in game["details"]:
# 			BC_analysis.append(move[0] == move[1])

# 	Agent_driving_score = np.mean(BC_analysis)

# 	print(id_, "&", round(Agent_driving_score,3), "&", round(BC_driving_score,3), "\\\\")

# 	# print("Agent driving -- accuracy:", np.mean(BC_analysis))