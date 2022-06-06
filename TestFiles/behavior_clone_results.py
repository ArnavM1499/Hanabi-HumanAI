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

class ComboPlayer(Agents.player.Player):
	def __init__(self, name, pnr, BC, Agent, driver = "BC"):
		self.name = name
		self.pnr = pnr
		self.BC = BC
		self.agent = Agent
		self.driver = driver # "BC" or "Agent"
		self.is_behavior_clone = True
		self.logger = []

	def get_action(self, game_state, base_player_model, partner_knowledge_model):
		bc_action = self.BC.get_action(game_state, base_player_model, partner_knowledge_model)
		agent_action = self.agent.get_action(game_state, base_player_model)

		self.logger.append((bc_action, agent_action))

		if self.driver == "BC":
			return bc_action
		else:
			return agent_action

	def inform(self, action, player, new_state, new_model):
		self.agent.inform(action, player, new_state, new_model)

	def get_logs(self):
		return self.logger

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

def run_n_games(n, id_string, driver):
	thesis_log = []

	for i in range(n):
		if (i%5 == 0):
			print(i, file=sys.stderr)
		
		if np.random.rand() < 0.5:
			chief_idx = 0

			BC = from_dict("BC-combo", 0, json_vals[id_string[0] + "9" + id_string[2:]])
			AGENT = from_dict("Agent-combo", 0, json_vals[id_string])
			P1 = ComboPlayer("Combo", 0, BC, AGENT, driver=driver)

			P2 = from_dict("Teammate", 1, json_vals[id_string])
		else:
			chief_idx = 1

			P1 = from_dict("Teammate", 0, json_vals[id_string])
			
			BC = from_dict("BC-combo", 1, json_vals[id_string[0] + "9" + id_string[2:]])
			AGENT = from_dict("Agent-combo", 1, json_vals[id_string])
			P2 = ComboPlayer("Combo", 1, BC, AGENT, driver=driver)
		
		pickle_file = open(pickle_file_name, "wb")
		pickle.dump(["NEW"], pickle_file)
		G = hanabi.Game([P1, P2], file_name, pickle_file)
		Result = G.run(100)
		pickle_file.close()

		thesis_log.append({"details":[P1,P2][chief_idx].get_logs(), "score":Result, "teammate": id_string})

	return thesis_log



for id_ in pool_ids:
	print("BC 50 games " + id_, file=sys.stderr)
	BC_driving = run_n_games(50, id_, "BC")
	print("Agent 50 games " + id_, file=sys.stderr)
	Agent_driving = run_n_games(50, id_, "Agent")

	with open("thesis_results/behavior_clone_results_" + id_, "wb") as f:
		pickle.dump(id_, f)
		pickle.dump(["BC driving", BC_driving], f)
		pickle.dump(["Agent driving", Agent_driving], f)
		print("thesis_results/behavior_clone_results_" + id_ + " generated", file=sys.stderr)