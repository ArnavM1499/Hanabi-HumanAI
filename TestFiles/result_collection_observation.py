import Agents
from Agents.ChiefAgent.chief_player import ChiefPlayer
import pickle
import hanabi
import numpy as np
import matplotlib.pyplot as plt
import json
from copy import deepcopy


with open("Agents/configs/players.json", "r") as f:
	json_vals = json.load(f)


def collect_run(pickle_file_name, agent_template1, agent_template2, file_name="blank.csv"):
	pickle_file = open(pickle_file_name,"wb")

	for i in range(1):
		P1 = deepcopy(agent_template1)
		P2 = deepcopy(agent_template2)
		pickle.dump(["NEW"], pickle_file)
		G = hanabi.Game([P1, P2], file_name, pickle_file)
		Result = G.run(100)

	pickle_file.close()

def try_pickle(file):
	try:
		return pickle.load(file)
	except:
		return None

def decode_action(a):
	TYPE = ["Hint color", "Hint number", "Play", "Discard"]
	LAMBDAS = [(lambda a : a%5), (lambda a : a%5 + 1), (lambda a: a%5), (lambda a:a%5)]

	return TYPE[a//5], LAMBDAS[a//5](a)

def analyze_game(pickle_file_name, data_collection_func, source_id, new_chief):
	data_set = []

	with open(pickle_file_name, 'rb') as f:
		row = try_pickle(f)

		while(row != None):
			if row[0] == "Action" and row[1].get_current_player() == new_chief.pnr:
				game_state = row[1]
				player_model = row[2]
				action = row[3]

				new_chief.get_action(game_state, player_model, action_default=action)

			elif row[0] == "Inform" and row[4] == new_chief.pnr:
				game_state = row[1]
				player_model = row[2]
				action = row[3]
				curr_player = row[5]

				if curr_player != new_chief.pnr:
					if new_chief.game_state_before_move == None:
						new_chief.generate_initial_state(game_state, player_model, action) # only for testing, chief would never have to have this called externally when actually playing
					
					prediction = new_chief.get_prediction()

				new_chief.inform(action, curr_player, game_state, player_model)

				if curr_player != new_chief.pnr:
					data_set.append(data_collection_func(source_id, new_chief, prediction, action))

			row = try_pickle(f)

	return data_set

def get_conditionals(source_id, new_chief, prediction, action):
	for idx, id_ in enumerate(list(new_chief.player_pool.get_player_dict().keys())):
		if source_id == id_:
			return new_chief.current_probabilities()["conditional probabilities"][idx]

def get_confidence(source_id, new_chief, prediction, action):
	for idx, id_ in enumerate(list(new_chief.player_pool.get_player_dict().keys())):
		if source_id == id_:
			return new_chief.current_probabilities()["agent likelihood"][idx]

def prediction_accuracy(source_id, new_chief, prediction, action, tag='A'):
	# Return: A, C, and AMH (Action-level, Category-level, and Action-level minus hint) based on tag
	if tag == 'A':
		return int(prediction == action)
	elif tag == 'C':
		return int(decode_action(prediction)[0] == decode_action(action)[0])
	else:
		prop1 = ("Hint" in decode_action(prediction)[0] and "Hint" in decode_action(action)[0])
		prop2 = (prediction == action)
		return int(prop1 or prop2)

def get_agents(should_match_type=True, should_match_id=True):
	class1 = ["00001","00002","00003"]
	class2 = ["00004","00005"]
	class3 = ["10001", "10002", "10003", "10004", "10005"]

	agent1_id = np.random.choice(class1 + class2 + class3)

	if should_match_id:
		agent2_id = agent1_id
	elif should_match_type:
		for c in [class1, class2, class3]:
			if agent1_id in c:
				agent2_id = np.random.choice(set(class2) - [agent1_id])
				break
	else:
		class_set = set(class1 + class2 + class3)

		for c in [class1, class2, class3]:
			if agent1_id in c:
				class_set -= c

		agent2_id = np.random.choice(class_set)

	chief_idx = np.random.choice([0,1])
	source_id = [agent1_id, agent2_id][chief_idx]

	json_dict = json_vals[agent1_id]
	json_dict["name"] = "P1"
	json_dict["pnr"] = 0
	agent1 = getattr(Agents, json_dict["player_class"])(**json_dict)

	json_dict = json_vals[agent2_id]
	json_dict["name"] = "P2"
	json_dict["pnr"] = 1
	agent2 = getattr(Agents, json_dict["player_class"])(**json_dict)

	return agent1, agent2, source_id, chief_idx
				


if __name__ == "__main__":
	import sys

	args = sys.argv[1:]

	if len(args) == 2 and args[0] == '--nruns':
		num_runs = int(args[1])
	else:
		num_runs = 50

	# Knowledge management results - effect of samples and knowledge rollbacks on accuracy of conditionals (TEST SEPARATE)
	SAMPLES = [1,5,10,50]
	AVOID_KNOWLEDGE_ROLLBACK = [True, False]

	# Belief processing results - showing how well bayesian inference works in certain scenarios
	MATCHING_AGENTS = [True, False]

	# Full agent results (can be collected over AGENT_IN_POOL as well)
	AGENT_IN_POOL = [True, False]

	DATA = {}

	DATA["Sample Conditionals"] = [[]]*len(SAMPLES) # Each entry for sample val v will be average conditional likelihood for agent's clone over n runs for a specific time-step
	DATA["Knowledge Rollback Conditionals"] = [[]]*len(AVOID_KNOWLEDGE_ROLLBACK) # same as above
	DATA["Inference Confidence of Source Clone"] = [[]]*len(MATCHING_AGENTS) # Each entry for list i will be average inference over n runs for specific time-step
	DATA["Action-level prediction accuracy"] = [[]]*len(AGENT_IN_POOL) # each entry for list i will be the average prediction accuracy for an entire game (so length n for n runs)
	DATA["Category-level prediction accuracy"] = [[]]*len(AGENT_IN_POOL)
	DATA["Action-level minus hint prediction accuracy"] = [[]]*len(AGENT_IN_POOL)

	ListKey = [(SAMPLES, "Sample Conditionals", get_conditionals),
			   (AVOID_KNOWLEDGE_ROLLBACK, "Knowledge Rollback Conditionals", get_conditionals),
			   (MATCHING_AGENTS, "Inference Confidence of Source Clone", get_confidence),
			   (AGENT_IN_POOL, "Action-level prediction accuracy", lambda s,n,p,a : prediction_accuracy(s,n,p,a,tag="A")),
			   (AGENT_IN_POOL, "Category-level prediction accuracy", lambda s,n,p,a : prediction_accuracy(s,n,p,a,tag="C")),
			   (AGENT_IN_POOL, "Action-level minus hint prediction accuracy", lambda s,n,p,a : prediction_accuracy(s,n,p,a,tag="AMH"))]


	for l,k,f in ListKey:
		for idx in range(len(l)):
			agg_data = []

			for run in range(num_runs):
				pool_ids = ["00001","00002","00003","00004","00005","10001", "10002", "10003", "10004", "10005"]

				if k == "Inference Confidence of Source Clone":
					agent1, agent2, source_id, player_num = get_agents(should_match_id=l[idx])
				else:
					agent1, agent2, source_id, player_num = get_agents()

				if "accuracy" in k and not l[idx]: # Agent not in pool
					pool_ids = list(set(pool_ids) - source_id)

				if k == "Sample Conditionals":
					chief_observer = ChiefPlayer("chief", player_num, "Agents/configs/players.json", pool_ids, num_samples=l[idx])
				elif k == "Knowledge Rollback Conditionals":
					chief_observer = ChiefPlayer("chief", player_num, "Agents/configs/players.json", pool_ids, avoid_knowledge_rollback=l[idx])
				else:
					chief_observer = ChiefPlayer("chief", player_num, "Agents/configs/players.json", pool_ids)
				
				pickle_file_name = "temp_pickle_storage"

				collect_run(pickle_file_name, agent1, agent2)
				data_point = analyze_game(pickle_file_name, f, source_id, chief_observer)

				agg_data.append(data_point)

			DATA[k][idx].append(np.array(agg_data).mean(axis=0))
			print("DATA COLLECTION LOG - ", DATA[k][idx], k, idx)

	with open("OBSERVATION_DATA.pkl", "wb") as f:
		pickle.dump(DATA, f)