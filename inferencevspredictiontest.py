from Agents.hardcode_player import HardcodePlayer2
from Agents.value_player import ValuePlayer
from Agents.experimental_player import ExperimentalPlayer
from Agents.ChiefAgent.chief_player import ChiefPlayer
import pickle
import hanabi
import numpy as np
import matplotlib.pyplot as plt
import json

new_chief = ChiefPlayer("chief", 0, "Agents/configs/players.json")

file_name = "blank.csv"
pickle_file_name = "chief_testing"
# pickle_file = open(pickle_file_name, "wb")

id_string = "10004"
pool_index = 8

# with open("Agents/configs/players.json", "r") as f:
#     json_vals = json.load(f)

# for i in range(1):
# 	P1 = new_chief.player_pool.from_dict("P1", 0, json_vals[id_string])
# 	P2 = new_chief.player_pool.from_dict("P2", 1, json_vals[id_string])
# 	pickle.dump(["NEW"], pickle_file)
# 	G = hanabi.Game([P1, P2], file_name, pickle_file)
# 	Result = G.run(100)

# pickle_file.close()

def try_pickle(file):
	try:
		return pickle.load(file)
	except:
		return None

def decode_action(a):
	TYPE = ["Hint color", "Hint number", "Play", "Discard"]
	LAMBDAS = [(lambda a : a%5), (lambda a : a%5 + 1), (lambda a: a%5), (lambda a:a%5)]

	return TYPE[a//5] + "->" + str(LAMBDAS[a//5](a))


DATA = {"prediction accuracy":[], "inference confidence of source agent":[], "entropy of knowledge":[], "entropy of pool":[]}

with open(pickle_file_name, 'rb') as f:
	row = try_pickle(f)

	while(row != None):
		if row[0] == "Action" and row[1].get_current_player() == 0:
			game_state = row[1]
			player_model = row[2]
			action = row[3]

			new_chief.get_action(game_state, player_model, action_default=action)
			# print("chief does", action)

		elif row[0] == "Inform" and row[4] == 0:
			game_state = row[1]
			player_model = row[2]
			action = row[3]
			curr_player = row[5]

			if curr_player != new_chief.pnr:
				prediction = new_chief.get_prediction()
				print()
				print()
				print("Chief predicts", prediction, "which is", decode_action(prediction))
				print("Action was", new_chief.action_to_key(action), "which is", decode_action(new_chief.action_to_key(action)))
				DATA["prediction accuracy"].append(int(prediction == new_chief.action_to_key(action)))

			if len(new_chief.move_tracking_table) > 0:
				DATA["inference confidence of source agent"].append(new_chief.move_tracking_table.iloc[-1]["agent distribution"][pool_index])

			if new_chief.entropy_of_pool() >= 0:
				DATA["entropy of knowledge"].append(new_chief.entropy_of_knowledge())
			
			if new_chief.entropy_of_pool() >= 0:
				DATA["entropy of pool"].append(new_chief.entropy_of_pool())

			# print("player",curr_player,"does",action)
			new_chief.inform(action, curr_player, game_state, player_model)

		row = try_pickle(f)

idx = 1

for d in DATA:
	plt.figure(idx)
	if d == "prediction accuracy":
		plt.plot(DATA[d], 'bo')
	# elif d == "inference confidence of source agent":
	# 	plt.plot([np.mean(DATA[d][i:i+5]) for i in range(len(DATA[d]) - 5)])
	else:
		plt.plot(DATA[d])
	idx += 1
	plt.title(d)

plt.show()