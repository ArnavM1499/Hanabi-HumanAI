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
pickle_file = open(pickle_file_name, "wb")

id_string = "10004"

with open("Agents/configs/players.json", "r") as f:
    json_vals = json.load(f)

for i in range(1):
	P1 = new_chief.player_pool.from_dict("P1", 0, json_vals[id_string])
	P2 = new_chief.player_pool.from_dict("P2", 1, json_vals[id_string])
	pickle.dump(["NEW"], pickle_file)
	G = hanabi.Game([P1, P2], file_name, pickle_file)
	Result = G.run(100)

pickle_file.close()

def try_pickle(file):
	try:
		return pickle.load(file)
	except:
		return None

B = []
A = []
C = []

DATA = {"prediction accuracy":[], "inference confidence":[], "entropy of knowledge":[], "entropy of pool":[]}

with open(pickle_file_name, 'rb') as f:
	row = try_pickle(f)

	while(row != None):
		# print(new_chief.player_pool.get_names()

		if row[0] == "Action" and row[1].get_current_player() == 0:
			game_state = row[1]
			player_model = row[2]
			action = row[3]

			new_chief.get_action(game_state, player_model, action_default=action)
			print("chief does", action)

		elif row[0] == "Inform" and row[4] == 0:
			game_state = row[1]
			player_model = row[2]
			action = row[3]
			curr_player = row[5]

			if curr_player != new_chief.pnr:
				prediction = new_chief.get_prediction()

			print("player",curr_player,"does",action)
			new_chief.inform(action, curr_player, game_state, player_model)

			






			# print(sorted(new_chief.player_pool.get_player_dict().keys()))
			# print(new_chief.move_tracking_table.loc[:,("agent distribution")].map(lambda L: [round(l,2) for l in L]))
			# print(new_chief.entropy_of_knowledge(), new_chief.entropy_of_knowledge(5))
			# print(new_chief.entropy_of_pool(), new_chief.entropy_of_pool(5))
			
			if (curr_player == 1):
				A.append(new_chief.move_tracking_table.tail(1)["agent distribution"])
				B.append(new_chief.move_tracking_table.tail(1)["MLE probabilities"])
				C.append(new_chief.move_tracking_table.tail(1)["conditional probabilities"])


		row = try_pickle(f)

for i, a in enumerate(sorted(new_chief.player_pool.get_player_dict().keys())):
	print(i, a, A)
	L = []

	for item in A:
		L.append(item.tolist()[0][i])

	plt.plot(L, label=a)

plt.title("Agents playing: parameter id" + id_string)

plt.legend(title="Behavior Clones of each parameter id")
plt.show()