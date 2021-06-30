from Agents.hardcode_player import HardcodePlayer2
from Agents.value_player import ValuePlayer
from Agents.experimental_player import ExperimentalPlayer
from Agents.ChiefAgent.chief_player import ChiefPlayer
import pickle
import hanabi
import numpy as np
import matplotlib.pyplot as plt


new_chief = ChiefPlayer("chief", 0, "agent_pool.json")

file_name = "blank.csv"
pickle_file_name = "chief_testing"
pickle_file = open(pickle_file_name, "wb")

for i in range(1):
	P1 = ValuePlayer("P1", 0)
	P2 = ValuePlayer("P2", 1)
	pickle.dump(["NEW"], pickle_file)
	G = hanabi.Game([P1, P2], file_name, pickle_file)
	Result = G.run(100)

pickle_file.close()

def try_pickle(file):
	try:
		return pickle.load(file)
	except:
		return None


with open(pickle_file_name, 'rb') as f:
	row = try_pickle(f)

	while(row != None):
		# print(new_chief.player_pool.get_names())

		if row[0] == "Inform" and row[4] == 1:
			print("indices", row[1].hinted_indices)
		

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

			print("player",curr_player,"does",action)

			new_chief.inform(action, curr_player, game_state, player_model)
			print(new_chief.move_tracking_table.loc[:,("agent distribution", "MLE probabilities")])


		row = try_pickle(f)

	print(new_chief.move_tracking_table.tail())