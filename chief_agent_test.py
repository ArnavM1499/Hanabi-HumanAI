from Agents.hardcode_player import HardcodePlayer2
from Agents.value_player import ValuePlayer
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
	P1 = HardcodePlayer2("P1", 0)
	P2 = HardcodePlayer2("P2", 1)
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

	PlayDiscardDistances = []
	HintDistances = []

	while(row != None):
		if row[0] == "Action" and row[1].get_current_player() == 0:
			game_state = row[1]
			player_model = row[2]
			action = row[3]

			new_chief.get_action(game_state, player_model, action_default=action)

		elif row[0] == "Inform" and row[4] == 0:
			game_state = row[1]
			player_model = row[2]
			action = row[3]
			curr_player = row[4]

			new_chief.inform(action, curr_player, game_state, player_model)


		row = try_pickle(f)

	print(new_chief.move_tracking_table.loc[99])