from Agents.cclr_player import CardCountingLeftRightPlayer
from Agents.fully_intentional_player import FullyIntentionalPlayer
from Agents.player import Action
import pickle
from hanabi import *
import numpy as np
#https://stackoverflow.com/questions/12761991/how-to-use-append-with-pickle-in-python

Agent_options = [CardCountingLeftRightPlayer("ccleft", 0, True, True), CardCountingLeftRightPlayer("ccright", 0, True, False), CardCountingLeftRightPlayer("noccleft", 0, False, True), CardCountingLeftRightPlayer("noccright", 0, False, False)]

file_name = "counting_noccleft.csv"
pickle_file_name = "counting_noccleft"
pickle_file = open(pickle_file_name, "wb")

for i in range(100):
	P1 = CardCountingLeftRightPlayer("P1", 0, False, True)
	P2 = CardCountingLeftRightPlayer("P2", 1, False, True)
	G = Game([P1, P2], file_name, pickle_file)
	Result = G.run(100)

pickle_file.close()

Option_probabilities = np.zeros(4)

def try_pickle(file):
	try:
		return pickle.load(file)
	except:
		return None

with open(pickle_file_name, 'rb') as f:
	row = try_pickle(f)

	while(row != None):
		if row[0] == 0:
			game_state = GameState(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
			player_model = BasePlayerModel(row[8], row[9], row[10])
			action = Action(row[11], row[12], row[13], row[14], row[15])

			for i in range(4):
				a = Agent_options[i].get_action(game_state, player_model)
				
				if (action == a):
					Option_probabilities[i] += 1

		row = try_pickle(f)

	print(Option_probabilities)