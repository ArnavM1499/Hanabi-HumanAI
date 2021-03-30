from Agents.cclr_player import CardCountingLeftRightPlayer
from Agents.player import Action
import pickle
from hanabi import *
import numpy as np
#https://stackoverflow.com/questions/12761991/how-to-use-append-with-pickle-in-python

Agent_options = [CardCountingLeftRightPlayer("ccleft", 0, True, True), CardCountingLeftRightPlayer("ccright", 0, True, False), CardCountingLeftRightPlayer("noccleft", 0, False, True), CardCountingLeftRightPlayer("noccright", 0, False, False)]

file_name = "bayes_testing_ccleft.csv"
pickle_file_name = "bayes_testing_ccleft"
pickle_file = open(pickle_file_name, "wb")

for i in range(20):
	P1 = CardCountingLeftRightPlayer("P1", 0, False, True)
	P2 = CardCountingLeftRightPlayer("P2", 1, False, True)
	G = Game([P1, P2], file_name, pickle_file)
	Result = G.run(100)

pickle_file.close()

Option_probabilities = np.ones(4)
Option_probabilities /= 4.0
print(Option_probabilities)

def try_pickle(file):
	try:
		return pickle.load(file)
	except:
		return None


### ########
### Note: This will not work if the player stores internal knowledge. We will need to set up players to be able to only store changes after they've run 100 times
### ########

Test = []
Test1 = []

with open(pickle_file_name, 'rb') as f:
	row = try_pickle(f)
	alpha = 0.75

	while(row != None):
		if row[0] == 0:
			game_state = GameState(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
			player_model = BasePlayerModel(row[8], row[9], row[10])
			action = Action(row[11], row[12], row[13], row[14], row[15])
			temp_probabilities = np.zeros(4)

			for i in range(4):
				choices = []

				for j in range(20):
					choices.append(Agent_options[i].get_action(game_state, player_model))

				choices = np.array(choices)
				s = max(np.sum(choices == action), 0.001)
				temp_probabilities[i] = s/len(choices) ## This only works if the option and the actual agent have the same pnr, so make sure they do

			Test.append(temp_probabilities[2])
			Test1.append(temp_probabilities[3])
			Option_probabilities = alpha*Option_probabilities + (1 - alpha)*Option_probabilities*temp_probabilities
			Option_probabilities = Option_probabilities/sum(Option_probabilities)

		row = try_pickle(f)

	print(Option_probabilities, np.mean(Test), np.mean(Test1))




# [8.97365266e-01 1.00684430e-01 7.21166815e-14 1.95030415e-03]
# [1.71623519e-05 7.18843241e-02 1.47694461e-02 9.13329067e-01]
# [1.00000000e+00 2.46519457e-16 2.15863212e-20 1.05259004e-21]
# [1.87187019e-08 9.99999981e-01 2.29032285e-21 5.41213699e-15]
# [7.88299011e-14 5.44776248e-03 9.94552238e-01 2.46566640e-13]
# [3.23918180e-04 1.58512942e-06 9.69128417e-01 3.05460798e-02]
# [1.00000000e+00 8.34361783e-18 5.23849779e-11 2.01059575e-22]
# [1.31636659e-10 1.39447143e-11 1.74324293e-10 1.00000000e+00]


