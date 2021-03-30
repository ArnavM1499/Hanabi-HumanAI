from Agents.cclr_player import CardCountingLeftRightPlayer
from Agents.fully_intentional_player import FullyIntentionalPlayer
from Agents.hardcode_player import HardcodePlayer
from Agents.experimental_player import ExperimentalPlayer
from Agents.player import Action
import pickle
from hanabi import *
import numpy as np
import matplotlib.pyplot as plt
#https://stackoverflow.com/questions/12761991/how-to-use-append-with-pickle-in-python

Agent_options = [ExperimentalPlayer("experimental", 0), FullyIntentionalPlayer("intent", 0), HardcodePlayer("hardcode0_6", 0, 0.6), HardcodePlayer("hardcode0_1", 0, 0.1), HardcodePlayer("hardcode0_9", 0, 0.9), CardCountingLeftRightPlayer("ccleft", 0, True, True)]

file_name = "blank.csv"
pickle_file_name = "hardcode0_4"
pickle_file = open(pickle_file_name, "wb")

for i in range(3):
	# P1 = FullyIntentionalPlayer("P1", 0)
	# P2 = FullyIntentionalPlayer("P2", 1)
	# P1 = CardCountingLeftRightPlayer("P1", 0, True, True)
	# P2 = CardCountingLeftRightPlayer("P2", 0, True, True)
	P1 = HardcodePlayer("P1", 0, 0.4)
	P2 = HardcodePlayer("P2", 0, 0.4)
	G = Game([P1, P2], file_name, pickle_file)
	Result = G.run(1000)

pickle_file.close()

Option_counts = [0]*len(Agent_options)
Option_plot_x = []

for i in Agent_options:
	Option_plot_x.append([])

def try_pickle(file):
	try:
		return pickle.load(file)
	except:
		return None

with open(pickle_file_name, 'rb') as f:
	row = try_pickle(f)

	while(row != None):
		if row[0].get_current_player() == 0:
			game_state = row[0]
			player_model = row[1]
			action = row[2]

			for i in range(len(Agent_options)):
				a = Agent_options[i].get_action(game_state, player_model)
				
				if (action == a):
					Option_counts[i] += 1

				Option_plot_x[i].append(Option_counts[i])

		row = try_pickle(f)

	print(Option_counts)

x = list(range(len(Option_plot_x[0])))
idx = 0

for option in Option_plot_x:
	plt.plot(x, option, linestyle='solid', label=Agent_options[idx].name)
	idx += 1

plt.title("Actual agent: " + pickle_file_name)
plt.legend()
plt.show()