from Agents.hardcode_player import HardcodePlayer2
from Agents.value_player import ValuePlayer
from Agents.experimental_player import ExperimentalPlayer
from Agents.ChiefAgent.chief_player import ChiefPlayer
import pickle
import hanabi
import numpy as np
import matplotlib.pyplot as plt

new_chief = ChiefPlayer("chief", 0, "Agents/configs/players.json")

file_name = "blank.csv"
pickle_file_name = "chief_testing"
pickle_file = open(pickle_file_name, "wb")

for i in range(1):
	P1 = ValuePlayer("P1", 0, hint_weight=50)
	P2 = ValuePlayer("P2", 1, hint_weight=50)
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

			print("player",curr_player,"does",action)
			new_chief.inform(action, curr_player, game_state, player_model)
			print(new_chief.move_tracking_table.loc[:,("conditional probabilities", "agent distribution")].applymap(lambda L: [round(l,2) for l in L]))
			
			if (curr_player == 1):
				A.append(new_chief.move_tracking_table.tail(1)["agent distribution"])
				B.append(new_chief.move_tracking_table.tail(1)["MLE probabilities"])
				C.append(new_chief.move_tracking_table.tail(1)["conditional probabilities"])


		row = try_pickle(f)



# C = [c.values[0].tolist() for c in C[1:]]

# for idx, name in enumerate(["Hardcode", "Value", "Experimental"]):
# 	plt.plot([a[idx] for a in C], label=name)

# plt.legend()
# plt.show()


A = [a.values[0].tolist() for a in A[1:]]
B = [b.values[0].tolist() for b in B[1:]]


# for idx, name in enumerate(["Hardcode", "Value"]):
# 	plt.plot([a[idx] for a in A], label=name)

# plt.legend()
# plt.title("Trying to recognize" + type(P2).__name__ + "- bayesian w/ boltzmann")
# plt.savefig("chiefplots/highboltzmann/bayesian_" + type(P2).__name__)

valueplayerlist = new_chief.player_pool.copies()[:-1]

for idx, name in enumerate(["Value-hint_weight=" + str(v.hint_weight) for v in valueplayerlist] + ["Hardcode"]):
	plt.plot([b[idx] for b in A], label=name)

plt.legend()
plt.title("Trying to recognize Value-hint_weight=" + str(P2.hint_weight) + " - bayesian w/ boltzmann")
plt.show()