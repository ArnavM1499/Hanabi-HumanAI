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

id_string = "10001"
pool_index = 5

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

def decode_action(a):
	TYPE = ["Hint color", "Hint number", "Play", "Discard"]
	LAMBDAS = [(lambda a : a%5), (lambda a : a%5 + 1), (lambda a: a%5), (lambda a:a%5)]

	return TYPE[a//5] + "->" + str(LAMBDAS[a//5](a))

P2 = new_chief.player_pool.from_dict("P2", 1, json_vals[id_string])

DATA = {"prediction accuracy":[], "inference confidences":[], "prediction/action type":[]}

with open(pickle_file_name, 'rb') as f:
	row = try_pickle(f)

	while(row != None):
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
				print()
				print("Chief predicts", prediction, "which is", decode_action(prediction))
				print("Action was", new_chief.action_to_key(action), "which is", decode_action(new_chief.action_to_key(action)))
				DATA["prediction accuracy"].append(int(prediction == new_chief.action_to_key(action)))
				DATA["prediction/action type"].append((prediction//5, new_chief.action_to_key(action)//5))

				if len(new_chief.move_tracking_table) > 0:
					DATA["inference confidences"].append(list(new_chief.move_tracking_table.iloc[-1]["agent distribution"]))

			new_chief.inform(action, curr_player, game_state, player_model)

		row = try_pickle(f)

plt.figure(1)
Confidences = np.transpose(np.array(DATA["inference confidences"]))

for i in range(len(Confidences)):
	plt.plot(Confidences[i], label=list(new_chief.player_pool.get_player_dict().keys())[i])

plt.title("Inference confidence for each agent's clone - Source agent (not-clone) = " + id_string)
plt.legend()

plt.figure(2)

actual_play_list = [[],[]]
actual_discard_list = [[],[]]
actual_hint_list = [[],[]]
predicted_play_list = [[],[]]
predicted_discard_list = [[],[]]
predicted_hint_list = [[],[]]

for i in range(len(DATA["prediction accuracy"])):
	acc = DATA["prediction accuracy"][i]
	ptype, atype = DATA["prediction/action type"][i]

	if ptype in [0,1]:
		predicted_hint_list[0].append(acc)
		predicted_hint_list[1].append(i)
	elif ptype == 2:
		predicted_play_list[0].append(acc)
		predicted_play_list[1].append(i)
	else:
		predicted_discard_list[0].append(acc)
		predicted_discard_list[1].append(i)

	if atype in [0,1]:
		actual_hint_list[0].append(0.2)
		actual_hint_list[1].append(i)
	elif atype == 2:
		actual_play_list[0].append(0.2)
		actual_play_list[1].append(i)
	else:
		actual_discard_list[0].append(0.2)
		actual_discard_list[1].append(i)

plt.plot(actual_play_list[1], actual_play_list[0], 'r*', label="Actually play")
plt.plot(actual_discard_list[1], actual_discard_list[0], 'b*', label="Actually discard")
plt.plot(actual_hint_list[1], actual_hint_list[0], 'g*', label="Actually hint")
plt.plot(predicted_play_list[1], predicted_play_list[0], 'ro', label="Predicted play")
plt.plot(predicted_discard_list[1], predicted_discard_list[0], 'bo', label="Predicted discard")
plt.plot(predicted_hint_list[1], predicted_hint_list[0], 'go', label="Predicted hint")
plt.title("Prediction accuracies with type coloring")

plt.legend(loc=(.1,.5))


plt.show()


# idx = 1

# for d in DATA:
# 	plt.figure(idx)
# 	if d == "prediction accuracy":
# 		plt.plot(DATA[d], 'bo')
# 	else:
# 		plt.plot(DATA[d])
# 	idx += 1
# 	plt.title(d)

# plt.show()