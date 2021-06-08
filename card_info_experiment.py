from Agents.cclr_player import CardCountingLeftRightPlayer
from Agents.fully_intentional_player import FullyIntentionalPlayer
from Agents.hardcode_player import HardcodePlayer
from Agents.experimental_player import ExperimentalPlayer
import pickle
import hanabi
import numpy as np
import matplotlib.pyplot as plt

file_name = "blank.csv"
pickle_file_name = "card_info"
pickle_file = open(pickle_file_name, "wb")

for i in range(100):
	P1 = HardcodePlayer("P1", 0)
	P2 = HardcodePlayer("P2", 1)
	G = hanabi.Game([P1, P2], file_name, pickle_file)
	Result = G.run(100)

pickle_file.close()

CardDrawnMove = dict()
CardPrevKnowledge = dict()
PosToCard = dict()
CardToPos = dict()
new_id = 0

PlayDiscardDistances = []
HintDistances = []

def unequal(X, Y):
	return X != Y

with open(pickle_file_name, 'rb') as f:
	row = try_pickle(f)

	move_num = 0

	while(row != None):
		print(move_num, len(PlayDiscardDistances), len(HintDistances))

		if row[0] == "Action" and row[1].get_current_player() == 0:
			move_num += 1

			game_state = row[1]
			player_model = row[2]
			action = row[3]

			# Shift positions accordingly if a card was played or discarded
			if action.type in [hanabi.PLAY, hanabi.DISCARD]:
				newPosToCard = dict()
				PlayDiscardDistances.append(move_num - CardDrawnMove[PosToCard[action.cnr]])

				for idx in PosToCard:
					if idx == action.cnr:
						del CardToPos[PosToCard[idx]]
						del PosToCard[idx]

					if idx > action.cnr:
						CardToPos[PosToCard[idx]] -= 1
						PosToCard[idx - 1] = PosToCard[idx]
						del PosToCard[idx]

			# Add all unmapped cards to the three dictionaries
			for idx, hand in enumerate(player_model.get_knowledge()):
				if idx in PosToCard:
					continue

				card_id = new_id
				new_id += 1

				PosToCard[idx] = card_id
				CardToPos[card_id] = idx
				CardDrawnMove[card_id] = move_num
				CardPrevKnowledge[card_id] = player_model.get_knowledge()[idx]

			# For most recent update to info
			for idx in PosToCard:
				if (unequal(CardPrevKnowledge[PosToCard[idx]], player_model.get_knowledge()[idx])):
					HintDistances.append(move_num - CardDrawnMove[PosToCard[idx]])
					CardPrevKnowledge[PosToCard[idx]] = player_model.get_knowledge()[idx]

			

		row = try_pickle(f)


def summarize_list(L, name):
	print(name, "---------------------------------------")
	print("        # of times recorded:", len(L))
	print("        mean:", np.mean(L))
	print("        variance:", np.var(L))

summarize_list(PlayDiscardDistances, "Moves before play/discard")
summarize_list(HintDistances, "Moves from draw till hint")