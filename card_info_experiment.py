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
PosToCard = dict()
CardToPos = dict()
new_id = 0

with open(pickle_file_name, 'rb') as f:
	row = try_pickle(f)

	while(row != None):
		if row[0] == "Action":
			game_state = row[1]
			player_model = row[2]
			action = row[3]

			if action.type in [hanabi.PLAY, hanabi.DISCARD]:
				newPosToCard = dict()

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


			# For mapped cards, update Position dicts if needed



			# For most recent action regarding our own cards, add distance from drawn move

			

		row = try_pickle(f)

