from hanabi import Game
import csv
from Agents.inner_state_player import InnerStatePlayer
from Agents.cclr_player import CardCountingLeftRightPlayer
from Agents.basic_protocol_player import BasicProtocolPlayer

count_cards = False
left_to_right = True

P1 = BasicProtocolPlayer("player 1", 0, {})
P2 = BasicProtocolPlayer("player 2", 1, {})

file_name = 'hanabi_data_nocardcounting_left1.csv'

with open(file_name, 'w') as f:
	writer = csv.writer(f, delimiter=',')
	writer.writerow(["Player","Action Type","Board","Discards","Hints available","Knowledge from hints"])

for i in range(1):
	G = Game([P1,P2], file_name)

	Result = G.run(100)
	print(Result)
