from hanabi import Game
import csv
from Agents.inner_state_player import InnerStatePlayer
from Agents.cclr_player import CardCountingLeftRightPlayer

count_cards = False
left_to_right = True

P1 = CardCountingLeftRightPlayer("player 1", 1, count_cards, left_to_right)
P2 = CardCountingLeftRightPlayer("player 2", 2, count_cards, left_to_right)

file_name = "hanabi_data_nocardcounting_left1.csv"

with open(file_name, "w") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(
        [
            "Player",
            "Action Type",
            "Board",
            "Discards",
            "Hints available",
            "Knowledge from hints",
        ]
    )

for i in range(10):
    G = Game([P1, P2], file_name)

    Result = G.run(100)
    print(Result)
