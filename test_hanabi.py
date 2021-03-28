from hanabi import Game
import csv
from Agents.inner_state_player import InnerStatePlayer
from Agents.cclr_player import CardCountingLeftRightPlayer
from Agents.basic_protocol_player import BasicProtocolPlayer
from Agents.hardcode_player import HardcodePlayer
from Agents.experimental_player import ExperimentalPlayer

P1 = ExperimentalPlayer("player 0", 0)
P2 = ExperimentalPlayer("player 1", 1)

# P1 = BasicProtocolPlayer("player 1", 0, {})
# P2 = BasicProtocolPlayer("player 2", 1, {})

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

results = []
hints = []
hits = []
turns = []
num_games = 1
for i in range(num_games):
    G = Game([P1, P2], file_name)
    Result = G.run(100)
    print(Result)
    results.append(Result)
    hints.append(G.hints)
    hits.append(G.hits)
    turns.append(G.turn)

results.sort()
print(
    "{} games: avg: {}, min: {}, max: {}, median: {}, mode: {}".format(
        num_games,
        sum(results) / num_games,
        results[0],
        results[-1],
        results[num_games // 2],
        max(set(results), key=results.count),
    ),
)
print(
    "average: hints left: {}, hits left: {}, turns: {}".format(
        sum(hints) / num_games, sum(hits) / num_games, sum(turns) / num_games
    )
)
