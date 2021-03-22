from hanabi import Game
import csv
from Agents.inner_state_player import InnerStatePlayer
from Agents.cclr_player import CardCountingLeftRightPlayer
from Agents.hardcode_player import HardcodePlayer

P1 = HardcodePlayer("player 0", 0)
P2 = HardcodePlayer("palyer 1", 1)

file_name = 'hanabi_data_nocardcounting_left1.csv'

with open(file_name, 'w') as f:
	writer = csv.writer(f, delimiter=',')
	writer.writerow(["Player","Action Type","Board","Discards","Hints available","Knowledge from hints"])

results = []
num_games = 10000
for i in range(num_games):
    G = Game([P1,P2], file_name)
    Result = G.run(100)
    print(Result)
    results.append(Result)

results.sort()
print("{} games: avg: {}, min: {}, max: {}, median: {}".format(
    num_games, sum(results) / num_games, results[0], results[-1], results[num_games // 2]))
