from hanabi import Game
from Agents.inner_state_player import InnerStatePlayer

P1 = InnerStatePlayer("player 1", 1)
P2 = InnerStatePlayer("player 2", 2)

G = Game([P1,P2])

Result = G.run(turns=50)
print(Result)