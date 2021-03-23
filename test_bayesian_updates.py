from Agents.cclr_player import CardCountingLeftRightPlayer

Agent_options = [CardCountingLeftRightPlayer("ccleft", 0, True, True), CardCountingLeftRightPlayer("ccright", 0, True, False), CardCountingLeftRightPlayer("noccleft", 0, False, True), CardCountingLeftRightPlayer("noccright", 0, False, False)]

P1 = CardCountingLeftRightPlayer("P1", 0, True, True)
P2 = CardCountingLeftRightPlayer("P2", 0, True, True)
G = Game([P1, P2], file_name)
Result = G.run(100)
