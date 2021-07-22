import sys
import os

cwd = os.getcwd()
game_net_dir = os.path.dirname(os.path.realpath(__file__))
if cwd != game_net_dir:
    sys.path.append(game_net_dir)
