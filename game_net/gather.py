from fire import Fire
import numpy as np
import os
from tqdm import tqdm
from common_game_functions import decode_state


def gather_json(dataset_dir, *paths):
    for path in tqdm(paths):
        if not os.path.isfile(path):
            print(path, "Not Found!")
            continue
        (p1, p2, _) = tuple(path.split("_"))
        player = [p1, p2]
        for state in tqdm(open(path)):
            p, a, s = decode_state(state)
            subdir = os.path.join(dataset_dir, player[p])
            if not os.path.isdir(subdir):
                os.makedirs(subdir)
            np.save(
                open(os.path.join(subdir, "{}.npy".format(str(a).zfill(2))), "ab+"),
                np.array(s),
            )


if __name__ == "__main__":
    Fire()
