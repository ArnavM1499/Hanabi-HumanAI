import numpy as np


def separate(in_file, out_file_prefix, num):
    file_counter = 0
    fout = open(out_file_prefix + str(file_counter), "ab+")
    with open(in_file, "rb") as fin:
        it = 0
        while True:
            try:
                np.save(fout, np.load(fin))
                it += 1
                if it % num == 0:
                    fout.close()
                    file_counter += 1
                    fout = open(out_file_prefix + str(file_counter), "ab+")
            except ValueError:
                break


separate("../Data/00005_all.npy", "../Data/00005_parsed", 10000)