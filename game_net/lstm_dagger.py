from glob import glob
import numpy as np
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from lstm_net import LSTMNet
from utils import pkl_to_lstm_np

sys.path.insert(0, "..")
from TestFiles.test_player import test_player, generate_data  # noqa

torch.manual_seed(0)
torch.multiprocessing.set_sharing_strategy("file_system")
print(sys.argv)


AGENT = sys.argv[1]
BC_NAME = list(AGENT)
BC_NAME[1] = "9"
BC_NAME = "".join(BC_NAME)

GAME_STATE_LENGTH = 583 + 20 + 10 * 125

DATA_DIR = "../log/dagger_data/{}".format(AGENT)
DATA_DIR = os.path.abspath(DATA_DIR)
DATA_TRAIN = os.path.join(DATA_DIR, "{}_base_train.npy".format(AGENT))
DATA_VAL = os.path.join(DATA_DIR, "{}_base_val.npy".format(AGENT))
MODEL_PATH = "./models/model_lstm_{}.pth".format(AGENT)
WRITER_PATH = "runs/dagger_{}".format(os.path.basename(MODEL_PATH).replace(".pth", ""))

BATCH_SIZE = 512
EPOCH = 60

ROUNDS = 40
INCREMENT = 20000
MAX_GAMES = 100000
TEST_GAME = 500
THREADS = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = SummaryWriter(WRITER_PATH)


class PickleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file, model=None):
        self.states = []
        self.actions = []
        self.add_file(dataset_file)
        if model is not None:
            self.mapping = [
                (lambda x: x < 10, 20),
                (lambda x: 10 <= x and x < 15, 21),
                (lambda x: 15 <= x and x < 20, 22),
            ]
            self.update_labels(model)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            torch.tensor(self.actions[idx].size()[0]),
        )

    def _set_weights(self):
        count = {
            i: sum(x.to(torch.int32).tolist().count(i) for x in self.actions)
            for i in range(20)
        }
        nonzero = len([v for v in count.values() if v != 0])
        total = sum(nonzero / x for x in count.values() if x != 0)
        self.weights = torch.tensor(
            [1 if count[i] == 0 else total / count[i] for i in range(20)]
        )

    def add_file(self, dataset_file):
        count = len(self)
        head = 0
        with open(dataset_file, "rb") as fin:
            while True:
                try:
                    if count < MAX_GAMES:
                        self.states.append(
                            torch.from_numpy(np.load(fin) * 0.333).type(torch.float32)
                        )
                        self.actions.append(
                            torch.from_numpy(np.load(fin)).type(torch.long)
                        )
                        count += 1
                    else:
                        self.states[head] = torch.from_numpy(np.load(fin) * 0.333).type(
                            torch.float32
                        )
                        self.actions[head] = torch.from_numpy(np.load(fin)).type(
                            torch.long
                        )
                        head = (head + 1) % MAX_GAMES
                except ValueError:
                    break
        print("current dataset size: ", len(self))
        self._set_weights()

    def update_labels(self, model):
        model.eval()
        with torch.no_grad():
            for i, (state, action) in enumerate(zip(tqdm(self.states), self.actions)):
                (state, action, length) = pack_games(
                    [[state, action, torch.tensor(action.size()[0])]]
                )
                pred = model(state.to(DEVICE), length).data.argmax(1).cpu()
                for j, (p, l) in enumerate(zip(pred, self.actions[i])):
                    for func, new_label in self.mapping:
                        if func(l) and (not func(p)):
                            self.actions[i][j] = new_label


def pack_games(games):
    games.sort(key=lambda x: -x[-1])
    padded_states = torch.nn.utils.rnn.pad_sequence([x[0] for x in games])
    packed_actions = torch.nn.utils.rnn.pack_sequence([x[1] for x in games])
    return (padded_states, packed_actions, [x[-1] for x in games])


num_units = 512
model = LSTMNet([num_units], num_units, 2, [], drop_out=True, drop_out_rate=0.3).to(
    DEVICE
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


valset_base = torch.utils.data.DataLoader(
    PickleDataset(DATA_VAL),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=pack_games,
)


def val(log_iter=0, include_cat=False, run_game=False):
    print("evaluation for timestamp:", log_iter)
    losses = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (states, actions, lengths) in enumerate(tqdm(valset_base)):
            states, actions = states.to(DEVICE), actions.to(DEVICE)
            pred = model(states, lengths)
            loss = loss_fn(pred.data, actions.data)
            losses.append(loss.item())
            correct += (
                (pred.data.argmax(1) == actions.data).type(torch.float).sum().item()
            )
            total += actions.data.shape[0]
    loss = round(sum(losses) / len(losses), 5)
    accuracy = round(correct / total, 5)
    if include_cat:
        hint, play, discard, cat_accuracy = eval_model("", cat3=True, load_model=False)
        LOGGER.add_scalar("Cat/Hint", hint, log_iter)
        LOGGER.add_scalar("Cat/Play", play, log_iter)
        LOGGER.add_scalar("Cat/Discard", discard, log_iter)
        LOGGER.add_scalar("Cat/Accuracy", cat_accuracy, log_iter)
    LOGGER.add_scalar("Loss/Val", loss, log_iter)
    LOGGER.add_scalar("Accuracy/Val", accuracy, log_iter)
    print("val loss: ", loss, " accuracy: ", accuracy)
    if run_game:
        result1 = test_player(AGENT, BC_NAME, TEST_GAME // 2)
        result2 = test_player(BC_NAME, AGENT, TEST_GAME // 2)
        savg = (result1[1] + result2[1]) / 2
        smin = (result1[2] + result2[2]) / 2
        smax = (result1[3] + result2[3]) / 2
        hits = (result1[7] + result2[7]) / 2
        LOGGER.add_scalar("Game/Average", savg, log_iter)
        LOGGER.add_scalar("Game/Min", smin, log_iter)
        LOGGER.add_scalar("Game/Max", smax, log_iter)
        LOGGER.add_scalar("Game/Hits", hits, log_iter)
    model.train()


def train():
    traindata = PickleDataset(DATA_TRAIN)
    trainset = torch.utils.data.DataLoader(
        traindata,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pack_games,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=traindata.weights.to(DEVICE))
    size = len(trainset)
    for r in range(1, ROUNDS + 1):
        for e in range(EPOCH):
            val((r - 1) * EPOCH * size + e * size, include_cat=True, run_game=True)
            for i, (states, actions, lengths) in enumerate(
                tqdm(trainset, desc="epoch: {}".format(e))
            ):
                states, actions = states.to(DEVICE), actions.to(DEVICE)
                pred = model(states, lengths)
                loss = loss_fn(pred.data, actions.data)
                accuracy = (
                    (pred.data.argmax(1) == actions.data).type(torch.float).mean()
                )
                LOGGER.add_scalar("Loss/Train", loss.item(), e * size + i)
                LOGGER.add_scalar("Accuracy/Train", accuracy.item(), e * size + i)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            torch.save(
                model.state_dict(),
                MODEL_PATH.replace(".pth", "_{}.pth".format(str(r).zfill(2))),
            )
        print("generating new data for round", r)
        generate_data(
            AGENT,
            DATA_DIR,
            BC_NAME,
            INCREMENT // 2 // THREADS,
            THREADS,
            "subprocess",
            r * 100,
        )
        generate_data(
            BC_NAME,
            DATA_DIR,
            AGENT,
            INCREMENT // 2 // THREADS,
            THREADS,
            "subprocess",
            r * 100,
        )
        round_id = str(r).zfill(2)
        pkl_to_lstm_np(
            DATA_DIR,
            *glob(os.path.join(DATA_DIR, "*_*_{}*.pkl".format(round_id))),
            train_split=1,
            suffix="_" + round_id,
        )
        print("loading new data for round", r)
        traindata.add_file(
            os.path.join(DATA_DIR, "{}_{}_train.npy".format(AGENT, round_id))
        )
        trainset = torch.utils.data.DataLoader(
            traindata,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=pack_games,
        )


def eval_model(save_matrix="", cat3=False, load_model=True):
    if load_model:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    corrects = [0] * 20
    totals = [0] * 20
    matrix = [[0 for i in range(20)] for j in range(20)]
    with torch.no_grad():
        for i, (states, actions, lengths) in enumerate(tqdm(valset_base)):
            states, actions = states.to(DEVICE), actions.to(DEVICE)
            pred = model(states, lengths)
            if cat3:
                for label_lo, label_hi, extra in [
                    (0, 10, 20),
                    (10, 15, 21),
                    (15, 20, 22),
                ]:
                    for label in range(label_lo, label_hi):
                        mask = actions.data == label
                        masked_pred = pred.data[mask, :]
                        masked_label = torch.masked_select(actions.data, mask)
                        if int(masked_label.shape[0]) > 0:
                            predicted = masked_pred.argmax(1).cpu()
                            corrects[label] += (
                                np.logical_or(
                                    np.logical_and(
                                        predicted >= label_lo, predicted < label_hi
                                    ),
                                    predicted == extra,
                                )
                                .type(torch.float)
                                .sum()
                                .item()
                            )
                        totals[label] += masked_label.shape[0]
                        for p in range(20):
                            matrix[label][p] += (
                                (masked_pred.argmax(1) == p)
                                .type(torch.float)
                                .sum()
                                .item()
                            )
            else:
                for label in range(20):
                    mask = actions.data == label
                    masked_pred = pred.data[mask, :]
                    masked_label = torch.masked_select(actions.data, mask)
                    if int(masked_label.shape[0]) > 0:
                        corrects[label] += (
                            (masked_pred.argmax(1) == masked_label)
                            .type(torch.float)
                            .sum()
                            .item()
                        )
                    totals[label] += masked_label.shape[0]
                    for p in range(20):
                        matrix[label][p] += (
                            (masked_pred.argmax(1) == p).type(torch.float).sum().item()
                        )

    if save_matrix.endswith(".npy"):
        np.save(save_matrix, np.array(matrix, dtype=np.int32))
    hint = sum(corrects[:10]) / sum(totals[:10])
    play = sum(corrects[10:15]) / sum(totals[10:15])
    discard = sum(corrects[15:20]) / sum(totals[15:20])
    print("Recall:")
    print("  Hint Color: {:.4f}".format(sum(corrects[:5]) / sum(totals[:5])))
    print("  Hint Number: {:.4f}".format(sum(corrects[5:10]) / sum(totals[5:10])))
    print("  Hint Overall: {:.4f}".format(hint))
    print("  Play: {:.4f}".format(play))
    print("  Discard: {:.4f}".format(discard))
    return hint, play, discard, sum(corrects) / sum(totals)


if __name__ == "__main__":
    if sys.argv[2] == "train":
        train()
    elif sys.argv[2] == "eval":
        eval_model(sys.argv[3] if len(sys.argv) > 3 else "")
    elif sys.argv[2] == "eval_cat":
        eval_model(sys.argv[3] if len(sys.argv) > 3 else "", cat3=True)
    else:
        raise NotImplementedError
