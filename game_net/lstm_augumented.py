import numpy as np
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from lstm_net import LSTMNet

torch.manual_seed(0)
torch.multiprocessing.set_sharing_strategy("file_system")
print(sys.argv)


AGENT = sys.argv[1]

GAME_STATE_LENGTH = 583 + 20  # base + extended (discardable / playable)
GAME_STATE_LENGTH = 728
GAME_STATE_LENGTH = 583 + 20 + 10 * 125

DATA_ALL = "../log/features0825/lstm_extended/{}_all.npy".format(AGENT)
DATA_ALL = "../log/jcdata/np1008/{}_all.npy".format(AGENT)
DATA_ALL = "../00005-allhintknowledges-100k/00005_all.npy"
DATA_TRAIN = DATA_ALL.replace("_all", "_train")
DATA_VAL = DATA_ALL.replace("_all", "_val")
MODEL_PATH = "../log/model_lstm_jc/model_lstm_{}-lr0.001_augument.pth".format(AGENT)
MODEL_PATH = "../model/102221-singlestage-noaugment-trainhints.pth"
WRITER_PATH = "runs/{}".format(os.path.basename(MODEL_PATH))

BATCH_SIZE = 512
EPOCH = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = SummaryWriter(WRITER_PATH)


class PickleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file, model=None):
        self.states = []
        self.actions = []
        with open(dataset_file, "rb") as fin:
            while True:
                try:
                    self.states.append(
                        torch.from_numpy(np.load(fin) * 0.333).type(torch.float32)
                    )
                    self.actions.append(torch.from_numpy(np.load(fin)).type(torch.long))
                except ValueError:
                    break
        if model is not None:
            self.mapping = [
                (lambda x: x < 10, 20),
                (lambda x: 10 <= x and x < 15, 21),
                (lambda x: 15 <= x and x < 20, 22),
            ]
            self.update_labels(model)
        self.set_weights()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            torch.tensor(self.actions[idx].size()[0]),
        )

    def set_weights(self):
        count = {
            i: sum(x.to(torch.int32).tolist().count(i) for x in self.actions)
            for i in range(20)
        }
        nonzero = len([v for v in count.values() if v != 0])
        total = sum(nonzero / x for x in count.values() if x != 0)
        self.weights = torch.tensor(
            [1 if count[i] == 0 else total / count[i] for i in range(20)]
        )

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
model = LSTMNet([num_units], num_units, 2, [], drop_out=True).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


valset = torch.utils.data.DataLoader(
    PickleDataset(DATA_VAL),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=pack_games,
)


def val(log_iter=0, include_cat=False):
    losses = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (states, actions, lengths) in enumerate(tqdm(valset)):
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
    print("  val loss: ", loss, " accuracy: ", accuracy)
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
    for e in range(2 * EPOCH):
        val(e * size, include_cat=True)
        for i, (states, actions, lengths) in enumerate(
            tqdm(trainset, desc="epoch: {}".format(e))
        ):
            states, actions = states.to(DEVICE), actions.to(DEVICE)
            pred = model(states, lengths)
            loss = loss_fn(pred.data, actions.data)
            accuracy = (pred.data.argmax(1) == actions.data).type(torch.float).mean()
            LOGGER.add_scalar("Loss/Train", loss.item(), e * size + i)
            LOGGER.add_scalar("Accuracy/Train", accuracy.item(), e * size + i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), MODEL_PATH)
    # traindata = PickleDataset(DATA_TRAIN, model)
    # trainset = torch.utils.data.DataLoader(
    #     traindata,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     collate_fn=pack_games,
    # )
    # loss_fn = torch.nn.CrossEntropyLoss(weight=traindata.weights.to(DEVICE))
    # for e in range(EPOCH, 2 * EPOCH):
    #     val(e * size, include_cat=True)
    #     for i, (states, actions, lengths) in enumerate(
    #         tqdm(trainset, desc="epoch: {}".format(e))
    #     ):
    #         states, actions = states.to(DEVICE), actions.to(DEVICE)
    #         pred = model(states, lengths)
    #         loss = loss_fn(pred.data, actions.data)
    #         accuracy = (pred.data.argmax(1) == actions.data).type(torch.float).mean()
    #         LOGGER.add_scalar("Loss/Train", loss.item(), e * size + i)
    #         LOGGER.add_scalar("Accuracy/Train", accuracy.item(), e * size + i)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     torch.save(model.state_dict(), MODEL_PATH)


def eval_model(save_matrix="", cat3=False, load_model=True):
    if load_model:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    corrects = [0] * 20
    totals = [0] * 20
    matrix = [[0 for i in range(20)] for j in range(20)]
    with torch.no_grad():
        for i, (states, actions, lengths) in enumerate(tqdm(valset)):
            for label in range(20):
                states, actions = states.to(DEVICE), actions.to(DEVICE)
                pred = model(states, lengths)
                mask = actions.data == label
                masked_pred = pred.data[mask, :]
                masked_label = torch.masked_select(actions.data, mask)
                if int(masked_label.shape[0]) > 0:
                    if cat3:
                        predicted = masked_pred.argmax(1).cpu()
                        if label < 10:
                            corrects[label] += (
                                np.logical_or(predicted < 10, predicted == 20)
                                .type(torch.float)
                                .sum()
                                .item()
                            )
                        elif label < 15:
                            corrects[label] += (
                                np.logical_or(
                                    np.logical_and(predicted >= 10, predicted < 15),
                                    predicted == 21,
                                )
                                .type(torch.float)
                                .sum()
                                .item()
                            )
                        else:
                            corrects[label] += (
                                np.logical_or(
                                    np.logical_and(predicted >= 15, predicted < 20),
                                    predicted == 22,
                                )
                                .type(torch.float)
                                .sum()
                                .item()
                            )
                    else:
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