import numpy as np
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from lstm_net import LSTMNet

torch.multiprocessing.set_sharing_strategy("file_system")
print(sys.argv)


AGENT = sys.argv[1]

GAME_STATE_LENGTH = 728  # base + extended (discardable / playable)


DATA_ALL = "../log/features0825/lstm_extended/{}_all.npy".format(AGENT)
DATA_TRAIN = DATA_ALL.replace("_all", "_train")
DATA_VAL = DATA_ALL.replace("_all", "_val")
MODEL_PATH = "../log/model_lstm_512/model_lstm_{}.pth".format(AGENT)
WRITER_PATH = "runs/{}".format(AGENT)

BATCH_SIZE = 512
EPOCH = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = SummaryWriter(WRITER_PATH)


class PickleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file):
        self.states = []
        self.actions = []
        # self.action_dict = []
        with open(dataset_file, "rb") as fin:
            while True:
                try:
                    self.states.append(
                        torch.from_numpy(np.load(fin) * 0.333).type(torch.float32)
                    )
                    # for vals in self.states[-1]:
                    #     for val2 in vals:
                    #         assert 0 <= val2 <= 1
                    self.actions.append(torch.from_numpy(np.load(fin)).type(torch.long))
                    assert len(self.actions[-1]) == len(self.states[-1])
                    # self.action_dict.append(torch.from_numpy(np.load(fin)).type(torch.float32))
                except ValueError:
                    break
        assert len(self.states) == len(self.actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            torch.tensor(self.actions[idx].size()[0]),
        )


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
    shuffle=True,
    collate_fn=pack_games
)


def val(log_iter=0):
    losses = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (states, action_values, lengths) in enumerate(tqdm(valset)):
            states, action_values = states.to(DEVICE), action_values.to(DEVICE)
            pred = model(states, lengths)
            loss = loss_fn(pred.data, action_values.data)
            losses.append(loss.item())
            correct += (
                (pred.data.argmax(1) == action_values.data).type(torch.float).sum().item()
            )
            total += action_values.data.shape[0]
    loss = round(sum(losses) / len(losses), 5)
    accuracy = round(correct / total, 5)
    print(" accuracy: ", accuracy)
    print(" val loss: ", loss)
    LOGGER.add_scalar("Loss/Val", loss, log_iter)
    LOGGER.add_scalar("Accuracy", accuracy, log_iter)
    model.train()


def train():
    trainset = torch.utils.data.DataLoader(
        PickleDataset(DATA_TRAIN),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pack_games
    )
    size = len(trainset)
    for e in range(EPOCH):
        val(e * size)
        for i, (states, action_values, lengths) in enumerate(
            tqdm(trainset, desc="epoch: {}".format(e))
        ):
            states, action_values = states.to(DEVICE), action_values.to(DEVICE)
            pred = model(states, lengths)
            loss = loss_fn(pred.data, action_values.data)
            LOGGER.add_scalar("Loss/Train", loss.item(), e * size + i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), MODEL_PATH)


def eval(save_matrix=""):
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
    print("Accuracy:")
    print("  Hint Color: {:.4f}".format(sum(corrects[:5]) / sum(totals[:5])))
    print("  Hint Number: {:.4f}".format(sum(corrects[5:10]) / sum(totals[5:10])))
    print("  Hint Overall: {:.4f}".format(sum(corrects[:10]) / sum(totals[:10])))
    print("  Play: {:.4f}".format(sum(corrects[10:15]) / sum(totals[10:15])))
    print("  Discard: {:.4f}".format(sum(corrects[15:20]) / sum(totals[15:20])))


if __name__ == "__main__":
    if sys.argv[2] == "train":
        train()
    elif sys.argv[2] == "eval":
        eval(sys.argv[3] if len(sys.argv) > 3 else "")
    else:
        raise NotImplementedError
