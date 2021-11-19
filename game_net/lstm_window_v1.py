import numpy as np
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")
print(sys.argv)


try:
    AGENT = sys.argv[1]
except IndexError:
    print("Agent not found! using default")
    AGENT = "00005"

GAME_STATE_LENGTH = 583 + 20  # base + extended (discardable / playable)

DATA_ALL = "../log/features0825/lstm_extended/{}_all.npy".format(AGENT)
DATA_TRAIN = DATA_ALL.replace("_all", "_train")
DATA_VAL = DATA_ALL.replace("_all", "_val")
MODEL_PATH = "../log/model_lstm_w10/model_lstm_{}.pth".format(AGENT)
WRITER_PATH = "runs/{}_w10".format(AGENT)

WINDOW = 10
BATCH_SIZE = 512
EPOCH = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = SummaryWriter(WRITER_PATH)


class PickleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file, window=-1):
        self.states = []
        self.actions = []
        with open(dataset_file, "rb") as fin:
            while True:
                try:
                    states = torch.from_numpy(np.load(fin) * 0.333).type(torch.float32)
                    actions = torch.from_numpy(np.load(fin))
                    length = actions.size()[0]
                    if window <= 0 or length <= window:
                        self.states.append(states)
                        self.actions.append(actions)
                    else:
                        for i in range(length - window):
                            self.states.append(states[i : i + window])
                            self.actions.append(actions[i : i + window])
                except ValueError:
                    break
        assert len(self.states) == len(self.actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx].type(torch.long),
            torch.tensor(self.actions[idx].size()[0]),
        )


def pack_games(games):
    games.sort(key=lambda x: -x[-1])
    padded_states = torch.nn.utils.rnn.pad_sequence([x[0] for x in games])
    packed_actions = torch.nn.utils.rnn.pack_sequence([x[1] for x in games])
    return (padded_states, packed_actions, [x[-1] for x in games])


class Net(torch.nn.Module):
    def __init__(
        self,
        input_fc_units,
        lstm_hidden_units,
        lstm_num_layers,
        output_fc_units,
        drop_out=False,
        drop_out_rate=0.5,
    ):
        super().__init__()
        self.input_fc = []
        for in_dim, out_dim in zip(
            [GAME_STATE_LENGTH] + input_fc_units, input_fc_units
        ):
            self.input_fc.append(torch.nn.Linear(in_dim, out_dim))
            self.input_fc.append(torch.nn.ReLU())
            if drop_out:
                self.input_fc.append(torch.nn.Dropout(drop_out_rate))
        self.input_fc = torch.nn.Sequential(*self.input_fc)
        self.lstm = torch.nn.LSTM(
            GAME_STATE_LENGTH if input_fc_units == [] else input_fc_units[-1],
            lstm_hidden_units,
            lstm_num_layers,
        )
        self.output_fc = []
        for in_dim, out_dim in zip(
            [lstm_hidden_units] + output_fc_units, output_fc_units
        ):
            self.output_fc.append(torch.nn.Linear(in_dim, out_dim))
            self.output_fc.append(torch.nn.ReLU())
            if drop_out:
                self.output_fc.append(torch.nn.Dropout(drop_out_rate))
        if output_fc_units == []:
            self.output_fc.append(torch.nn.Linear(lstm_hidden_units, 20))
        else:
            self.output_fc.append(torch.nn.Linear(output_fc_units[-1], 20))
        # self.output_fc.append(torch.nn.Softmax())
        self.output_fc = torch.nn.Sequential(*self.output_fc)

    def forward(self, padded, lengths):
        padded_output = self.input_fc(padded)
        packed_output = torch.nn.utils.rnn.pack_padded_sequence(padded_output, lengths)
        packed_output, _ = self.lstm(packed_output)
        padded_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        padded_output = self.output_fc(padded_output)
        packed_output = torch.nn.utils.rnn.pack_padded_sequence(padded_output, lengths)
        return packed_output


num_units = 512
model = Net([num_units], num_units, 2, [], drop_out=True).to(DEVICE)
# model = Net([512], 512, 2, []).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


valset = torch.utils.data.DataLoader(
    PickleDataset(DATA_VAL, window=WINDOW),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=pack_games,
)


def val(log_iter=0):
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
    print("  val loss: ", loss, " accuracy: ", accuracy)
    LOGGER.add_scalar("Loss/Val", loss, log_iter)
    LOGGER.add_scalar("Accuracy/Val", accuracy, log_iter)
    model.train()


def train():
    trainset = torch.utils.data.DataLoader(
        PickleDataset(DATA_TRAIN, window=WINDOW),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pack_games,
    )
    size = len(trainset)
    for e in range(EPOCH):
        val(e * size)
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
        eval(sys.argv[3] if len(sys.argv) > 2 else "")
    else:
        raise NotImplementedError
