import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")

GAME_STATE_LENGTH = 558

DATA_ALL = "../log/features0722/lstm/10005_all.npy"
DATA_TRAIN = DATA_ALL.replace("_all", "_train")
DATA_VAL = DATA_ALL.replace("_all", "_val")
MODEL_PATH = "../log/model_lstm.pth"

BATCH_SIZE = 256
EPOCH = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = SummaryWriter()


class PickleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file):
        self.states = []
        self.actions = []
        with open(dataset_file, "rb") as fin:
            while True:
                try:
                    self.states.append(
                        torch.from_numpy(np.load(fin) * 0.333).type(torch.float32)
                    )
                    self.actions.append(torch.from_numpy(np.load(fin)))
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


model = Net([512], 512, 2, [], drop_out=True).to(DEVICE)
# model = Net([512], 512, 2, []).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


trainset = torch.utils.data.DataLoader(
    PickleDataset(DATA_TRAIN),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=pack_games,
)
valset = torch.utils.data.DataLoader(
    PickleDataset(DATA_VAL),
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


train()


import pdb  # noqa E402

pdb.set_trace()
