import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")

GAME_STATE_LENGTH = 583 + 20

DATA_ALL = "../Data/00005_parsed0_all.npy"
DATA_TRAIN = DATA_ALL.replace("_all", "_train")
DATA_VAL = DATA_ALL.replace("_all", "_val")
MODEL_PATH = "../model/model_lstm_two_stage.pth"

BATCH_SIZE = 600
EPOCH = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = SummaryWriter()


class PickleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file):
        self.states = []
        self.action_cats = []
        self.action_hints = []
        self.action_cards = []
        with open(dataset_file, "rb") as fin:
            while True:
                try:
                    self.states.append(
                        torch.from_numpy(np.load(fin) * 0.333).type(torch.float32)
                    )
                    action_label = torch.from_numpy(np.load(fin)).type(torch.long)
                    buffer_cat = []
                    for idx in action_label:
                        idx = int(idx)
                        if idx < 10:  # hint color / number
                            buffer_cat.append(0)
                        elif idx < 15:  # play
                            buffer_cat.append(1)
                        else:  # discard
                            buffer_cat.append(2)
                    self.action_cats.append(torch.tensor(buffer_cat))
                    self.action_hints.append(action_label)
                    self.action_cards.append(torch.fmod(action_label, 5))
                except ValueError:
                    break
        assert len(self.states) == len(self.action_cats)
        assert len(self.states) == len(self.action_hints)
        assert len(self.states) == len(self.action_cards)
        print("finished loading dataset: " + dataset_file)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.action_cats[idx],
            self.action_hints[idx],
            self.action_cards[idx],
            torch.tensor(self.action_cats[idx].size()[0]),
        )

    def _convert_action(self, action):
        return


def pack_games(games):
    games.sort(key=lambda x: -x[-1])
    padded_states = torch.nn.utils.rnn.pad_sequence([x[0] for x in games])
    packed_action_cats = torch.nn.utils.rnn.pack_sequence([x[1] for x in games])
    packed_action_hints = torch.nn.utils.rnn.pack_sequence([x[2] for x in games])
    packed_action_cards = torch.nn.utils.rnn.pack_sequence([x[3] for x in games])
    return (
        padded_states,
        packed_action_cats,
        packed_action_hints,
        packed_action_cards,
        [x[-1] for x in games],
    )


class SingleNet(torch.nn.Module):
    def __init__(
        self,
        input_fc_units,
        lstm_hidden_units,
        lstm_num_layers,
        output_fc_units,
        output_units,
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
            self.output_fc.append(torch.nn.Linear(lstm_hidden_units, output_units))
        else:
            self.output_fc.append(torch.nn.Linear(output_fc_units[-1], output_units))
        self.output_fc = torch.nn.Sequential(*self.output_fc)

    def forward(self, padded, lengths):
        padded_output = self.input_fc(padded)
        packed_output = torch.nn.utils.rnn.pack_padded_sequence(padded_output, lengths)
        packed_output, _ = self.lstm(packed_output)
        padded_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        padded_output = self.output_fc(padded_output)
        packed_output = torch.nn.utils.rnn.pack_padded_sequence(padded_output, lengths)
        return packed_output


class FullNet(torch.nn.Module):
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
        self.net_first = SingleNet(
            input_fc_units,
            lstm_hidden_units,
            lstm_num_layers,
            output_fc_units,
            3,  # num output
            drop_out,
            drop_out_rate,
        )
        self.net_hint = SingleNet(
            input_fc_units,
            lstm_hidden_units,
            lstm_num_layers,
            output_fc_units,
            10,
            drop_out,
            drop_out_rate,
        )
        self.net_play = SingleNet(
            input_fc_units,
            lstm_hidden_units,
            lstm_num_layers,
            output_fc_units,
            5,
            drop_out,
            drop_out_rate,
        )
        self.net_discard = SingleNet(
            input_fc_units,
            lstm_hidden_units,
            lstm_num_layers,
            output_fc_units,
            5,
            drop_out,
            drop_out_rate,
        )
        self.nets = [self.net_first, self.net_hint, self.net_play, self.net_discard]

    def forward(self, padded, lengths):
        return tuple(net(padded, lengths) for net in self.nets)


model = FullNet([512], 512, 2, [], drop_out=True).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)


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
    correct_cat = 0
    correct_hint = 0
    correct_play = 0
    correct_discard = 0
    total_cat = 0
    total_hint = 0
    total_play = 0
    total_discard = 0
    model.eval()
    print("starting no grad")
    asdf = 0
    with torch.no_grad():
        for i, (states, action_cats, action_hints, action_cards, lengths) in enumerate(
            tqdm(valset)
        ):
            asdf += 1
            states, action_cats, action_hints, action_cards = (
                states.to(DEVICE),
                action_cats.to(DEVICE),
                action_hints.to(DEVICE),
                action_cards.to(DEVICE),
            )
            mask_hint = action_cats.data == 0
            mask_play = action_cats.data == 1
            mask_discard = action_cats.data == 2
            (pred_cat, pred_hint, pred_play, pred_discard) = model(states, lengths)
            masked_label_hint = torch.masked_select(action_hints.data, mask_hint)
            masked_label_play = torch.masked_select(action_cards.data, mask_play)
            masked_label_discard = torch.masked_select(action_cards.data, mask_discard)
            masked_pred_hint = pred_hint.data[mask_hint, :]
            masked_pred_play = pred_play.data[mask_play, :]
            masked_pred_discard = pred_discard.data[mask_discard, :]
            loss = 0
            loss += loss_fn(pred_cat.data, action_cats.data).item()
            loss += loss_fn(masked_pred_hint, masked_label_hint).item()
            loss += loss_fn(masked_pred_play, masked_label_play).item()
            loss += loss_fn(masked_pred_discard, masked_label_discard).item()
            losses.append(loss)
            correct_cat += (
                (pred_cat.data.argmax(1) == action_cats.data)
                .type(torch.float)
                .sum()
                .item()
            )
            correct_hint += (
                (masked_pred_hint.data.argmax(1) == masked_label_hint.data)
                .type(torch.float)
                .sum()
                .item()
            )
            correct_play += (
                (masked_pred_play.data.argmax(1) == masked_label_play.data)
                .type(torch.float)
                .sum()
                .item()
            )
            correct_discard += (
                (masked_pred_discard.data.argmax(1) == masked_label_discard.data)
                .type(torch.float)
                .sum()
                .item()
            )
            total_cat += action_cats.data.shape[0]
            total_hint += masked_label_hint.data.shape[0]
            total_play += masked_label_play.data.shape[0]
            total_discard += masked_label_discard.data.shape[0]
    print("sizeof: " + str(asdf))
    loss = round(sum(losses) / len(losses), 5)
    acc_cat = correct_cat / total_cat
    acc_hint = correct_hint / total_hint
    acc_play = correct_play / total_play
    acc_discard = correct_discard / total_discard
    print(
        "  val loss: {:.4f} category: {:.4f} hint: {:.4f} play: {:.4f} discard: {:.4f}".format(  # noqa E501
            loss,
            acc_cat,
            acc_hint,
            acc_play,
            acc_discard,
        )
    )
    LOGGER.add_scalar("Loss/Val", loss, log_iter)
    LOGGER.add_scalar("Category Accuracy/Val", acc_cat, log_iter)
    LOGGER.add_scalar("Hint Accuracy/Val", acc_hint, log_iter)
    LOGGER.add_scalar("Play Accuracy/Val", acc_play, log_iter)
    LOGGER.add_scalar("Discard Accuracy/Val", acc_discard, log_iter)
    model.train()


def train():
    size = len(trainset)
    for e in range(EPOCH):
        val(e * size)
        for i, (states, action_cats, action_hints, action_cards, lengths) in enumerate(
            tqdm(trainset, desc="epoch: {}".format(e))
        ):
            loss = 0
            states, action_cats, action_hints, action_cards = (
                states.to(DEVICE),
                action_cats.to(DEVICE),
                action_hints.to(DEVICE),
                action_cards.to(DEVICE),
            )
            mask_hint = action_cats.data == 0
            mask_play = action_cats.data == 1
            mask_discard = action_cats.data == 2
            (pred_cat, pred_hint, pred_play, pred_discard) = model(states, lengths)
            masked_label_hint = torch.masked_select(action_hints.data, mask_hint)
            masked_label_play = torch.masked_select(action_cards.data, mask_play)
            masked_label_discard = torch.masked_select(action_cards.data, mask_discard)
            masked_pred_hint = pred_hint.data[mask_hint, :]
            masked_pred_play = pred_play.data[mask_play, :]
            masked_pred_discard = pred_discard.data[mask_discard, :]
            loss = 0
            loss += loss_fn(pred_cat.data, action_cats.data)
            loss += loss_fn(masked_pred_hint, masked_label_hint)
            loss += loss_fn(masked_pred_play, masked_label_play)
            loss += loss_fn(masked_pred_discard, masked_label_discard)

            acc_cat = (
                (pred_cat.data.argmax(1) == action_cats.data).type(torch.float).mean()
            )
            acc_hint = (
                (masked_pred_hint.data.argmax(1) == masked_label_hint.data)
                .type(torch.float)
                .mean()
            )
            acc_play = (
                (masked_pred_play.data.argmax(1) == masked_label_play.data)
                .type(torch.float)
                .mean()
            )
            acc_discard = (
                (masked_pred_discard.data.argmax(1) == masked_label_discard.data)
                .type(torch.float)
                .mean()
            )
            LOGGER.add_scalar("Loss/Train", loss.item(), e * size + i)
            LOGGER.add_scalar("Category Accuracy/Train", acc_cat.item(), e * size + i)
            LOGGER.add_scalar("Hint Accuracy/Train", acc_hint.item(), e * size + i)
            LOGGER.add_scalar("Play Accuracy/Train", acc_play.item(), e * size + i)
            LOGGER.add_scalar(
                "Discard Accuracy/Train", acc_discard.item(), e * size + i
            )
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), MODEL_PATH)


train()
