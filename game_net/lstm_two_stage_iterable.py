from fire import Fire
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")

GAME_STATE_LENGTH = 728

DATA_ALL = "../exp00001-200k/00001_all.npy"
DATA_TRAIN = DATA_ALL.replace("_all", "_train")
DATA_VAL = DATA_ALL.replace("_all", "_val")
MODEL_PATH = "../model/101521-singlestage-traininfo.pth"

BATCH_SIZE = 600
EPOCH = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = SummaryWriter()


class PickleIterable(torch.utils.data.IterableDataset):
    def __init__(self, dataset_file, load_num=200):
        super(PickleIterable).__init__()
        self.load_num = load_num
        self.dataset_file = dataset_file
        self.fin = open(dataset_file, "rb")

    def __iter__(self):
        try:
            while True:
                states = []
                action_cats = []
                action_hints = []
                action_cards = []
                for i in range(self.load_num):
                    states.append(torch.from_numpy(np.load(self.fin) * 0.333).type(torch.float32))
                    action_label = torch.from_numpy(np.load(self.fin)).type(torch.long)
                    buffer_cat = []
                    for idx in action_label:
                        idx = int(idx)
                        if idx < 10:  # hint color / number
                            buffer_cat.append(0)
                        elif idx < 15:  # play
                            buffer_cat.append(1)
                        else:  # discard
                            buffer_cat.append(2)
                    action_cats.append(torch.tensor(buffer_cat))
                    action_hints.append(action_label)
                    action_cards.append(torch.fmod(action_label, 5))
                for i in range(self.load_num):
                    yield states[i], action_cats[i], action_hints[i], action_cards[i], torch.tensor(action_cats[i].size()[0])
        except ValueError:
            self.fin.close()
            self.fin = open(self.dataset_file, "rb")


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
    PickleIterable(DATA_TRAIN),
    batch_size=BATCH_SIZE,
    collate_fn=pack_games,
)
valset = torch.utils.data.DataLoader(
    PickleIterable(DATA_VAL),
    batch_size=BATCH_SIZE,
    collate_fn=pack_games,
)


def val(log_iter=0):
    losses = []
    cat_loss = 0
    hint_loss = 0
    play_loss = 0
    disc_loss = 0
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
            cat_loss = loss_fn(pred_cat.data, action_cats.data).item()
            hint_loss = loss_fn(masked_pred_hint, masked_label_hint).item()
            play_loss = loss_fn(masked_pred_play, masked_label_play).item()
            disc_loss = loss_fn(masked_pred_discard, masked_label_discard).item()
            loss = cat_loss + hint_loss + play_loss + disc_loss
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
        "  cat loss: {:.4f} hint loss: {:.4f} play loss: {:.4f} disc loss: {:.4f} category: {:.4f} hint: {:.4f} play: {:.4f} discard: {:.4f}".format(  # noqa E501
            cat_loss,
            hint_loss,
            play_loss,
            disc_loss,
            acc_cat,
            acc_hint,
            acc_play,
            acc_discard,
        )
    )
    LOGGER.add_scalar("Loss/Val", loss, log_iter)
    LOGGER.add_scalar("Cat Loss/Val", cat_loss, log_iter)
    LOGGER.add_scalar("Hint Loss/Val", hint_loss, log_iter)
    LOGGER.add_scalar("Play Loss/Val", play_loss, log_iter)
    LOGGER.add_scalar("Disc Loss/Val", disc_loss, log_iter)
    LOGGER.add_scalar("Category Accuracy/Val", acc_cat, log_iter)
    LOGGER.add_scalar("Hint Accuracy/Val", acc_hint, log_iter)
    LOGGER.add_scalar("Play Accuracy/Val", acc_play, log_iter)
    LOGGER.add_scalar("Discard Accuracy/Val", acc_discard, log_iter)
    model.train()


def train():
    size = 465
    for e in range(EPOCH):
        val(e * size)
        for i, (states, action_cats, action_hints, action_cards, lengths) in enumerate(
            tqdm(trainset, desc="epoch: {}".format(e))
        ):
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
            cat_loss = loss_fn(pred_cat.data, action_cats.data)
            hint_loss = loss_fn(masked_pred_hint, masked_label_hint)
            play_loss = loss_fn(masked_pred_play, masked_label_play)
            disc_loss = loss_fn(masked_pred_discard, masked_label_discard)
            loss += cat_loss.item() + hint_loss.item() + play_loss.item() + disc_loss.item()

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
            log_iter = e * size + i
            LOGGER.add_scalar("Loss/Train", loss, e * size + i)
            LOGGER.add_scalar("Cat Loss/Val", cat_loss.item(), log_iter)
            LOGGER.add_scalar("Hint Loss/Val", hint_loss.item(), log_iter)
            LOGGER.add_scalar("Play Loss/Val", play_loss.item(), log_iter)
            LOGGER.add_scalar("Disc Loss/Val", disc_loss.item(), log_iter)
            LOGGER.add_scalar("Category Accuracy/Train", acc_cat.item(), e * size + i)
            LOGGER.add_scalar("Hint Accuracy/Train", acc_hint.item(), e * size + i)
            LOGGER.add_scalar("Play Accuracy/Train", acc_play.item(), e * size + i)
            LOGGER.add_scalar(
                "Discard Accuracy/Train", acc_discard.item(), e * size + i
            )
            cat_loss.backward()
            hint_loss.backward()
            play_loss.backward()
            disc_loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), MODEL_PATH)


def eval(log_iter=0):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    cats_encode = [0] * 9
    categories = [[0, 0, 0]] * 3
    corrects = [0] * 20
    totals = [0] * 20
    with torch.no_grad():
        for i, (states, action_cats, action_hints, action_cards, lengths) in enumerate(tqdm(valset)):
            for label in range(20):
                states, action_cats, action_hints, action_cards = (
                    states.to(DEVICE),
                    action_cats.to(DEVICE),
                    action_hints.to(DEVICE),
                    action_cards.to(DEVICE),
                )
                # pred = model(states, lengths)
                mask_hint = action_cats.data == 0
                mask_play = action_cats.data == 1
                mask_discard = action_cats.data == 2
                (pred_cat, pred_hint, pred_play, pred_discard) = model(states, lengths)
                cats = (3 * action_cats.data.type(torch.long) + pred_cat.data.argmax(1).type(torch.long)).type(torch.int)
                for j in range(9):
                    cats_encode[j] = cats.count_nonzero(cats)
                    cats = cats - 1

    total_sum_cats = sum(cats_encode) / 8
    for i in range(9):
        cats_encode[i] = total_sum_cats - cats_encode[i]
    print(cats_encode)
    print("Accuracy:")
    print("Hint Color: {:.4f}".format(sum(corrects[:5]) / sum(totals[:5])))
    print("Hint Number: {:.4f}".format(sum(corrects[5:10]) / sum(totals[5:10])))
    print("Hint Overall: {:.4f}".format(sum(corrects[:10]) / sum(totals[:10])))
    print("Play: {:.4f}".format(sum(corrects[10:15]) / sum(totals[10:15])))
    print("Discard: {:.4f}".format(sum(corrects[15:20]) / sum(totals[15:20])))


if __name__ == "__main__":
    Fire()