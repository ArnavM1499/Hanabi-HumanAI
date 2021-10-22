import torch

GAME_STATE_LENGTH = 583 + 20  # base + extended (discardable / playable)
GAME_STATE_LENGTH = 728

default_config = {
    "input_fc_units": [512],
    "lstm_hidden_units": 512,
    "lstm_num_layers": 2,
    "output_fc_units": [],
    "drop_out": True,
    "layer_norm": False,
}


class LSTMNet(torch.nn.Module):
    def __init__(
        self,
        input_fc_units,
        lstm_hidden_units,
        lstm_num_layers,
        output_fc_units,
        layer_norm=False,
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
            if layer_norm:
                self.input_fc.append(torch.nn.LayerNorm((out_dim,)))
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
            if layer_norm:
                self.output_fc.append(torch.nn.LayerNorm((out_dim,)))
            if drop_out:
                self.output_fc.append(torch.nn.Dropout(drop_out_rate))
        if output_fc_units == []:
            self.output_fc.append(torch.nn.Linear(lstm_hidden_units, 20))
        else:
            self.output_fc.append(torch.nn.Linear(output_fc_units[-1], 20))
        self.output_fc = torch.nn.Sequential(*self.output_fc)

    def forward(self, padded, lengths):
        padded_output = self.input_fc(padded)
        packed_output = torch.nn.utils.rnn.pack_padded_sequence(padded_output, lengths)
        packed_output, _ = self.lstm(packed_output)
        padded_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        padded_output = self.output_fc(padded_output)
        packed_output = torch.nn.utils.rnn.pack_padded_sequence(padded_output, lengths)
        return packed_output


class LSTMNetAugumented(torch.nn.Module):
    def __init__(
        self,
        input_fc_units,
        lstm_hidden_units,
        lstm_num_layers,
        output_fc_units,
        output_final=23,
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
            self.output_fc.append(torch.nn.Linear(lstm_hidden_units, output_final))
        else:
            self.output_fc.append(torch.nn.Linear(output_fc_units[-1], output_final))
        self.output_fc = torch.nn.Sequential(*self.output_fc)

    def forward(self, padded, lengths):
        padded_output = self.input_fc(padded)
        packed_output = torch.nn.utils.rnn.pack_padded_sequence(padded_output, lengths)
        packed_output, _ = self.lstm(packed_output)
        padded_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        padded_output = self.output_fc(padded_output)
        packed_output = torch.nn.utils.rnn.pack_padded_sequence(padded_output, lengths)
        return packed_output
