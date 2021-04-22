import torch.nn as nn

######################################################
######################################################
##################      FNN      #####################
######################################################
######################################################


class FNNEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate, nonlinear=True):

        super(FNNEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.nonlinear = nonlinear

        print('DAN: input {}, hidden {}, output {}'.format(self.input_size, self.hidden_size, self.output_size))

        # first hidden layers
        if self.nonlinear:
            self.hidden = nn.ModuleList([nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
                                         nn.ReLU(),
                                         nn.Dropout(self.dropout_rate)])
        else:
            self.hidden = nn.ModuleList([nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
                                         nn.Dropout(self.dropout_rate)])

        # optional deep layers
        for k in range(1, self.num_layers):
            if self.nonlinear:
                self.hidden.extend([nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout_rate)])
            else:
                self.hidden.extend([nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                                    nn.Dropout(self.dropout_rate)])

        # output linear function (readout)
        self.final = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, x):

        y = x
        for i in range(len(self.hidden)):
            y = self.hidden[i](y)

        out = self.final(y)

        return out
