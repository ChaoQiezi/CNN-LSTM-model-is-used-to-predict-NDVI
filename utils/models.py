# @Author   : ChaoQiezi
# @Time     : 2024/1/15  16:01
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 定义模型"""

import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self,
                 input_size=6,
                 embedding_size=128,
                 hidden_size=256,
                 lstm_layers=3,
                 dropout=0.5):
        super().__init__()
        self.fc = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, lstm_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(F.relu(self.fc(x)))
        output, (hidden, cell) = self.rnn(embedded)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self,
                 input_size=6,
                 embedding_size=128,
                 hidden_size=256,
                 output_size=1,
                 lstm_layers=3,
                 dropout=0.5):
        super().__init__()
        self.embedding = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, lstm_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        # x = x.unsqueeze(0)  # add dimension, original shape=(batch_size, feature_size), ==> (1, batch_size, feature)

        embedded = self.dropout(F.relu(self.embedding(x)))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc(output[-1]).squeeze()

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, x, teacher_forcing_ratio=0.5):
        # time_step = y.shape[1]
        time_step = 12
        hidden, cell = self.encoder(x)

        outputs = torch.zeros((x.shape[0], time_step)).to(self.device)
        # decoder_input = x[:, -1, :]

        for time_ix in range(time_step):
            decoder_input = x[:, time_ix:time_ix+1, :]
            decoder_input = torch.transpose(decoder_input, 0, 1)
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, time_ix] = output

            # teacher_forcing = random.random() < teacher_forcing_ratio

            # decoder_input = y[:, time_ix] if teacher_forcing else output

        return outputs

# 创建编码解码的lstm模型
encoder = Encoder(6, 128, 256)
decoder = Decoder(6, 128, 256)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Seq2Seq(encoder, decoder, device)
summary(encoder, (12, 6))
summary(model, (12, 6))


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        :param input_size: 输入动态特征项数
        :param hidden_size: LSTM的隐藏层大小
        :param num_layers: LSTM层数
        :param output_size: 输出时间序列长度(default: 12, 12个月份)
        """
        super().__init__()
        self.causal_conv1d = nn.Conv1d(input_size, 128, 5)
        self.fc1 = nn.Linear(4, 128)
        self.rnn = nn.LSTM(128, hidden_size, num_layers, batch_first=True)
        self.fc2 = nn.Linear(384, output_size)

    def forward(self, dynamic_x, static_x):
        # 因果卷积
        conv1d_out = (self.causal_conv1d(F.pad(torch.transpose(dynamic_x, 1, 2), (2, 0))))
        # conv1d_out = self.causal_conv1d(F.pad(torch.transpose(dynamic_x, 1, 2), (2, 0)))
        # conv1d_out = self.causal_conv1d(torch.transpose(dynamic_x, 1, 2))
        # LSTM层
        lstm_out, _ = self.rnn(torch.transpose(conv1d_out, 1, 2))
        # 只使用最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # (-1, 256)
        static_out = self.fc1(static_x)  # (-1, 2) ==> (-1, 128)
        # static_out = self.fc1(static_x)  # (-1, 2) ==> (-1, 128)  2024/5/11: 静态特征由2变为4, 新增Lon、Lat
        merged_out = torch.cat([lstm_out, static_out], dim=1)  # (-1, 256 + 128)
        # 全连接层
        out = self.fc2(merged_out)  # (-1, 12)

        return out