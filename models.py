import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderClassifier(nn.Module):

    def __init__(self, input_dim, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.drp = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.input_dim, 325)
        self.fc2 = nn.Linear(325, 396)
        self.fc3 = nn.Linear(396, 517)
        self.fc4 = nn.Linear(517, 666)
        self.fc5 = nn.Linear(666, 646)
        self.fc6 = nn.Linear(646, 476)
        self.fc7 = nn.Linear(476, 1)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.drp(x)
        x = F.gelu(self.fc2(x))
        x = self.drp(x)
        x = F.gelu(self.fc3(x))
        x = self.drp(x)
        x = F.gelu(self.fc4(x))
        x = self.drp(x)
        x = F.gelu(self.fc5(x))
        x = self.drp(x)
        x = F.gelu(self.fc6(x))
        x = self.drp(x)
        x = self.fc7(x)
        return x
    
    def clone(self):
        clone = EncoderClassifier(self.input_dim, self.dropout)
        clone.load_state_dict(self.state_dict())
        if next(self.parameters()).is_cuda:
            clone.cuda()
        return clone


class RobertaClassifier(nn.Module):

    def __init__(self, input_dim, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.drp = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.drp(x)
        x = F.gelu(self.fc2(x))
        x = self.drp(x)
        x = F.gelu(self.fc3(x))
        x = self.drp(x)
        x = F.gelu(self.fc4(x))
        x = self.drp(x)
        x = F.gelu(self.fc5(x))
        x = self.drp(x)
        x = self.fc6(x)
        return x
    
    def clone(self):
        clone = RobertaClassifier(self.input_dim, self.dropout)
        clone.load_state_dict(self.state_dict())
        if next(self.parameters()).is_cuda:
            clone.cuda()
        return clone
    

class EncoderModel(nn.Module):

    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, embeds, attention_mask):
        src_key_padding_mask = ~attention_mask.bool()
        embeds = self.transformer_encoder(embeds, src_key_padding_mask=src_key_padding_mask)
        embeds = embeds.mean(dim=1)
        return embeds

    def clone(self):
        clone = EncoderModel(self.d_model, self.nhead, self.num_layers)
        clone.load_state_dict(self.state_dict())
        if next(self.parameters()).is_cuda:
            clone.cuda()
        return clone


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BiLSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
    
    def forward(self, x, attention_mask):

        attention_mask = attention_mask.unsqueeze(-1)
        x = x * attention_mask

        lstm_out, _ = self.lstm(x)
        output = lstm_out.mean(dim=1)
        return output
    
    def clone(self):
        clone = BiLSTMModel(self.input_dim, self.hidden_dim, self.num_layers)
        clone.load_state_dict(self.state_dict())
        if next(self.parameters()).is_cuda:
            clone.cuda()
        return clone