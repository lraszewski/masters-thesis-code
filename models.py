import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        return x
    
    def clone(self):
        clone = ClassificationHead(self.input_dim, self.hidden_dim)
        clone.load_state_dict(self.state_dict())
        if next(self.parameters()).is_cuda:
            clone.cuda()
        return clone


class RobertaPooler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, embeds, attention_mask):
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(embeds.size()).float()
        embeds = torch.sum(embeds * attention_mask_expanded, 1) / torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        embeds = F.normalize(embeds, p=2, dim=1)
        return embeds
    
    def clone(self):
        return RobertaPooler()
    

class EncoderModel(nn.Module):

    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, embeds, attention_mask):
        attention_mask = (1.0 - attention_mask) * -1e9
        embeds = self.transformer_encoder(embeds, src_key_padding_mask=attention_mask)
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


class EncoderClassifierModel(EncoderModel):

    def __init__(self, d_model, nhead, num_layers):
        super().__init__(d_model, nhead, num_layers)
        self.classifier = nn.Linear(d_model, 1)
    
    def forward(self, embeds, attention_mask):
        embeddings = super().forward(embeds, attention_mask)
        logits = self.classifier(embeddings)
        return logits
    
    def clone(self):
        clone = EncoderClassifierModel(self.d_model, self.nhead, self.num_layers)
        clone.load_state_dict(self.state_dict())
        if next(self.parameters()).is_cuda:
            clone.cuda()
        return clone