from model.transformer import Encoder, subsequent_mask
from torch import nn 

# signature in wordsequence:
# self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)


# NOTE: input_size == embedding dim + char hidden dim (in conf)
#       if you want a certain number of heads may need to adjust char hidden dim to make it divide

class TransformerLabeller(nn.Module):
    def __init__(self, input_size: int, num_layers: int, dropout: float, num_heads: int) -> None:
        """Initialiser

        Args:
            input_size (int): input size == embedding size
            num_layers (int): number of encoder layers
            dropout (float): amount of dropout to be applied
            num_heads (int): number of attention heads to use, note requirement of input size % heads == 0
        """
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = num_heads
        
        self.encoder = Encoder(self.num_layers, self.input_size, self.input_size * 4, self.heads, self.dropout)
        

    def forward(self, x):
        '''
        x: (batch_size, seq_len, hidden_dim)
        input size == hidden_dim
        '''
        return self.encoder(x, subsequent_mask(x.size(1)))