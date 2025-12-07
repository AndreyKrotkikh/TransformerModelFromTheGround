import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """Превращает ID токенов в векторы."""
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        # TODO: Определить слой эмбеддингов
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # TODO: Реализовать логику
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """Добавляет информацию о порядке слов."""
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        # TODO: Создать матрицу позиционных кодировок
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        

    def forward(self, x):
        # TODO: Сложить x и позиционные кодировки
        x = self.dropout(x + self.pe[:x.size(1)])
        return x

class MultiHeadAttention(nn.Module):
    """Механизм многоголового внимания."""
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        # TODO: Определить матрицы W_q, W_k, W_v, W_o
        pass

    def forward(self, q, k, v, mask):
        # TODO: Реализовать attention score и объединение голов
        pass

class FeedForward(nn.Module):
    """Полносвязная сеть (FFN) внутри блока."""
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # TODO: Два линейных слоя и активация
        pass

    def forward(self, x):
        pass

# Далее пойдут сборные блоки: EncoderLayer, DecoderLayer, Transformer