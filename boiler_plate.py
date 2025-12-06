import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """Превращает ID токенов в векторы."""
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        # TODO: Определить слой эмбеддингов
        pass

    def forward(self, x):
        # TODO: Реализовать логику
        pass

class PositionalEncoding(nn.Module):
    """Добавляет информацию о порядке слов."""
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        # TODO: Создать матрицу позиционных кодировок
        pass

    def forward(self, x):
        # TODO: Сложить x и позиционные кодировки
        pass

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