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
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model // h
        # TODO: Определить матрицы W_q, W_k, W_v, W_o
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        # q: [batch_size, seq_len, d_model]
        # k: [batch_size, seq_len, d_model]
        # v: [batch_size, seq_len, d_model]
        # mask: [batch_size, seq_len, seq_len]
        # TODO: Реализовать attention score и объединение голов
        batch_size = q.size(0)
        q = self.W_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) # [batch_size, seq_len, d_model] -> [batch_size, h, seq_len, d_k]
        k = self.W_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) # [batch_size, seq_len, d_model] -> [batch_size, h, seq_len, d_k] 
        v = self.W_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) # [batch_size, seq_len, d_model] -> [batch_size, h, seq_len, d_k]
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # [batch_size, h, seq_len, d_k] * [batch_size, h, d_k, seq_len] -> [batch_size, h, seq_len, seq_len]
        attn_score = attn_score.masked_fill(mask == 0, float('-inf')) # [batch_size, h, seq_len, seq_len]
        attn_probs = torch.softmax(attn_score, dim=-1) # [batch_size, h, seq_len, seq_len]
        attn_probs = self.dropout(attn_probs) # [batch_size, h, seq_len, seq_len]
        output = torch.matmul(attn_probs, v) # [batch_size, h, seq_len, seq_len] * [batch_size, h, seq_len, d_k] -> [batch_size, h, seq_len, d_k]
        output = output.transpose(1, 2) # [batch_size, h, seq_len, d_k] -> [batch_size, seq_len, h, d_k]
        output = output.contiguous() # Изначально: Когда ты создаешь тензор, PyTorch аккуратно укладывает числа в памяти друг за другом. Это contiguous (непрерывный) тензор.
        # Transpose: Когда ты делаешь .transpose(1, 2), PyTorch — хитрец. Он не перемещает числа в памяти (это было бы долго). Он просто меняет "правила чтения" (метаданные/strides).
        # Логически тензор повернут.
        # Физически числа лежат в памяти в старом порядке.
        # Проблема с .view(): Метод .view() очень прост. Он берет кусок памяти и говорит: "Теперь эти 512 чисел — это один вектор". Он ожидает, что числа лежат рядом. Но после транспонирования они разбросаны!
        # Решение: Метод .contiguous() создает новую копию тензора, где числа физически перекладываются в новом, правильном порядке. Теперь .view() может спокойно отрезать нужные куски.
        output = output.view(batch_size, -1, self.d_model) # [batch_size, seq_len, h, d_k] -> [batch_size, seq_len, d_model]
        output = self.W_o(output) # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        return output

class FeedForward(nn.Module):
    """Полносвязная сеть (FFN) внутри блока."""
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # TODO: Два линейных слоя и активация
        pass

    def forward(self, x):
        pass

# Далее пойдут сборные блоки: EncoderLayer, DecoderLayer, Transformer