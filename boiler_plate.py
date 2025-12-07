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
        output = output.contiguous() 
        # Изначально: Когда ты создаешь тензор, PyTorch аккуратно укладывает числа в памяти друг за другом. Это contiguous (непрерывный) тензор.
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
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        output = self.W_1(x)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.W_2(output)
        return output

# Далее пойдут сборные блоки: EncoderLayer, DecoderLayer, Transformer

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float):
        super().__init__()
        # TODO: Инициализировать:
        # 1. Один слой MultiHeadAttention
        # 2. Один слой FeedForward
        # 3. Два слоя LayerNorm (один для attention, один для FFN)
        # 4. Слой Dropout (для применения перед сложением)
        self.attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model


    def forward(self, x, mask):
        # TODO: Реализовать проход данных
        # x -> Attention -> Dropout -> Add (к x) -> Norm
        # -> FeedForward -> Dropout -> Add (к результату) -> Norm
        x = self.attn_norm(x + self.dropout(self.attn(x, x, x, mask)))
        x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, seq_len: int, dropout: float, h: int, d_ff: int, n_layers: int):
        super().__init__()
        # TODO: Инициализация компонентов
        self.embed = InputEmbeddings(d_model, vocab_size)
        self.pe = PositionalEncoding(d_model, seq_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # TODO: Проход через Embed -> PE -> Layers -> Norm
        x = self.embed(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float):
        super().__init__()
        # TODO: Инициализировать:
        # 1. Self-Attention (с маской)
        # 2. Cross-Attention (источник K, V извне)
        # 3. FeedForward
        # 4. Три слоя LayerNorm
        # 5. Три слоя Dropout
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # TODO: Реализовать поток данных через 3 подслоя
        x = self.attn_norm(x + self.dropout(self.attn(x, x, x, tgt_mask)))
        x = self.cross_attn_norm(x + self.dropout(self.cross_attn(x, encoder_output, encoder_output, src_mask)))
        x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        return x
    
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, seq_len: int, dropout: float, h: int, d_ff: int, n_layers: int):
        super().__init__()
        # TODO: Инициализация (Embed, PE, Layers, Norm)
        self.embed = InputEmbeddings(d_model, vocab_size)
        self.pe = PositionalEncoding(d_model, seq_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # TODO: Embed -> PE -> Loop over layers -> Norm
        x = self.embed(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        x = self.norm(x)
        return x
    
class Transformer(nn.Module):
    def __init__(
        self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pe: PositionalEncoding, tgt_pe: PositionalEncoding, d_model: int, vocab_size: int
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pe = src_pe
        self.tgt_pe = tgt_pe
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.projection = nn.Linear(d_model, vocab_size)
        # TODO: Сохранить компоненты и создать Projection Layer

    def encode(self, src, src_mask):
        # TODO: Вспомогательный метод для энкодера
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # TODO: Вспомогательный метод для декодера
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # TODO: Вспомогательный метод для проекции
        return self.projection(x)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        return self.project(decoder_output)
  
vocab_size = 10000
d_model = 512
seq_len = 100
dropout = 0.1
h = 8
d_ff = 2048
n_layers = 6
model = Transformer(Encoder(vocab_size, d_model, seq_len, dropout, h, d_ff, n_layers), Decoder(vocab_size, d_model, seq_len, dropout, h, d_ff, n_layers), InputEmbeddings(d_model, vocab_size), InputEmbeddings(d_model, vocab_size), PositionalEncoding(d_model, seq_len, dropout), PositionalEncoding(d_model, seq_len, dropout), d_model, vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train() # Переводим модель в режим обучения (важно для Dropout)
    total_loss = 0

    for batch in dataloader:
        # 1. Переносим данные на GPU/CPU
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_mask = batch['src_mask'].to(device)
        
        # 2. Подготовка входа и целей (Teacher Forcing)
        # Предполагаем, что tgt: [<sos>, Word1, Word2, <eos>]
        
        decoder_input = tgt[:, :-1] # Убираем последний токен (<eos>)
        labels = tgt[:, 1:]         # Убираем первый токен (<sos>)

        # Важно: маска для декодера должна соответствовать новой длине (seq_len - 1)
        # Обычно её пересоздают здесь или обрезают batch['tgt_mask']
        tgt_mask = batch['tgt_mask'][:, :-1, :-1].to(device) 

        # 3. Сброс градиентов
        optimizer.zero_grad()

        # 4. Прямой проход (Forward pass)
        # Получаем логиты: [batch_size, seq_len-1, vocab_size]
        output = model(src, decoder_input, src_mask, tgt_mask)

        # 5. Вычисление ошибки
        # CrossEntropyLoss требует плоский ввод: (N, C) и (N)
        # output.reshape(-1, vocab_size) -> [batch * (seq_len-1), vocab_size]
        # labels.reshape(-1) -> [batch * (seq_len-1)]
        loss = criterion(output.reshape(-1, output.shape[-1]), labels.reshape(-1))

        # 6. Обратный проход и шаг оптимизатора
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
    