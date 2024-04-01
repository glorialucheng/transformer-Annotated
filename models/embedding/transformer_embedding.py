"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


# 直接将token embedding和position embedding相加
# embedding后的维度为：batch * seq_len * d_model
class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        # 注意这里的tok_emb为batch * seq_len * d_model，但是pos_emb为seq_len * d_model
        # 因为采用固定位置编码，对于每个序列，相同位置的位置编码都相同
        # 即PosEmb(s1, idx) == PosEmb(s2, idx) == PosEmb(s3, idx) ...
        return self.drop_out(tok_emb + pos_emb)
