import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, regression_output_channels=1,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(ViT, self).__init__()
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, regression_output_channels))

    def forward(self, x):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E

        x = self.conv3x3(x)

        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)  # .shape = n, n_query_channels, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        # y = torch.sigmoid(y)
        # y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps


class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=16, embedding_dim=128, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  # takes shape S,N,E

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)

        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)

    def forward(self, x):
        embeddings = self.embedding_convPxP(x).flatten(2)  # .shape = n,c,s = n, embedding_dim, s
        # embeddings = nn.functional.pad(embeddings, (1,0))  # extra special token at start ?
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)

        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        return x


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)
