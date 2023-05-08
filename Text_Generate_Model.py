"""
Throughout our implementations, we have researched some applications like beam search
and studied from the paper: Show, Attend and Tell
https://arxiv.org/abs/1502.03044
"""
import torch
import torch.nn as nn
import torchvision
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, img_dim=14):
        super(Encoder, self).__init__()
        self.img_dim = img_dim

        # pretrained ImageNet ResNet-101
        resnet = torchvision.models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d((img_dim, img_dim))

    def forward(self, img, masked_img):
        """get original image feature"""
        img_out = self.resnet(img)
        img_out = self.pooling(img_out)
        img_out = img_out.permute(0, 2, 3, 1)

        """get masked image feature"""
        masked_img_out = self.resnet(masked_img)
        masked_img_out = self.pooling(masked_img_out)
        masked_img_out = masked_img_out.permute(0, 2, 3, 1)

        # concat
        out = torch.cat((img_out.reshape(img_out.size(0), -1), masked_img_out.reshape(masked_img_out.size(0), -1)),
                        dim=1)

        return out


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, att_dim):
        super(Attention, self).__init__()
        self.enc_att = nn.Linear(enc_dim, att_dim)
        self.dec_att = nn.Linear(dec_dim, att_dim)
        self.att = nn.Linear(att_dim, 1)

    def forward(self, input, dec_h):
        enc_att = self.enc_att(input)
        dec_att = self.dec_att(dec_h)
        att = self.att(torch.nn.functional.relu(enc_att + dec_att.unsqueeze(1)))

        prob = torch.nn.functional.softmax(att, dim=1)
        prob_weighted_encoding = (input * prob).sum(dim=1)
        prob = prob.squeeze(2)
        return prob_weighted_encoding, prob


class Decoder(nn.Module):
    def __init__(self, att_dim, embed_dim, dec_dim, vocab_size, enc_dim=2048, bio_embed_mat=None, drop_out_rate=0.4):
        """
        Decoder with attention to generate text
        """
        super(Decoder, self).__init__()
        self.enc_dim = enc_dim
        self.vocab_size = vocab_size

        self.att = Attention(enc_dim, dec_dim, att_dim)
        if bio_embed_mat is not None:
            # Use BioVec
            self.embed = nn.Embedding.from_pretrained(bio_embed_mat, freeze=False)
        else:
            # original vec from corpus
            self.embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.embed.weight, -1.0, 1.0)  # init weight
        self.drop_out = nn.Dropout(p=drop_out_rate)

        self.decoding = nn.LSTMCell(embed_dim + enc_dim, dec_dim, bias=True)
        self.h = nn.Linear(enc_dim, dec_dim)
        self.c = nn.Linear(enc_dim, dec_dim)
        self.actv = nn.Linear(dec_dim, enc_dim)

        self.fc = nn.Linear(dec_dim, vocab_size)
        nn.init.uniform_(self.fc.weight, -1.0, 1.0)  # init weight
        self.fc.bias.data.fill_(0)

    def forward(self, input, captions, caption_lens):
        input = input.view(input.size(0), -1, self.enc_dim) # flatten image
        embeddings = self.embed(captions) # get embeddings of caption
        pred_cap_lens = (caption_lens.squeeze(1) - 1).tolist()  # caption lens to predict

        # hidden layer and cell of LSTM
        h = self.h(input.mean(dim=1))
        c = self.c(input.mean(dim=1))

        pred_cap_list = torch.zeros(input.size(0), max(pred_cap_lens), self.vocab_size).to(device)
        probs = torch.zeros(input.size(0), max(pred_cap_lens), input.size(1)).to(device)

        for t in range(max(pred_cap_lens)):
            t_len = sum([l > t for l in pred_cap_lens])
            att_vec, prob = self.att(input[:t_len], h[:t_len])
            gate = torch.sigmoid(self.actv(h[:t_len]))
            att_vec = gate * att_vec

            h, c = self.decoding(
                torch.cat([embeddings[:t_len, t, :], att_vec], dim=1),
                (h[:t_len], c[:t_len])
            )

            pred = self.fc(self.drop_out(h))

            pred_cap_list[:t_len, t, :] = pred
            probs[:t_len, t, :] = prob

        return pred_cap_list, captions, pred_cap_lens, probs
