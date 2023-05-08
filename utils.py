import torch
import json
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageFile
from skimage.transform import pyramid_expand

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_wordMap(data_path, saved_path=None):
    # read dataset
    data = pd.read_csv(data_path).dropna()
    reports = data['findings'].to_list()

    min_word_freq = 5  # words occuring less than this threshold will be labelled as <unk>
    word_freq = Counter()  # Counter object to find the freq of each word

    # generate word map
    for report in reports:
        # update word frequency
        try:
            word_freq.update(report.split(' '))
        except Exception as e:
            print(report)

    # create word map (dict mapping words to ids with 0 reserved for padding)
    words = [word for word in word_freq.keys() if word_freq[word] > min_word_freq]
    word2id = {word: id for id, word in enumerate(words, 1)}
    word2id['<unk>'] = len(word2id) + 1
    word2id['<start>'] = len(word2id) + 1
    word2id['<end>'] = len(word2id) + 1
    word2id['<pad>'] = 0

    # create reverse word map (dict mapping ids to words)
    id2word = {value: word for word, value in word2id.items()}

    # store the word map
    if saved_path is not None:
        with open(saved_path+'id2word.json', 'w', encoding='utf-8') as f:
            json.dump(id2word, f)
        with open(saved_path+'word2id.json', 'w', encoding='utf-8') as f:
            json.dump(word2id, f)

    return word2id, id2word



def save_checkpoint(epoch, encoder, decoder, decoder_optimizer, bleu4, criterion, regularizer, grad_clip, saved_model_path):
    """
    Save the model with best performance during training
    """
    state = {
        'epoch': epoch,
        'bleu-4': bleu4,
        'encoder': encoder,
        'decoder': decoder,
        'decoder_optimizer': decoder_optimizer,
        'criterion': criterion,
        'regularizer': regularizer,
        'grad_clip': grad_clip
    }
    print('Saving the best model')
    torch.save(state, saved_model_path)


def plot_loss_curve(loss_path):
    with open(loss_path, 'r', encoding='utf-8') as f:
        loss = json.load(f)

    train_loss_list = loss['train_loss']
    val_loss_list = loss['val_loss']

    plt.figure(figsize=(9, 5))
    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label='val')

    plt.legend()
    plt.title('Loss per epoch')
    plt.show()


def extract_prediction_image_feature(image_path, encoder):
    # read img and processing
    original_img = cv2.resize(np.array(Image.open(image_path[0]).convert('RGB')), (256, 256))
    masked_img = cv2.resize(np.array(Image.open(image_path[1]).convert('RGB')), (256, 256))
    original_img = torch.FloatTensor(np.transpose(np.array(original_img), (2, 0, 1)) / 255.0).unsqueeze(0).to(device)
    masked_img = torch.FloatTensor(np.transpose(np.array(masked_img), (2, 0, 1)) / 255.0).unsqueeze(0).to(device)

    # encoding
    img_feature = encoder(original_img, masked_img).view(1, -1, 2048)

    return img_feature


def beam_search(trained_model, image_path, word2id, id2word, beam_size=3):
    """
    Predict report on an given image
    Ref:
    """
    # load trained encoder and decoder
    encoder = trained_model['encoder']
    decoder = trained_model['decoder']

    # get image feature
    img_feature = extract_prediction_image_feature(image_path, encoder)

    # Initializing beam search
    out = img_feature.expand(beam_size, img_feature.size(1), 2048)
    beam_sequences = torch.LongTensor([[word2id['<start>']]] * beam_size).to(device)
    beam_prev_chars = beam_sequences
    beam_scores = torch.zeros(beam_size, 1).to(device)
    beam_probs = torch.ones(beam_size, 1, encoder.img_dim, encoder.img_dim).to(device)
    done_seqs, done_prob, done_scores = [], [], []

    steps = 50
    h = decoder.h(out.mean(dim=1))
    c = decoder.c(out.mean(dim=1))

    # start searching
    for step in range(1, steps, 1):
        # decoder step
        dec_embed = decoder.embed(beam_prev_chars).squeeze(1)
        att_weighted_encoding, probs = decoder.att(out, h)
        att_weighted_encoding = torch.sigmoid(decoder.actv(h)) * att_weighted_encoding
        probs = probs.view(-1, encoder.img_dim, encoder.img_dim)
        h, c = decoder.decoding(torch.cat([dec_embed, att_weighted_encoding], dim=1), (h, c))
        scores = torch.nn.functional.log_softmax(decoder.fc(h), dim=1)

        # add score
        scores += beam_scores.expand_as(scores)

        if step == 1:
            beam_scores, beam_index = scores[0].topk(beam_size, 0, True, True)
        else:
            beam_scores, beam_index = scores.view(-1).topk(beam_size, 0, True, True)

        prev_idx = torch.div(beam_index, len(id2word), rounding_mode='trunc')
        next_idx = beam_index % len(id2word)

        beam_sequences = torch.cat([beam_sequences[prev_idx], next_idx.unsqueeze(1)], dim=1)
        beam_probs = torch.cat([beam_probs[prev_idx], probs[prev_idx].unsqueeze(1)], dim=1)
        undone_idx = [ind for ind, next_word in enumerate(next_idx) if next_word != word2id['<end>']]
        done_idx = list(set(range(len(next_idx))) - set(undone_idx))

        # add words to done sequence
        if len(done_idx) > 0:
            done_seqs += beam_sequences[done_idx].tolist()
            done_prob += beam_probs[done_idx].tolist()
            done_scores += beam_scores[done_idx]
        beam_size -= len(done_idx)

        if beam_size == 0:
            break

        beam_sequences, beam_probs = beam_sequences[undone_idx], beam_probs[undone_idx]
        h, c = h[prev_idx[undone_idx]], c[prev_idx[undone_idx]]
        out = out[prev_idx[undone_idx]]
        beam_scores = beam_scores[undone_idx].unsqueeze(1)
        beam_prev_chars = next_idx[undone_idx].unsqueeze(1)

    # choose the the sequence with max score as the predicted report
    pred_report_id = done_seqs[done_scores.index(max(done_scores))]
    pred_report = [id2word[i] for i in pred_report_id]
    probs = done_prob[done_scores.index(max(done_scores))]

    return pred_report, probs


def plot_attention_map(pred_report, probs, image_path):
    # load image
    image = Image.open(image_path).convert('RGB')
    image = image.resize([336, 336], Image.LANCZOS)

    # plot the image and attention map
    plt.figure(figsize=(12, 8))
    n_cols = 5
    n_rows = (len(pred_report) // n_cols) + (len(pred_report) % n_cols)
    alpha = 0

    for i in range(len(pred_report)):
        prob = pyramid_expand(np.array(probs[i]), upscale=24, sigma=8)

        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(image)
        if i > 0:
            alpha = 0.75
        plt.imshow(prob, alpha=alpha)
        plt.text(0, 1, pred_report[i], fontsize=15, backgroundcolor='white')

        plt.set_cmap('gray')
        plt.axis('off')

    plt.show()


