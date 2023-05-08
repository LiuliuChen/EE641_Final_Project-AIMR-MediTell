import json
import numpy as np
from PIL import ImageFile
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

from utils import save_checkpoint, get_wordMap, plot_loss_curve
from Text_Generate_Model import Encoder, Decoder
from Data_Loading import get_DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(data_loader, encoder, decoder, criterion, decoder_optimizer, regularizer,
              grad_clip, word2id, device, mode='TRAIN'):
    """
    Epoch running script
    """
    loss_list = []
    true_report = [] # true reports
    pred_report = []  # predicted reports

    # set model mode
    if mode=='TRAIN':
        # train mode (dropout and batchnorm is used)
        encoder.train()
        decoder.train()
    elif mode == 'VAL':
        encoder.eval()
        decoder.eval()
    else:
        # test data set
        encoder.eval()
        decoder.eval()


    for i, (imgs, masked_imgs, reports, report_lens, all_reports) in enumerate(tqdm(data_loader, total=len(data_loader))):
        imgs = imgs.to(device)
        masked_imgs = masked_imgs.to(device)
        reports = reports.to(device)
        report_lens = report_lens.to(device)

        # forward
        cat_imgs = encoder(imgs, masked_imgs)
        pred, reports, pred_report_lens, probs = decoder(cat_imgs, reports, report_lens)
        true_seq = reports[:, 1:]

        # remove unpadded timesteps
        pred_copy = pred.clone()
        pred = pack_padded_sequence(pred, pred_report_lens, batch_first=True, enforce_sorted=False).data
        true_seq = pack_padded_sequence(true_seq, pred_report_lens, batch_first=True, enforce_sorted=False).data

        # loss function
        loss = criterion(pred, true_seq) + regularizer * ((1. - probs.sum(dim=1)) ** 2).mean()
        loss_list.append(loss.item())

        """Updating params for training process"""
        if mode == 'TRAIN':
            decoder_optimizer.zero_grad()
            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            decoder_optimizer.step()

        if mode == 'VAL' or mode == 'TEST':
            # true report
            for j in range(all_reports.shape[0]):
                img_report = all_reports[j].tolist()
                img_report = list(
                    map(lambda x: [word for word in x if word not in {word2id['<start>'], word2id['<pad>']}], img_report))
                true_report.append(img_report)

            # prediction
            _, preds = torch.max(pred_copy, dim=2)
            preds = preds.tolist()
            preds_copy = list()
            for j, pred in enumerate(preds):
                preds_copy.append(preds[j][:pred_report_lens[j]])  # remove pads
            pred_report.extend(preds_copy)

            # calculate BLEU scores
            bleu1 = corpus_bleu(true_report, pred_report, weights=(1, 0, 0, 0))
            bleu2 = corpus_bleu(true_report, pred_report, weights=(1 / 2, 1 / 2, 0, 0))
            bleu3 = corpus_bleu(true_report, pred_report, weights=(1 / 3, 1 / 3, 1 / 3, 0))
            bleu4 = corpus_bleu(true_report, pred_report)

    if mode == 'TRAIN':
        return np.mean(loss_list)
    else:
        # VAL, TEST
        return np.mean(loss_list), [bleu1, bleu2, bleu3, bleu4]


def train(train_loader, val_loader, word2id, saved_model_path, saved_loss_path, bio_embed_mat_path=None):
    """model parameters"""
    num_epochs = 5
    enc_dim = 2048
    att_dim = 512  # attention layer dimension
    dec_dim = 512  # decoder layer dimension
    embed_dim = 512

    lr = 4e-4  # learning rate
    grad_clip = 5.
    regularizer = 1.
    decay_rate = 0.8
    decay_patience = 8

    # record the bleu4 score and choose the best model based on bleu4 score
    best_bleu4 = 0

    # Check if BioVec will be used or not
    if bio_embed_mat_path is not None:
        print('Using BioVec for text embedding...')
        bio_embed_mat = torch.load(bio_embed_mat_path)
    else:
        print('Using corpus count for text embedding...')
        bio_embed_mat = None

    # set encoder and decoder
    encoder = Encoder().to(device)
    decoder = Decoder(att_dim, embed_dim, dec_dim, len(word2id), enc_dim, bio_embed_mat).to(device)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=lr)

    # lr scheduler
    decoder_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='max', factor=decay_rate,
                                                                patience=decay_patience)

    criterion = nn.CrossEntropyLoss().to(device)

    # training epoch
    train_loss_list, val_loss_list = [], []
    for epoch in range(1, num_epochs + 1):
        loss_train = run_epoch(train_loader, encoder, decoder, criterion,
                                            decoder_optimizer, regularizer, grad_clip, word2id, device, mode='TRAIN')
        loss_val, val_bleus = run_epoch(val_loader, encoder, decoder, criterion,
                                            decoder_optimizer, regularizer, grad_clip, word2id, device, mode='VAL')

        # reduce the learning rate on plateau
        decoder_lr_scheduler.step(val_bleus[3])

        # check model with best performance
        is_best = True if val_bleus[3] > best_bleu4 else False
        best_bleu4 = val_bleus[3] if val_bleus[3] > best_bleu4 else best_bleu4

        print(
            f'epoch: {epoch}, train loss: {loss_train:.4f}, valid loss: {loss_val:.4f}, BLEU-1: {val_bleus[0]}, '
            f'BLEU-2: {val_bleus[1]}, BLEU-3: {val_bleus[2]}, best BLEU-4: {best_bleu4:.4f}')

        train_loss_list.append(loss_train)
        val_loss_list.append(loss_val)

        # save the checkpoint
        if is_best:
            save_checkpoint(epoch, encoder, decoder, decoder_optimizer, val_bleus[3], criterion,
                            regularizer, grad_clip, saved_model_path)

    # store train and val loss
    loss_record = {
        'train_loss': train_loss_list,
        'val_loss': val_loss_list
    }

    with open(saved_loss_path, 'w', encoding='utf-8') as f:
        json.dump(loss_record, f)


def test(trained_model_path, test_loader, word2id):
    # load trained model
    trained_model = torch.load(trained_model_path, map_location=device)
    # get performance on test set
    _, test_bleus = run_epoch(test_loader, trained_model['encoder'], trained_model['decoder'], trained_model['criterion'],
                              trained_model['decoder_optimizer'], trained_model['regularizer'],
                              trained_model['grad_clip'], word2id, device, mode='TEST')

    print('Final performance on test dataset')
    print(f'BLEU-1: {test_bleus[0]}, BLEU-2: {test_bleus[1]}, BLEU-3: {test_bleus[2]}, BLEU-4: {test_bleus[3]}')




def main():
    """File Path"""
    data_path = '../../autodl-tmp/medical_projects/archive/indiana_images_info.csv'  # image file - caption projection
    train_path = '../../autodl-tmp/medical_projects/archive/Final_Train_Data.csv'  # training set
    cv_path = '../../autodl-tmp/medical_projects/archive/Final_CV_Data.csv'  # validation set
    test_path = '../../autodl-tmp/medical_projects/archive/Final_Test_Data.csv'  # test set
    masked_folder = '../../autodl-tmp/medical_projects/Masked_Synthesized_Indiana_University_Dataset'  # masked image folder
    image_folder = '../../autodl-tmp/medical_projects/archive/images/images_normalized'  # original image folder
    saved_model_path = './text_generation_best.pth'  # model saved path
    saved_loss_path = './loss_record.json'
    saved_word_map_folder = './'
    bio_embed_mat_path = './bio_embed_mat.pt'

    # get word map
    word2id, id2word = get_wordMap(data_path, saved_word_map_folder)

    # load data sets
    batch_size = 32
    workers = 2
    train_loader, val_loader, test_loader = get_DataLoader(train_path, cv_path, test_path, image_folder,
                                                           masked_folder, word2id, batch_size, workers)

    # training
    train(train_loader, val_loader, word2id, saved_model_path, saved_loss_path, bio_embed_mat_path=None)

    # plot loss curve
    plot_loss_curve(saved_loss_path)

    # test model performance on the testset
    test(saved_model_path, test_loader, word2id)


if __name__ == "__main__":
    main()
