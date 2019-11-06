import json

import numpy
import torch
from torch import nn
from tqdm import tqdm

from models import ESIM
from utils.args import Args
from utils.vocabs import Vocabulary
from utils.seq import pad_sequence
from utils.load_save import *


def prepare_data():
    vocab = Vocabulary().load('./embedding/word2id.json')
    emb = load_npy('./embedding/w2v_embedding.npy')

    train_sentences1 = load_json('./data/train/sentences1.json')
    train_sentences2 = load_json('./data/train/sentences2.json')
    train_targets = load_json('./data/train/targets.json')

    dev_sentences1 = load_json('./data/dev/sentences1.json')
    dev_sentences2 = load_json('./data/dev/sentences2.json')
    dev_targets = load_json('./data/dev/targets.json')

    # map targets to int
    targets_map = {
        'contradiction':0,
        'neutral':1,
        'entailment':2,
        '-':3
    }
    train_targets = list(map(lambda x:targets_map[x], train_targets))
    dev_targets = list(map(lambda x:targets_map[x], dev_targets))
    # use 3/4 of max length as inputs
    sentence1_length = int(3/4*max(map(lambda x: len(x), train_sentences1)))
    sentence2_length = int(3/4*max(map(lambda x: len(x), train_sentences2)))

    # map word to id
    train_ids1, train_ids2, dev_ids1, dev_ids2 = list(map(
        lambda x: vocab.transfrom_sentences(x),
        [train_sentences1, train_sentences2, dev_sentences1, dev_sentences2]
    ))

    train_ids1 = torch.LongTensor(pad_sequence(train_ids1, length=sentence1_length))
    dev_ids1 = torch.LongTensor(pad_sequence(train_ids1, length=sentence1_length))
    train_ids2 = torch.LongTensor(pad_sequence(train_ids2, length=sentence2_length))
    dev_ids2 = torch.LongTensor(pad_sequence(dev_ids2, length=sentence2_length))

    # build dataloader for training
    train_loader = build_dataloader(
        train_ids1, train_ids2, torch.LongTensor(train_targets), batch_size=64)
    dev_loader = build_dataloader(
        dev_ids1, dev_ids2, torch.LongTensor(dev_targets), batch_size=64)
    return train_loader, dev_loader, emb

if __name__ == '__main__':
    train_loader, dev_loader, emb = prepare_data()
    # build model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = Args(
        emb_dim=200,
        hidden_dim=64,
        num_layers=2,
        dropout=0.5,
        output_dim=4,
        device=str(device)
    )
    model = ESIM(args).to(device)
    embedding = nn.Embedding.from_pretrained(torch.FloatTensor(emb)).to(device)
    optr = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': embedding.parameters()}
    ])
    criterion = nn.CrossEntropyLoss()

    # train a model and do validation
    EPOCHS = 30
    CHECKPOINT_PER = 1
    VALIDATE = True
    for epoch in range(EPOCHS):
        train_losses = []
        for sent1, sent2, targets in tqdm(train_loader):
            optr.zero_grad()
            sent1 = embedding(sent1.to(device))
            sent2 = embedding(sent2.to(device))
            targets = targets.to(device)

            pred = model(sent1, sent2)
            loss = criterion(pred, targets)
            train_losses.append(loss.item())
            loss.backward()
            optr.step()
        print("epoch: {} \t train_loss: {}".format(epoch+1, np.mean(train_losses)))

        if epoch % CHECKPOINT_PER == 0:
            save_model(
                './checkpoint', epoch, [model, embedding], np.mean(train_losses), device)

        if VALIDATE:
            print('validating...')
            with torch.no_grad():
                validate_losses = []
                acc = 0
                for sent1, sent2, targets in tqdm(dev_loader):
                    sent1 = embedding(sent1.to(device))
                    sent2 = embedding(sent2.to(device))
                    targets = targets.to(device)

                    pred = model(sent1, sent2)
                    
                    loss = criterion(pred, targets)
                    validate_losses.append(loss.item())
                    acc += (pred.argmax(dim=1) == targets).sum().item()
                print("epoch: {} \t validate_loss: {} \t true_num: {}".\
                    format(epoch+1, np.mean(validate_losses), acc))
    

