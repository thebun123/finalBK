from torch.utils.data import Dataset
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import random
import argparse
import utils
import copy
from net import Net
from data import VoiceDataset

parser = argparse.ArgumentParser(description='Recognition Voice Number 0-9')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--net_type', default='audio_net', type=str, help='model')
parser.add_argument('--feature_type', default='log_spectrogram', type=str, help='feature type')
# parser.add_argument('--depth', default=18, type=int, choices=[18, 34, 50, 101, 152], help='depth model')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--seed', default=random.randint(1, 1000), type=int, help='random seed')
parser.add_argument('--epochs', default=50, type=int, help='num epochs')

TRAIN_DATA_PATH = './train'
VAL_DATA_PATH = './val'
TEST_DATA_PATH = './test'
MODEL_PATH = './model'
FIG_ACC_PATH = './figure/acc'
FIG_LOSS_TRAIN_PATH = './figure/loss_train'
FIG_LOSS_VAL_PATH = './figure/loss_val'
CONF_MATRIX_PATH = './conf_matrix'


def init_path():
    # create model path
    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    # create fig acc
    if not os.path.isdir(FIG_ACC_PATH):
        os.mkdir(FIG_ACC_PATH)
    # create loss training
    if not os.path.isdir(FIG_LOSS_TRAIN_PATH):
        os.mkdir(FIG_LOSS_TRAIN_PATH)
    # create loss validation
    if not os.path.isdir(FIG_LOSS_VAL_PATH):
        os.mkdir(FIG_LOSS_VAL_PATH)
    # create conf matrix
    if not os.path.isdir(CONF_MATRIX_PATH):
        os.mkdir(CONF_MATRIX_PATH)


def lr_schedule(args, optimizer, epoch, lr_decay_epoch=4):
    # update learning rate every 4 epochs
    lr = args.lr*(0.6**(min(epoch, args.epochs)//lr_decay_epoch))
    # weight decay unchanged
    weight_decay = args.weight_decay
    for param in optimizer.param_groups:
        param['lr'] = lr
        param['weight_decay'] = weight_decay
    return optimizer, lr


def train(args, train_loader, val_loader, test_loader):

    # name model = net_type   + feature_type + seed
    name = args.net_type + '_' + args.feature_type + '_' + str(args.seed)
    MODEL = os.path.join(MODEL_PATH, name+'.pth')
    # check device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # audio_net = Net(args.depth)
    audio_net = Net()
    if os.path.isfile(MODEL):
        state = torch.load(MODEL)
        audio_net.load_state_dict(state['model_state'])
    else:
        state = {'epoch': 0,
                 'acc_each_epoch': [],
                 'val_loss': [],
                 'training_loss': [],
                 'best_acc': 0.0,
                 'model_state': None
                 }

    best_model = copy.deepcopy(audio_net)
    # use device to train
    audio_net.to(device)

    # loss function
    loss_f = nn.CrossEntropyLoss()

    # adam optimizer
    optimizer = optim.Adam(audio_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_accuracy = state['best_acc']
    for epoch in range(args.epochs):
        state['epoch'] += 1
        optimizer, lr = lr_schedule(args, optimizer, epoch)
        training_loss, val_loss = 0.0, 0.0
        audio_net.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = audio_net(inputs)
            loss = loss_f(outputs, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        state['training_loss'].append(training_loss)

        audio_net.eval()
        num_correct, num_examples = 0, 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = audio_net(inputs)
            loss = loss_f(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            num_correct += (predicted == targets).sum().item()
            num_examples += targets.size(0)
        val_loss /= len(val_loader.dataset)
        state['val_loss'].append(val_loss)
        accuracy = num_correct / num_examples

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            state['best_acc'] = best_accuracy
            state['model_state'] = audio_net.state_dict()
            print('saved best model with accuracy {}'.format(best_accuracy))
            best_model = copy.deepcopy(audio_net)
        state['acc_each_epoch'].append(accuracy)
        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(state['epoch'],
                                                                                                    training_loss,
                                                                                                    val_loss,
                                                                                                    accuracy))
    torch.save(state, MODEL)

    # test model
    conf_matrix = np.zeros((10, 10), dtype=np.int)
    best_model.to(device)
    best_model.eval()
    num_correct, num_examples = 0, 0
    for data in test_loader:
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs, 1)
        for i in range(target.size(0)):
            conf_matrix[predicted[i], target[i]] += 1
        num_correct += (predicted == target).sum().item()
        num_examples += inputs.size(0)
    score = 100 * num_correct / num_examples
    print('Your score : {:.2f}'.format(score))

    # save figure
    utils.save_fig_schedule_accuracy(fig_path=FIG_ACC_PATH, data=state['acc_each_epoch'], type='Accuracy', name=name, score=score)
    utils.save_fig_schedule_loss(fig_path=FIG_LOSS_TRAIN_PATH, data=state['training_loss'], type='Training', name=name)
    utils.save_fig_schedule_loss(fig_path=FIG_LOSS_VAL_PATH, data=state['val_loss'], type='Validation', name=name)
    utils.save_confusion_matrix(conf_matrix, name=name, dst_path=CONF_MATRIX_PATH)
    print('Saved Confusion Matrix')


if __name__ == '__main__':

    # make some folder to save model, fig, conf_matrix
    init_path()
    args = parser.parse_args()

    # choose random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # build dataset
    print('Building dataset')
    train_10 = VoiceDataset(TRAIN_DATA_PATH)
    val_10 = VoiceDataset(VAL_DATA_PATH)
    test_10 = VoiceDataset(TEST_DATA_PATH)

    # data augment
    # train_10_augment = VoiceDataset(TRAIN_DATA_PATH, frequency_mask=True, time_mask=True, max_width=5, num_mask=2)
    # make new dataset: train + train_augment
    # train_concat = torch.utils.data.ConcatDataset((train_10_augment, train_10))

    # build dataloader
    print('Building DataLoader')
    train_loader = torch.utils.data.DataLoader(train_10, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_10, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_10, batch_size=args.batch_size, shuffle=True)

    print('Train model with random seed : {}'.format(args.seed))
    train(args, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)


