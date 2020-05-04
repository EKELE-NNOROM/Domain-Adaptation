import os
import argparse

import numpy as np
import tqdm
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

import data_loader
from model import DANN
from utils import DEVICE

def get_parser():
    from argparse import ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=.5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--nepoch', type=int, default=100)
    parser.add_argument('--source', type=str, default='mnist')
    parser.add_argument('--target', type=str, default='mnist_m')
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--result_path', type=str, default='result/result.csv')
    return parser

def test(model, dataset_name, epoch):
    alpha = 0
    dataloader = data_loader.load_test_data(dataset_name)
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(dataloader):
            t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE)
            class_output, _ = model(input=t_img, alpha=alpha)
            prob, pred = torch.max(class_output.data, 1)
            n_correct += (pred == t_label.long()).sum().item()

    acc = float(n_correct) / len(dataloader.dataset) * 100
    return acc


def train(args, model, optimizer, src_dataloader, tar_dataloader):
    best_acc = -float('inf')
    n_dataloader = min(len(src_dataloader), len(tar_dataloader))
    class_loss = torch.nn.CrossEntropyLoss()
    for epoch in range(args.nepoch):
        model.train()
        i = 1
        for (src_data, tar_data) in tqdm(zip(enumerate(src_dataloader), enumerate(tar_dataloader)), total=n_dataloader, leave=False):
            _, (x_src, y_src) = src_data
            _, (x_tar, _) = tar_data
            x_src, y_src, x_tar = x_src.to(DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE)
            # p - normalize the training progress to range from 0 to 1
            p = float(i + epoch * n_dataloader) / args.nepoch /n_dataloader
            # alpha - hyperparameter for scaling the reversed gradient changes during training to gradually confuse the discriminator
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            class_output, err_s_domain = model(input=x_src, alpha=alpha)
            err_s_label = class_loss(class_output, y_src)
            _, err_t_domain = model(
                                    input=x_tar, alpha=alpha, source=False)
            err_domain = err_t_domain + err_s_domain
            err = err_s_label + args.gamma * err_domain

            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            i + 1

        progress = 'Epoch: [{}/{}], classify_loss: {:.4f}, domain_loss_s: {:.4f}, domain_loss_t: {:.4f}, domain_loss: {:.4f},total_loss: {:.4f}'.format(
            epoch, args.nepoch, err_s_label.item(), err_s_domain.item(), err_t_domain.item(), err_domain.item(), err.item())
        print(progress)
        fhand = open(args.result_path, 'a')
        fhand.write(progress + '\n')

        # test
        src_accuracy = test(model, args.source, epoch)
        tar_accuracy = test(model, args.target, epoch)
        test_info = 'Source accuracy: {:.4f}, target accuracy: {:.4f}'.format(src_accuracy, tar_accuracy)
        fhand.write(test_info + '\n')
        print(test_info)
        fhand.close()

        if best_acc < tar_accuracy:
            best_acc = tar_accuracy
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            torch.save(model, '{}/mnist_mnistm.pth'.format(args.model_path))
    print('Test acc: {:.4f}'.format(best_acc))


def main(args):
    src_loader, tar_loader = data_loader.load_data()
    model = DANN().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    train(args, model, optimizer, src_loader, tar_loader)



if __name__ == '__main__':
    torch.random.manual_seed(10)
    parser = get_parser()
    args = parser.parse_args()
    main(args)
