"""
@author : Sumin Lee
A Global-local Embedding Module for Fashion Landmark Detection
ICCV 2019 Workshop 'Computer Vision for Fashion, Art, and Design'
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import Network
from utils import cal_loss, Evaluator
import utils


def main(args):
    # random seed
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # load dataset
    if args.dataset == 'deepfashion':
        ds = pd.read_csv(os.path.join(args.data_dir, 'info/df_info.csv'))
        from dataset import DeepFashionDataset as DataManager
    elif args.dataset == 'fld':
        ds = pd.read_csv(os.path.join(args.data_dir, 'info/fld_info.csv'))
        from dataset import FLDDataset as DataManager
    else:
        raise ValueError

    print('dataset : %s' % (args.dataset))
    if not args.evaluate:
        train_dm = DataManager(ds[ds['evaluation_status'] == 'train'], root=args.data_dir)
        train_dl = DataLoader(train_dm, batch_size=args.batchsize, shuffle=True)

        if os.path.exists('models') is False:
            os.makedirs('models')

    test_dm = DataManager(ds[ds['evaluation_status'] == 'test'], root=args.data_dir)
    test_dl = DataLoader(test_dm, batch_size=args.batchsize, shuffle=False)

    # Load model
    print("Load the model...")
    net = torch.nn.DataParallel(Network(dataset=args.dataset, flag=args.glem))
    if torch.cuda.is_available():
        net = net.cuda()
    # net = torch.nn.DataParallel(Network(dataset=args.dataset, flag=args.glem)).cuda()
    if not args.weight_file == None:
        weights = torch.load(args.weight_file)
        if args.update_weight:
            weights = utils.load_weight(net, weights)
        net.load_state_dict(weights)

    # evaluate only
    if args.evaluate:
        print("Evaluation only")
        test(net, test_dl, 0)
        return

    # learning parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

    print("Start training")
    for epoch in range(args.epoch):
        lr_scheduler.step()
        train(net, optimizer, train_dl, epoch)
        test(net, test_dl, epoch)


def train(net, optimizer, trainloader, epoch):
    train_step = len(trainloader)
    net.train()
    # for i, sample in enumerate(trainloader):
    for i, sample in enumerate(tqdm(trainloader)):
        for key in sample:
            if torch.cuda.is_available():
                sample[key] = sample[key].cuda()
        output = net(sample)
        loss = cal_loss(sample, output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, args.epoch, i + 1, train_step, loss.item()))

    save_file = 'model_%02d.pkl'
    print('Saving Model : ' + save_file % (epoch + 1))
    torch.save(net.state_dict(), './models/' + save_file % (epoch + 1))


def test(net, test_loader, epoch):
    net.eval()
    test_step = len(test_loader)
    print('\nEvaluating...')
    with torch.no_grad():
        evaluator = Evaluator()
        # for i, sample in enumerate(test_loader):
        for i, sample in enumerate(tqdm(test_loader)):
            for key in sample:
                if torch.cuda.is_available():
                    sample[key] = sample[key].cuda()
            output = net(sample)
            evaluator.add(output, sample)
            if (i + 1) % 100 == 0:
                print('Val Step [{}/{}]'.format(i + 1, test_step))

        results = evaluator.evaluate()
        print('Epoch {}/{}'.format(epoch + 1, args.epoch))
        print(
            '|  L.Collar  |  R.Collar  |  L.Sleeve  |  R.Sleeve  |   L.Waist  |   R.Waist  |    L.Hem   |   R.Hem    |     ALL    |')
        print(
            '|   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |'
            .format(results['lm_dist'][0], results['lm_dist'][1], results['lm_dist'][2], results['lm_dist'][3],
                    results['lm_dist'][4], results['lm_dist'][5], results['lm_dist'][6], results['lm_dist'][7],
                    results['lm_dist_all']))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/Users/mackim/datasets/deepfashion/fashion_landmark_detection/',
                        help="root path to data directory")
    parser.add_argument('--dataset', type=str, default='fld', help="deepfashion or fld")

    parser.add_argument('--batchsize', type=int, default=50,
                        help='batchsize')
    parser.add_argument('--epoch', type=int, default=30,
                        help='the number of epoch')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--evaluate', type=bool, default=True,
                        help='evaluation only')
    parser.add_argument('--weight_file', type=str, default=None,
                        help='weight file')
    parser.add_argument('--glem', type=bool, default=True,
                        help='global-local embedding module')
    parser.add_argument('--update-weight', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    main(args)
