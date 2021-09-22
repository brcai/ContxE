import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse

from Train import train

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',default='yago1830',type=str, help='dataset to train on')
parser.add_argument('--max_epoch',default=500, type=int,help='number of total epochs (min value: 500)')
parser.add_argument('--dim',default=500, type=int,help='number of dim')
parser.add_argument('--batch',default=512, type=int,help='number of batch size')
parser.add_argument('--lr',default=0.1, type=float,help='number of learning rate')
parser.add_argument('--gamma1',default=1, type=float,help='number of margin')
parser.add_argument('--gamma2',default=1, type=float,help='number of margin')
parser.add_argument('--eta',default=20, type=int,help='number of negative samples per positive')
parser.add_argument('--cuda',default=False, type=bool,help='use cuda or cpu')
parser.add_argument('--loss',default='logloss', type=str,help='loss function')
parser.add_argument('--cmin',default=0.005, type=float,help='cmin')
parser.add_argument('--gran',default=1, type=int,help='time unit for ICEWS datasets')
parser.add_argument('--thre',default=1, type=int,help='the mini threshold of time classes in yago and wikidata')
parser.add_argument('--cuda_idx',default=0, type=int,help='cuda index')


def main(args):
    print(args)
    train(data_dir=args.dataset,
          dim=args.dim,
          batch=args.batch,
          lr =args.lr,
          max_epoch=args.max_epoch,
          gamma1 = args.gamma1,
          gamma2 = args.gamma2,
          lossname = args.loss,
          negsample_num=args.eta,
          cuda_able = args.cuda,
          cmin = args.cmin,
          gran = args.gran,
          count = args.thre,
          cuda_idx = args.cuda_idx
          )              


if __name__ == '__main__':
    main(parser.parse_args())
