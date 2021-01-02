from utils import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
from torch.autograd import Variable
import timeit


def verify(model, fmatrix, in_frames, candidate_id):
    # fmatrix of size [T, feat_dim]
    # (1) build frames with context first
    frames = build_frames(fmatrix, in_frames)
    # (2) build input tensor to network
    x = Variable(torch.FloatTensor(np.array(frames)))
    # (3) Infer the prediction through network
    y_ = model(x)
    # (4) Sum up the logprobs to obtain final decission
    pred = y_.sum(dim=0)
    class_probs = F.softmax(pred, dim=0)
    return class_probs[candidate_id]

def main(opts):
    with open(opts.train_cfg, 'r') as cfg_f:
        cfg = json.load(cfg_f)
    with open(cfg['spk2idx'], 'r') as spk2idx_f:
        spk2idx = json.load(spk2idx_f)
        idx2spk = dict((v, k) for k, v in spk2idx.items())
    # Feed Forward Neural Network
    model = nn.Sequential(nn.Linear(cfg['input_dim'] * cfg['in_frames'],
                                    cfg['hsize']),
                          nn.ReLU(),
                          nn.Linear(cfg['hsize'], cfg['hsize']),
                          nn.ReLU(),
                          nn.Linear(cfg['hsize'], cfg['hsize']),
                          nn.ReLU(),
                          nn.Linear(cfg['hsize'], cfg['num_spks']),
                          nn.LogSoftmax(dim=1))
    print('Created model:')
    print(model)
    print('-')
    # load weights
    model.load_state_dict(torch.load(opts.weights_ckpt))
    print('Loaded weights')
    out_log = open(opts.log_file, 'w')
    verif_cand_list = []
    with open(opts.ve_cand_list_file, 'r') as verif_cand_f:
        for l in verif_cand_f.readlines():
            candidate = l[:6]
            for cand, id in spk2idx.items():
                if (cand==candidate):
                    verif_cand_list.append(id)
        print(verif_cand_list)

    with open(opts.ve_list_file, 'r') as verif_f:
        verif_list = [l.rstrip() for l in verif_f]
        print(len(verif_cand_list))
        print(len(verif_list))
        timings = []
        beg_t = timeit.default_timer()
        for verif_i, verif_file in enumerate(verif_list, start=1):
            print(verif_i-1)
            verif_path = os.path.join(opts.db_path, verif_file + '.' + opts.ext)
            fmatrix = read_fmatrix(verif_path)
            score = verify(model, fmatrix, cfg['in_frames'], verif_cand_list[verif_i-1]).item()
            out_log.write('{}\t{}\t{}\n'.format(verif_file, idx2spk[verif_cand_list[verif_i-1]], str(score)))
            print('{}\t{}\t'.format(verif_file, idx2spk[verif_cand_list[verif_i-1]], str(score)))
            if opts.verbose:
                end_t = timeit.default_timer()
                timings.append(end_t - beg_t)
                beg_t = timeit.default_timer()
            
                print('Processing verif file {} ({}/{}) with shape: {}'
                      ', mtime: {:.3f} s'.format(verif_file, verif_i, len(verif_list),
                                                 fmatrix.shape,
                                                 np.mean(timings)))
    out_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply trained MLP to classify')

    
    parser.add_argument('--db_path', type=str, default='mcp',
                        help='path to feature files (default: ./mcp)')
    parser.add_argument('--ve_list_file', type=str, default='spk_rec.test',
                        help='list with names of files to verify (default. spk_rec.test)')
    parser.add_argument('--ve_cand_list_file', type=str, default='spk_rec.test',
                        help='list with names of files to verify (default. spk_rec.test.candidates)')
    parser.add_argument('--weights_ckpt', type=str, default=None, help='model: ckpt file with weigths')
    parser.add_argument('--log_file', type=str, default='spk_verif.log',
                        help='result file (default: spk_classification.log')
    parser.add_argument('--train_cfg', type=str, default='ckpt/train.opts',
                        help="arguments used for training (default: ckpt/train.opts)")
    parser.add_argument('--ext', type=str, default='mcp',
                        help='Extension of feature files (default mcp)')
    parser.add_argument('--verbose', action='store_true', 
                        help='Print information about required time, input shape, etc.')


    opts = parser.parse_args()
    if opts.weights_ckpt is None:
        raise ValueError('Weights ckpt not specified!')
    main(opts)

