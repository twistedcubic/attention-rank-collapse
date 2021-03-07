import argparse
import copy
import os
import pathlib
import pdb
import sys

import numpy as np
import scipy
import torch
import torch.nn as nn
from tqdm import tqdm

import arguments
import plot
from configuration_san import SANConfig
from functions import create_path
from modeling_san import SANForSequenceClassification


"""
Letter Sorting.
One metric is how many pairs out of order. Rather than strict matching.
allow repeat letters. All seq same len
"""
cuda = torch.cuda.is_available()


def evaluate_metric3(pred_l, labels):
    """
    measure the number of mis-predicted numbers.
    """
    cor_perc = torch.eq(pred_l, labels).sum() / pred_l.size(0) / pred_l.size(1)
    return cor_perc


def evaluate_metric2(pred_l, gt_l):
    """
    Generate metric, number of mismatched pairs, normalized
    gt_l: ground truth array of ints. not their ordering!
    pred_l: is prediction of their ordering.
    """
    total_mis = 0
    vals, pred_order = torch.topk(pred_l, k=pred_l.size(-1), dim=-1, largest=False)  # all_data.clone()
    for i, pred_seq in enumerate(pred_l):
        mis_pair = 0
        for j, cur_num in enumerate(pred_seq[:-1]):
            #
            if args.no_repeat:
                mis_pair += (pred_seq[j + 1 :] <= cur_num).sum()
            else:
                mis_pair += (pred_seq[j:] < cur_num).sum()
        cor_pair = scipy.special.comb(len(pred_seq), 2) - mis_pair
        total_mis += cor_pair
    return total_mis.cpu().item() / len(pred_l)


def evaluate_metric(pred_l, gt_l):
    """
    Generate metric, number of mismatched pairs, normalized
    gt_l: ground truth array of ints. not their ordering!
    pred_l: is prediction of their ordering.
    """
    total_mis = 0
    vals, pred_order = torch.topk(pred_l, k=pred_l.size(-1), dim=-1, largest=False)  # all_data.clone()

    for i, pred_idx in enumerate(pred_l):
        mis_pair = 0
        pred_nums = gt_l[i][pred_order[i]]
        for j, cur_idx in enumerate(pred_idx):  # .tolist():
            gt_num = pred_nums[j]  # gt_l[i][j]
            mis_pair += (pred_nums[:j] > gt_num).sum()

            mis_pair += (pred_nums[j:] < gt_num).sum()
        total_mis += mis_pair
    return total_mis.cpu().item() / len(pred_l) / len(pred_idx)


def run_eval(pred_l, gt_ordering):
    """
    gt_l: ground truth array of ints. not their ordering! pred_l is prediction of their ordering.
    """
    cnt = 0
    match_cnt = 0
    for i, pred in enumerate(pred_l):
        pred = pred.argmax(-1)
        total_match = (pred == gt_ordering[i]).sum()
        match_cnt += total_match
        if total_match == len(pred):
            cnt += 1
    return cnt / len(pred_l), match_cnt.item() / len(pred_l) / len(pred)


def gen_data(n_data, args):
    """
    Generate data tuples. Letter sequence and ground truth ordering.
    """
    # n_data = args.n_data
    n_labels = args.num_labels
    all_data = torch.stack([torch.randperm(n_labels) for _ in range(n_data)])
    # vals, all_labels = torch.topk(all_data, k=n_labels, dim=-1, largest=False) #all_data.clone()
    all_labels = all_data.clone()
    return all_data, all_labels


def gen_data_repeat(n_data, args):
    """
    Generate data tuples. Letter sequence and ground truth ordering.
    Allow repeats
    """
    # n_data = args.n_data
    n_labels = args.num_labels
    all_data = torch.randint(n_labels, (n_data, args.seq_len))

    labels, idx = torch.sort(all_data, dim=-1)
    return all_data, labels


class SortingDataset(torch.utils.data.Dataset):
    def __init__(self, n_data, args):
        super(SortingDataset, self).__init__()
        if args.no_repeat:
            self.all_data, self.all_labels = gen_data(n_data, args)
        else:
            self.all_data, self.all_labels = gen_data_repeat(n_data, args)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return (self.all_data[idx], self.all_labels[idx])

    def branch_subset(self, n_data):
        new_set = copy.deepcopy(self)
        new_set.all_data = self.all_data[:n_data]
        new_set.all_labels = self.all_labels[:n_data]
        return new_set

    def cuda(self):
        self.all_data, self.all_labels = self.all_data.cuda(), self.all_labels.cuda()
        # self.all_labels = self.all_labels.cuda()


"""
def SortingDataloader(torch.DataLoader ):
    def __init__(self, dataset, batch_size=10):
        self.dataset = dataset
"""


class SortingModel(nn.Module):
    def __init__(self, args):
        super(SortingModel, self).__init__()
        self.embed_mod = nn.Embedding(args.n_vocab, args.hidden_dim)
        self.transformer = nn.Transformer(
            d_model=args.hidden_dim, nhead=args.width, num_encoder_layers=0, num_decoder_layers=args.depth
        )
        self.classifier = nn.Linear(args.hidden_dim, args.num_labels)
        self.args = args

    def forward(self, X):
        """
        X is input to be sorted, Long index tensors
        """
        X = self.embed_mod(X)
        # for layer in self.transformers:
        #    X, s = layer(X)
        X = self.transformer(X, X)
        pred = self.classifier(X)
        return pred


def main_bert(args):
    # args = parse_args()
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    args.n_vocab = args.num_labels
    n_train_data = args.n_train_data
    n_eval_data = args.n_eval_data

    data_path = "data/sort_train_data{}_seq{}_v{}.pt".format(n_train_data, args.seq_len, args.n_vocab)
    pathlib.Path(data_path).parent.mkdir(exist_ok=True, parents=True)
    cache_data = True
    if cache_data:
        if os.path.exists(data_path):
            data_set = torch.load(data_path)
            train_data = data_set.branch_subset(n_train_data)
            print("dataset len {}".format(len(train_data)))
        else:
            train_data = SortingDataset(n_train_data, args)
            torch.save(train_data, data_path)
    else:
        train_data = SortingDataset(n_eval_data, args)

    # train_data = SortingDataset(n_train_data, args)
    if args.train_as_eval:
        eval_data = train_data.branch_subset(n_eval_data)
    else:
        eval_data = SortingDataset(n_eval_data, args)
    if cuda:
        train_data.cuda()
        eval_data.cuda()
    batch_sz = args.batch_sz
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=False)  # shuffle=False
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=len(eval_data), shuffle=False)
    config = SANConfig(
        vocab_size=args.n_vocab,
        hidden_size=args.hidden_dim,
        num_hidden_layers=args.depth,
        num_attention_heads=args.width,
        hidden_act="gelu",
        intermediate_size=args.hidden_dim,
        do_mlp=args.do_mlp,
        num_labels=args.num_labels,
    )

    model = SANForSequenceClassification(config=config)
    model_path = "modelsort{}d{}_{}h{}label{}.pt".format(
        args.width, args.depth, args.seq_len, args.hidden_dim, args.n_vocab
    )
    pathlib.Path(model_path).parent.mkdir(exist_ok=True, parents=True)
    if os.path.exists(model_path) and not args.do_train:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        args.n_epochs = 0
    if cuda:
        model = model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    config.cur_step = 0
    model.train()
    for epoch in tqdm(range(args.n_epochs)):
        for data in train_loader:
            X, labels = data
            pred = model(X)
            # (pooledoutput, hidden states)
            logits = pred[0]
            loss = loss_fn(logits.view(-1, args.num_labels), labels.view(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()
            config.cur_step += 1
        sys.stdout.write("train loss {}".format(loss))

    path_len = min(args.path_len, args.depth)  # 2
    n_repeat = args.n_repeat
    with open(args.res_file, "a") as f:
        f.write("~{}  \n".format(args))

    #'''
    # model_path = 'modelsort{}d{}_{}h{}'.format(args.width, args.depth, args.seq_len, args.hidden_dim)
    if not os.path.exists(model_path) or args.do_train:
        state_dict = model.state_dict()  # torch.load(model_path)
        torch.save(state_dict, model_path)

    ### eval ###
    config.no_sub_path = args.no_sub_path
    config.do_rank = args.do_rank
    config.compute_alpha = args.compute_alpha
    path_len = min(args.path_len, args.depth)  # 2
    n_repeat = args.n_repeat
    #'''
    metric_ar = np.zeros((1, args.depth + 1 - path_len))
    std_ar = np.zeros((1, args.depth + 1 - path_len))
    model.eval()

    while path_len <= args.depth:
        print("path len {}".format(path_len))
        for data in eval_loader:
            X, labels = data

            score_ar = np.zeros((n_repeat,))
            eval_loss_ar = np.zeros((n_repeat,))
            comp_match_ar = np.zeros((n_repeat,))
            part_match_ar = np.zeros((n_repeat,))

            for i in range(n_repeat):
                with torch.no_grad():
                    path_idx_l = []
                    for _ in range(args.n_paths):
                        # path_idx_l = [path_idx_l]
                        path_idx_l.append(create_path(path_len, args, all_heads=args.all_heads))

                    # must be nested idx arrays!
                    pred = model(X, path_idx_l=path_idx_l)
                    pred = pred[0]
                    comp_match_ar[i], part_match_ar[i] = run_eval(pred, labels)
                    eval_loss_ar[i] = loss_fn(pred.view(-1, args.num_labels), labels.view(-1))

                    pred = torch.argmax(pred, dim=-1)
                    # score_ar[i] = evaluate_metric2(pred, X)
                    score_ar[i] = evaluate_metric3(pred, labels)

            res_str = "path_len {} avg_mismatched_pairs {} eval_loss {} complete_match {} partial_match {}".format(
                path_len, score_ar.mean(), eval_loss_ar.mean(), comp_match_ar.mean(), part_match_ar.mean()
            )
            # print('\npath_len {} avg_mismatched_pairs {} eval_loss {} complete_match {} partial_match {}'.format(path_len, score_ar.mean(), eval_loss_ar.mean(), comp_match_ar.mean(), \
            #                                                                                                     part_match_ar.mean()))
            metric_ar[0, path_len - args.path_len] = score_ar.mean()
            std_ar[0, path_len - args.path_len] = score_ar.std()
            print("\n", res_str)
        with open(args.res_file, "a") as f:
            # f.write('~{}  \n'.format(args ))
            # f.write('path_len {} d/w {} mis_match {} eval_loss {} comp_match {} partial_match {}\n'.format(path_len, args.depth/args.width, score, eval_loss, complete_match, partial_match))
            f.write(res_str + "\n")

        path_len += 1  # = min(path_len+1, args.depth)

    plot_arg = plot.PlotArg(np.arange(metric_ar.shape[-1]), metric_ar, std=std_ar)
    # plot_arg.legend = ['# reversed pairs (lower better)']
    plot_arg.legend = ["Sorting Accuracy"]
    plot_arg.x_label = "Path length"
    # plot_arg.y_label = '# reversed pairs'
    plot_arg.y_label = "Average Sorting Accuracy"
    # plot_arg.title = 'Reversed Pairs vs Path Length for Sorting'
    plot.plot_scatter(plot_arg, fname="sorting{}d{}_{}".format(args.width, args.depth, args.hidden_dim))
    #'''
    plot_res_path = "sort_res{}d{}_{}_{}.pt".format(args.width, args.depth, args.hidden_dim, args.n_paths)
    torch.save({"metric_ar": metric_ar, "std_ar": std_ar}, os.path.join(plot.res_dir, plot_res_path))

    plot_res_ar = np.zeros((4, metric_ar.shape[-1]))
    plot_std_ar = np.zeros((4, metric_ar.shape[-1]))
    for i, pathLen in enumerate([5, 20]):
        try:
            results = torch.load("sort_res{}d{}_{}_{}.pt".format(args.width, args.depth, args.hidden_dim, pathLen))
        except FileNotFoundError:
            print("Note: Must run script for both 5 and 20 paths combinations to produce combined plot.")
            break
        plot_res_ar[i] = results["metric_ar"][0]
        plot_std_ar[i] = results["std_ar"][0]

    # this number is obtained by running the model and using all paths
    plot_res_ar[2, :] = 0.98
    plot_res_ar[3, :] = 0.1
    x_ar = np.tile(np.arange(metric_ar.shape[-1], dtype=float), (4, 1))
    x_ar[0] -= 0.05
    x_ar[1] += 0.05

    plot_arg = plot.PlotArg(x_ar, plot_res_ar, std=plot_std_ar)

    plot_arg.legend = ["5 paths", "20 paths", "Entire model", "Random predictor"]
    plot_arg.x_label = "Path length"
    # plot_arg.y_label = '# correctly predicted pairs' #'# reversed pairs'
    plot_arg.y_label = "Average Sorting Accuracy"
    plot_arg.title = ""  #'# Reversed Pairs vs Path Length for Sorting'
    plot.plot_scatter(plot_arg, fname="sort{}d{}_{}multi".format(args.width, args.depth, args.hidden_dim))
    #'''


def main(args):

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    args.n_vocab = args.num_labels
    # args.n_data = 10
    n_train_data = args.n_train_data  # 10
    n_eval_data = args.n_eval_data  # args.batch_sz #30
    train_data = SortingDataset(n_train_data, args)
    eval_data = SortingDataset(n_eval_data, args)
    if cuda:
        train_data.cuda()
        eval_data.cuda()
    batch_sz = args.batch_sz  # 2 #10
    # args.hidden_dim = 128
    # args.depth = 2
    # args.width = 2
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=False)  # shuffle=False
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=len(eval_data), shuffle=False)
    model = SortingModel(args)
    if cuda:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in tqdm(range(args.n_epochs)):
        for data in train_loader:
            X, labels = data
            pred = model(X)
            pdb.set_trace()
            loss = loss_fn(pred.view(-1, args.num_labels), labels.view(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()
        sys.stdout.write("train loss {}".format(loss))

    for data in eval_loader:
        X, labels = data
        with torch.no_grad():
            pred = model(X)
            complete_match, partial_match = run_eval(pred, labels)
            eval_loss = loss_fn(pred.view(-1, args.num_labels), labels.view(-1))
            pred = torch.argmax(pred, dim=-1)
            score = evaluate_metric(pred, X)

            print(
                "\navg mismatched pairs {} eval loss {}eval complete m {} eval partial match {}".format(
                    score, eval_loss, complete_match, partial_match
                )
            )
        with open(args.res_file, "a") as f:
            f.write("~{}  \n".format(args))
            f.write(
                "{} {} {} {} {}\n".format(args.depth / args.width, score, eval_loss, complete_match, partial_match)
            )


if __name__ == "__main__":
    args = arguments.parse_args()
    args.res_file = "results_sort.txt"
    if args.bert:
        main_bert(args)
    else:
        main(args)
