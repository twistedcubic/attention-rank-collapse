import copy
import os
import sys

import numpy as np
import sklearn
import torch
import torch.nn as nn
from tqdm import tqdm

import arguments
import plot
from configuration_san import SANConfig
from functions import create_path
from modeling_san import SANForSequenceClassification


"""
Boolean prediction of whether a point is within the convex hull
"""
cuda = torch.cuda.is_available()


def evaluate_metric(pred_ar, gt_l, args):
    """
    How close the prediction probabilities agree with the ground truth labels.
    Cross entropy.
    """
    fn = nn.CrossEntropyLoss()

    pred_ar = pred_ar.view(-1, args.num_labels)  ##n_labels
    dist = fn(pred_ar, gt_l.view(-1))
    return dist


def evaluate_metric_auc(pred_ar, gt_l, args):
    """
    How close the prediction probabilities agree with the ground truth labels.
    AUC score.
    """
    total_match = 0
    pred_ar = torch.nn.functional.softmax(pred_ar, dim=-1)
    auc = sklearn.metrics.roc_auc_score(gt_l.cpu().numpy(), pred_ar[:, 1].cpu().numpy())

    return auc


def evaluate_metric_softmax_match(pred_ar, gt_l, args):
    """
    How close the prediction probabilities agree with the ground truth labels.
    """
    total_match = 0
    gt_l = gt_l
    # pred_ar = torch.nn.functional.softmax(pred_ar, dim=-1)
    pred = torch.argmax(pred_ar, dim=-1)
    total_match = float((torch.eq(pred, gt_l)).sum())
    return total_match / len(pred_ar)


def gen_convex_hull_sample(args):
    seq_len = args.seq_len
    coords = np.random.uniform(low=0, high=10, size=(seq_len, 2))  ####
    # center = np.array([5, 5])
    # coords -= center.reshape(1,-1)
    coords += np.random.normal(scale=3.3, size=(1, 2))

    label = torch.zeros((seq_len,), dtype=torch.int64)

    coords_sorted = sorted(coords, key=lambda x: x[0])
    high_coord = coords_sorted[0]
    high_idx = 0

    hull_boundary_set = set()
    hull_boundary_set.add(tuple(coords_sorted[0]))
    hull_boundary_set.add(tuple(coords_sorted[-1]))

    while high_idx < seq_len - 1:  # high != coords_sorted[-1]:
        # for i, coord in enumerate(coords_sorted):
        # find the max-sloped points, then the min-sloped points
        slopes_high = coords_sorted[high_idx + 1 :] - high_coord
        slopes_high = slopes_high[:, 1] / slopes_high[:, 0]
        argmax = np.argmax(slopes_high)
        high_coord = coords_sorted[high_idx + 1 :][argmax]
        high_idx = argmax + high_idx + 1  # idx_sorted[0]
        hull_boundary_set.add(tuple(high_coord))

    high_idx = 0
    high_coord = coords_sorted[0]
    while high_idx < seq_len - 1:  # high != coords_sorted[-1]:
        # for i, coord in enumerate(coords_sorted):
        # find the max-sloped points, then the min-sloped points
        slopes_high = coords_sorted[high_idx + 1 :] - high_coord
        slopes_high = slopes_high[:, 1] / slopes_high[:, 0]
        argmax = np.argmin(slopes_high)
        high_coord = coords_sorted[high_idx + 1 :][argmax]
        high_idx = argmax + high_idx + 1  # idx_sorted[0]
        hull_boundary_set.add(tuple(high_coord))

    for i, coord in enumerate(coords):
        if tuple(coord) in hull_boundary_set:
            label[i] = 1
    return torch.from_numpy(coords), label


def gen_data(n_data, args):
    """
    Generate data, random points, generate binary labels
    """
    seq_len = args.seq_len
    all_coords = torch.zeros(n_data, seq_len, 2)
    all_labels = torch.zeros(n_data, seq_len, dtype=torch.int64)
    for i in range(n_data):
        coords, label = gen_convex_hull_sample(args)
        all_coords[i] = coords
        all_labels[i] = label
    return all_coords, all_labels


def evaluate_metric0(pred_l, gt_l):
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
    # can normalize by {n \choose 2}
    # n_pairs = scipy.special.comb(len(pred_idx) , 2)
    # return total_mis.cpu().item()/len(pred_l)/2/n_pairs #len(pred_idx )
    # return total_mis.cpu().item()/len(pred_l)/2/len(pred_idx )
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


def gen_data_repeat(n_data, args):
    """
    Generate data tuples. Letter sequence and ground truth ordering.
    Allow repeats
    """
    # permutation.
    # letters = ['a'+i for i in range(26)]

    # n_data = args.n_data
    n_labels = args.num_labels
    # all_data = torch.arange(26)
    # all_data = torch.repeat(all_data, 0, n_data)
    # all_data = torch.stack([torch.randperm(n_labels) for _ in range(n_data) ])
    all_data = torch.randint(n_labels, (n_data, n_labels))

    sort_idx = torch.argsort(all_data, dim=-1)  # all_data.clone()

    true_labels = torch.zeros(n_data, n_labels, dtype=torch.int64)
    arange = torch.arange(n_labels, dtype=true_labels.dtype)
    # better use scatter
    for i, label in enumerate(sort_idx):
        true_labels[i][label] = arange

    return all_data, true_labels


class ConvexHullDataset(torch.utils.data.Dataset):
    def __init__(self, n_data, args):
        super(ConvexHullDataset, self).__init__()
        self.all_data, self.all_labels = gen_data(n_data, args)

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


class ConvexHullModel(nn.Module):
    """
    Model for sanity testing/cross referencing.
    """

    def __init__(self, args):
        super(ConvexHullModel, self).__init__()
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
    # args.n_data = 10
    n_train_data = args.n_train_data  # 10
    n_eval_data = args.n_eval_data  # 10 #30
    model_path = "model{}d{}_{}h{}_{}.pt".format(args.width, args.depth, args.seq_len, args.hidden_dim, args.n_epochs)
    print("model_path {}".format(model_path))
    data_path = "data/train_data_convexhull{}.pt".format(n_train_data)
    cache_data = True
    if cache_data or not os.path.exists(model_path):
        if os.path.exists(data_path):
            data_set = torch.load(data_path)
            train_data = data_set.branch_subset(n_train_data)
            print("dataset len {}".format(len(train_data)))
        else:
            train_data = ConvexHullDataset(n_train_data, args)
            torch.save(train_data, data_path)
    else:
        train_data = ConvexHullDataset(n_eval_data, args)

    if args.train_as_eval:
        eval_data = train_data.branch_subset(n_eval_data)
    else:
        eval_data = ConvexHullDataset(n_eval_data, args)
    if cuda:
        train_data.cuda()
        eval_data.cuda()
    batch_sz = args.batch_sz
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=False)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=len(eval_data), shuffle=False)
    config = SANConfig(
        vocab_size=2,  # no embeddings for convex hull.
        no_embed=True,
        hidden_size=args.hidden_dim,
        num_hidden_layers=args.depth,
        num_attention_heads=args.width,
        hidden_act="gelu",
        intermediate_size=args.hidden_dim,
        do_mlp=args.do_mlp,
        do_ffn2_embed=args.ffn2,
        max_position_embeddings=-1,
        num_labels=args.num_labels,
    )

    # This has been fitted for *token*-wise prediction
    # No embeddings layer, feedforward layer that projects coordinates into hidden_dim space instead
    model = SANForSequenceClassification(config=config)

    if os.path.exists(model_path) and not args.do_train:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        args.n_epochs = 0
    if cuda:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    do_regularization = True
    if do_regularization:
        opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)
        milestones = [15, 30, 40, 50, 60]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.91)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-5)  # , weight_decay=1e-3)

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
        if do_regularization:
            scheduler.step()

    # model_path = 'model{}d{}_{}h{}'.format(args.width, args.depth, args.seq_len, args.hidden_dim)
    if not os.path.exists(model_path) or args.do_train:
        state_dict = model.state_dict()  # torch.load(model_path)
        torch.save(state_dict, model_path)

    ### eval ###
    config.no_sub_path = args.no_sub_path
    config.do_rank = args.do_rank
    config.compute_alpha = args.compute_alpha
    path_len = min(args.path_len, args.depth)  # 2
    n_repeat = args.n_repeat
    with open(args.res_file, "a") as f:
        f.write("~{}  \n".format(args))
    model.eval()
    metric_ar = np.zeros((2, args.depth + 1 - path_len))
    std_ar = np.zeros((2, args.depth + 1 - path_len))
    while path_len <= args.depth:
        print("path len {}".format(path_len))
        for data in eval_loader:
            X, labels = data

            eval_loss_ar = np.zeros((n_repeat,))
            soft_match_ar = np.zeros((n_repeat,))


            for i in range(n_repeat):
                with torch.no_grad():
                    path_idx_l = []
                    for _ in range(args.n_paths):
                        # path_idx_l = [path_idx_l]
                        path_idx_l.append(create_path(path_len, args, all_heads=args.all_heads))
                        # path_idx_l.append(create_path(np.random.randint(1, high=path_len+1), args, all_heads=args.all_heads ))

                    # must be nested idx arrays!
                    pred = model(X, path_idx_l=path_idx_l)
                    pred = pred[0]

                    eval_loss_ar[i] = loss_fn(pred.view(-1, args.num_labels), labels.view(-1))
                    # pred = torch.argmax(pred, dim=-1)

                    soft_match_ar[i] = evaluate_metric_softmax_match(
                        pred.view(-1, args.num_labels), labels.view(-1), args
                    )
            res_str = "path_len {} eval_loss {} token_acc {}".format(
                path_len, eval_loss_ar.mean(), soft_match_ar.mean()
            )  # , comp_match_ar.mean(),
            metric_ar[0, path_len - args.path_len] = soft_match_ar.mean()
            metric_ar[1, path_len - args.path_len] = soft_match_ar.mean()
            std_ar[0, path_len - args.path_len] = soft_match_ar.std()
            std_ar[1, path_len - args.path_len] = soft_match_ar.std()
            print("\n", res_str)

        # print('Enter desired path len between 1 and {}: '.format(args.depth ))
        # path_len = input('Enter desired path len between 1  '  )
        # path_len = input()
        # path_len = int(path_len)
        with open(args.res_file, "a") as f:
            # f.write('~{}  \n'.format(args ))
            # f.write('path_len {} d/w {} mis_match {} eval_loss {} comp_match {} partial_match {}\n'.format(path_len, args.depth/args.width, score, eval_loss, complete_match, partial_match))
            f.write(res_str + "\n")

        path_len += 1  # = min(path_len+1, args.depth)

    plot_arg = plot.PlotArg(np.arange(metric_ar.shape[-1]), metric_ar, std=std_ar)
    # plot_arg.legend = ['AUC', 'Exact Match']
    plot_arg.legend = ["Accuracy", "Token Acc"]
    plot_arg.x_label = "Path length"
    # plot_arg.y_label = 'AUC'
    plot_arg.y_label = "Token Prediction Accuracy"
    # plot_arg.title = 'AUC vs Path Length for Convex Hull'
    plot.plot_scatter(plot_arg, fname="convex_hull{}d{}_{}".format(args.width, args.depth, args.hidden_dim))

    plot_res_path = os.path.join(plot.res_dir, "convex_hull_res{}d{}_{}_{}.pt".format(
        args.width, args.depth, args.hidden_dim, args.n_paths)
    )  #'convex_hull_res{}.pt'.format(args.n_paths)
    torch.save({"metric_ar": metric_ar, "std_ar": std_ar}, plot_res_path)

    plot_res_ar = np.zeros((4, metric_ar.shape[-1]))
    plot_std_ar = np.zeros((4, metric_ar.shape[-1]))
    for i, pathLen in enumerate([5, 20]):
        # torch.load('sort_res{}d{}_{}_{}.pt'.format(args.width, args.depth, args.hidden_dim, pathLen) )
        try:
            results = torch.load(
                "convex_hull_res{}d{}_{}_{}.pt".format(args.width, args.depth, args.hidden_dim, pathLen)
            )
        except FileNotFoundError:
            print("Note: Must run script for both 5 and 20 paths combinations to produce combined plot.")
            break
        plot_res_ar[i] = results["metric_ar"][0]
        plot_std_ar[i] = results["std_ar"][0]

    # this number is obtained by running the model and using all paths
    plot_res_ar[2, :] = 0.9
    plot_res_ar[3, :] = 0.54

    x_ar = np.tile(np.arange(metric_ar.shape[-1], dtype=float), (4, 1))
    x_ar[0] -= 0.05  # np.arange(metric_ar.shape[-1])-0.05
    x_ar[1] += 0.05  # np.arange(metric_ar.shape[-1])+0.05

    plot_arg = plot.PlotArg(x_ar, plot_res_ar, std=plot_std_ar)
    plot_arg.legend = ["5 paths", "20 paths", "Entire model", "Majority predictor"]
    plot_arg.x_label = "Path length"
    # plot_arg.y_label = 'AUC'
    plot_arg.y_label = "Token Prediction Accuracy"
    # plot_arg.title = 'AUC vs Path Length for Convex Hull'
    plot.plot_scatter(
        plot_arg,
        fname="convex_hull{}d{}_{}l{}multi".format(args.width, args.depth, args.hidden_dim, args.seq_len),
        loc="best",
        bbox=(0.5, 0.4, 0.5, 0.5),
    )  # plt.legend(loc='best', bbox_to_anchor=(0.5, 0.4, 0.5, 0.5))


def main(args):

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    args.n_vocab = args.num_labels
    # args.n_data = 10
    n_train_data = args.n_train_data  # 10
    eval_n_data = args.batch_sz  # 30
    train_data = ConvexHullDataset(n_train_data, args)
    eval_data = ConvexHullDataset(eval_n_data, args)
    if cuda:
        train_data.cuda()
        eval_data.cuda()
    batch_sz = args.batch_sz  # 2 #10
    # args.hidden_dim = 128
    # args.depth = 2
    # args.width = 2
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=False)  # shuffle=False
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=len(eval_data), shuffle=False)
    model = ConvexHullModel(args)
    if cuda:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in tqdm(range(args.n_epochs)):
        for data in train_loader:
            X, labels = data
            pred = model(X)
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
    args.num_labels = 2
    if args.bert:
        main_bert(args)
    else:
        main(args)
