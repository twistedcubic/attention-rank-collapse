import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import arguments
import plot
from configuration_san import SANConfig
from modeling_san import SANForSequenceClassification


"""
Boolean prediction of whether a point is within the convex hull
"""
cuda = torch.cuda.is_available()


def gen_circle_data(args):
    rad = 0.3
    n_total = args.n_train_data
    theta = torch.arange(n_total) * 2 * np.pi / n_total

    theta = theta.view(-1, 1)
    x = torch.zeros_like(theta)
    y = torch.zeros_like(theta)
    x[: n_total // 4] = rad * torch.cos(theta[: n_total // 4])
    y[: n_total // 4] = rad * torch.sin(theta[: n_total // 4])
    x[n_total // 4 : 2 * n_total // 4] = -rad * torch.cos(np.pi - theta[n_total // 4 : 2 * n_total // 4])
    y[n_total // 4 : 2 * n_total // 4] = rad * torch.sin(np.pi - theta[n_total // 4 : 2 * n_total // 4])
    x[2 * n_total // 4 : 3 * n_total // 4] = -rad * torch.cos(theta[2 * n_total // 4 : 3 * n_total // 4] - np.pi)
    y[2 * n_total // 4 : 3 * n_total // 4] = -rad * torch.sin(theta[2 * n_total // 4 : 3 * n_total // 4] - np.pi)
    x[3 * n_total // 4 : 4 * n_total // 4] = rad * torch.cos(2 * np.pi - theta[3 * n_total // 4 : 4 * n_total // 4])
    y[3 * n_total // 4 : 4 * n_total // 4] = -rad * torch.sin(2 * np.pi - theta[3 * n_total // 4 : 4 * n_total // 4])
    data = torch.cat((x, y), -1)
    # noise = torch.rand_like(data)*.08-.04
    # data += noise
    return data


class CircleDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super(CircleDataset, self).__init__()
        # self.all_data, self.all_labels = gen_data(n_data, args)
        self.all_data = gen_circle_data(args)
        self.args = args

    def __len__(self):

        return len(self.all_data) // 2 - 1

    def __getitem__(self, idx):

        return (
            torch.stack((self.all_data[idx], self.all_data[idx + len(self)]), 0),
            torch.stack((self.all_data[idx + 1], self.all_data[idx + 1 + len(self)]), 0),
        )

    def branch_subset(self, n_data):
        new_set = copy.deepcopy(self)
        new_set.all_data = self.all_data[:n_data]
        new_set.all_labels = self.all_labels[:n_data]
        return new_set

    def cuda(self):
        # self.all_data, self.all_labels = self.all_data.cuda(), self.all_labels.cuda()
        self.all_data = self.all_data.cuda()


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


def main(args):
    # args = parse_args()
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    # args.num_labels = 2
    args.n_vocab = args.num_labels
    # args.n_data = 10
    n_train_data = args.n_train_data
    n_eval_data = args.n_eval_data
    model_path = "modelCircle{}d{}_h{}n{}{}{}.pt".format(
        args.width,
        args.depth,
        args.hidden_dim,
        args.n_train_data,
        "mlp" if args.do_mlp else "",
        "skip" if args.circle_skip else "",
    )
    data_path = "data/circle_train_data{}.pt".format(n_train_data)
    cache_data = False
    if cache_data:  # or not os.path.exists(model_path):
        if os.path.exists(data_path):
            data_set = torch.load(data_path)
            train_data = dataset
            # train_data = data_set.branch_subset(n_train_data)
            print("dataset len {}".format(len(train_data)))
        else:
            train_data = CircleDataset(args)
            torch.save(train_data, data_path)
    else:
        train_data = CircleDataset(args)

    if args.train_as_eval:
        eval_data = train_data.branch_subset(n_eval_data)
    else:
        eval_data = CircleDataset(args)
    if cuda:
        train_data.cuda()
        eval_data.cuda()
    batch_sz = args.batch_sz
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=False)  # shuffle=False
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=len(eval_data), shuffle=False)
    config = SANConfig(
        vocab_size=2,  # args.n_vocab,
        no_embed=True,
        hidden_size=args.hidden_dim,
        num_hidden_layers=args.depth,
        num_attention_heads=args.width,
        hidden_act="gelu",
        intermediate_size=args.hidden_dim,
        do_mlp=args.do_mlp,
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
    else:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    if cuda:
        model = model.cuda()

    loss_fn = nn.MSELoss(reduction="none")

    if args.circle_skip:
        LN = (
            lambda x: x
        )  # x : .5 * x/torch.sqrt((x**2).sum(-1)).unsqueeze(-1) #torch.nn.LayerNorm(2, eps=1e-5, elementwise_affine=False)
    else:
        LN = lambda x: x
    config.cur_step = 0

    zeros_like_X = torch.zeros(args.batch_sz, 2, 2)
    for epoch in tqdm(range(args.n_epochs)):
        X = torch.zeros(args.batch_sz, 2, 2)
        for data in train_loader:
            prev_X = X if args.circle_skip else zeros_like_X
            X, labels = data
            pred = model(LN(X + prev_X[: len(X)]))
            # pred = model(X )
            # (pooledoutput, hidden states)
            logits = pred[0]

            # loss = loss_fn(logits.view(-1 ), labels.view(-1))
            loss = loss_fn(logits, labels)

            # loss[::2] *= 2
            loss = loss.sum()

            opt.zero_grad()
            loss.backward()
            opt.step()
            config.cur_step += 1
        sys.stdout.write("train loss {}".format(loss))

    # model_path = 'model{}d{}_{}h{}'.format(args.width, args.depth, args.seq_len, args.hidden_dim)
    if not os.path.exists(model_path) or args.do_train:
        state_dict = model.state_dict()
        torch.save(state_dict, model_path)

    ### eval ###
    config.no_sub_path = args.no_sub_path
    config.do_rank = args.do_rank
    path_len = min(args.path_len, args.depth)  # 2
    n_repeat = args.n_repeat
    with open(args.res_file, "a") as f:
        f.write("~{}  \n".format(args))
    model.eval()
    metric_ar = np.zeros((2, args.depth + 1 - path_len))
    cur_pt = train_data[0][0]
    n_eval = n_eval_data  # 600 #100 #20 #80 for 84 dim
    traj = torch.zeros((n_eval, 4))
    traj[0, :2] = cur_pt[0]  # torch.Tensor([.5,0])
    traj[0, 2:] = cur_pt[1]
    cur_pt = cur_pt.unsqueeze(0)

    if args.circle_skip:
        cur_pt_no_LN = cur_pt
        cur_pt = LN(cur_pt)
    for i in range(1, n_eval):
        with torch.no_grad():
            logits = model(cur_pt)
            if args.circle_skip:
                pred_pt = logits[0]
                cur_pt_no_LN = logits[0] + cur_pt_no_LN
                cur_pt = LN(cur_pt_no_LN)
                traj[i, :2] = pred_pt[0][0]
                traj[i, 2:] = pred_pt[0][1]
                # traj[i, :2] = cur_pt[0][0]
                # traj[i, 2:] = cur_pt[0][1]
            else:
                cur_pt = logits[0]
                traj[i, :2] = cur_pt[0][0]
                traj[i, 2:] = cur_pt[0][1]

    print("traj {}".format(traj))
    x_ar = torch.stack((traj[:, 0], traj[:, 2]), 0)
    y_ar = torch.stack((traj[:, 1], traj[:, 3]), 0)
    plot_arg = plot.PlotArg(x_ar, y_ar)
    plot_arg.legend = ["sequence 1", "sequence 2"]
    plot_arg.x_label = "x"
    plot_arg.y_label = "y"
    # plot_arg.title = 'Trajectories of point sequences via recurrent\n attention, {} hidden dim{}{}'\
    plot_arg.title = "{} hidden dim, {} skip, {} MLP".format(
        args.hidden_dim, "w/" if args.circle_skip else "no", "w/" if args.do_mlp else "no"
    )
    # plot_arg.title = 'Training trajectories of point sequences'
    plot.plot_scatter(
        plot_arg,
        fname="circle{}d{}_{}{}{}".format(
            args.width, args.depth, args.hidden_dim, "mlp" if args.do_mlp else "", "skip" if args.circle_skip else ""
        ),
        xlim=[-0.56, 0.56],
        ylim=[-0.56, 0.56],
    )

    res_path = "resCircle{}d{}_h{}n{}".format(args.width, args.depth, args.hidden_dim, args.n_train_data)  # ,\
    #                                                       'mlp' if args.do_mlp else '', 'skip' if args.circle_skip else '')

    if os.path.exists(res_path):
        res_dict = torch.load(res_path)
    else:
        res_dict = {}
    if args.do_mlp:
        res_key = "mlp"
    elif args.circle_skip:
        res_key = "skip"
    else:
        res_key = "base"
    res_dict[res_key] = traj
    dir_path = plot.res_dir
    res_dict["info"] = "L{}H{}dim{}".format(args.depth, args.width, args.hidden_dim)
    torch.save(res_dict, os.path.join(dir_path, res_path))
    res_path1 = "resCircle{}d{}_h{}n{}{}{}".format(
        args.width,
        args.depth,
        args.hidden_dim,
        args.n_train_data,
        "mlp" if args.do_mlp else "",
        "skip" if args.circle_skip else "",
    )
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    torch.save(traj, os.path.join(dir_path, res_path1))


if __name__ == "__main__":
    args = arguments.parse_args()
    args.num_labels = 2
    args.depth = 1

    main(args)
