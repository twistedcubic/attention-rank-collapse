import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_labels", default=3, type=int, help="number of labels")
    parser.add_argument("--n_train_data", default=30, type=int, help="number of train data")
    parser.add_argument("--n_eval_data", default=30, type=int, help="number of eval data")
    parser.add_argument("--seed", default=None, type=int, help="set seed")
    parser.add_argument("--n_epochs", default=10, type=int, help="number of epochs")
    parser.add_argument("--width", default=2, type=int, help="width of SAN")
    parser.add_argument("--depth", default=4, type=int, help="depth of SAN")
    parser.add_argument("--hidden_dim", default=128, type=int, help="hidden dim")
    parser.add_argument("--seq_len", default=None, type=int, help="sequence len")
    parser.add_argument("--batch_sz", default=30, type=int, help="batch sz")
    parser.add_argument("--path_len", default=1, type=int, help="path length for a target path to be studied")
    parser.add_argument("--n_paths", default=10, type=int, help="Number of paths in path combination")
    parser.add_argument("--n_repeat", default=3, type=int, help="number of times to repeat")
    parser.add_argument(
        "--no_repeat",
        action="store_const",
        default=False,
        const=True,
        help="whether have repeat chars when generating sorting data",
    )
    parser.add_argument("--train_as_eval", action="store_const", default=False, const=True, help="train as eval")
    parser.add_argument(
        "--no_sub_path", action="store_const", default=False, const=True, help="no skip connection during inference"
    )
    parser.add_argument("--do_rank", action="store_const", default=False, const=True, help="do rank computation")
    parser.add_argument(
        "--compute_alpha", action="store_const", default=False, const=True, help="compute alpha ratio in LN"
    )
    parser.add_argument(
        "--circle_skip",
        action="store_const",
        default=False,
        const=True,
        help="apply skip connection to circle experiment",
    )
    parser.add_argument("--do_mlp", action="store_const", default=False, const=True, help="whether to do MLP")
    parser.add_argument(
        "--ffn2",
        action="store_const",
        default=False,
        const=True,
        help="whether to do 2nd feedforward layer in initial embeddings module",
    )
    parser.add_argument(
        "--do_train", action="store_const", default=False, const=True, help="train model even if cached ones exist"
    )
    parser.add_argument("--all_heads", action="store_const", default=False, const=True, help="use all heads / paths")
    parser.add_argument(
        "--bert",
        action="store_const",
        default=True,
        const=True,
        help="use models defined in modeling_sort.py rather than sanity check models. Currently tautological.",
    )
    parser.add_argument("--res_file", default="results/results.txt", type=str, help="path to results file ")

    args = parser.parse_args()

    if args.all_heads:
        args.n_paths = 1
        args.path_len = args.depth
    if args.seq_len is None:
        # e.g. for sorting
        args.seq_len = args.num_labels
    return args
