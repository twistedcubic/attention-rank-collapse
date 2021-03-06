import numpy
import numpy as np


def create_path(path_len, args, all_heads=False):
    path_idx_l = np.random.randint(0, high=args.width, size=(args.depth,))
    n_high = args.depth - path_len  # number of high indices
    rand_idx = np.random.choice(args.depth, size=(n_high,), replace=False)
    path_idx_l[rand_idx] = args.width
    if all_heads:
        print("Using all heads!")
        path_idx_l[:] = -1
    return path_idx_l


def composite_norm(x):
    """
    Composite l_1 l_\infty norm.
    Input: x, input tensor. Should be 2d.
    """
    x = x.detach().cpu().numpy()
    norm = numpy.linalg.norm(x, float("inf")) * numpy.linalg.norm(x, 1)

    return norm
