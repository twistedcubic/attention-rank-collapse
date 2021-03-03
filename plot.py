

from __future__ import unicode_literals
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
import numpy
import scipy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os.path as osp
import os
import sort

import pdb
'''
Visualization functions, e.g. visualizing paths contributions.
Also contains utilities functions, eg composite norm.
'''

res_dir = 'results'
#res_dir = '/home/yihdong/results'

class PlotArg:
    def __init__(self, x, y, std=None):
        '''
        x has shape d
        y has shape k x d
        '''
        self.title = ''
        self.x_label = 'x'
        self.y_label = 'y'
        self.x = x
        self.y = y
        self.std = np.zeros(tuple(y.shape)) if std is None else std
        self.legend = ['']*len(y)

def compute_rep(args):
    """
    Compute path representation as function of k
    """
    rep_ar = np.zeros((args.depth,))
    alpha = args.alpha
    for i, k in enumerate(range(args.depth)):
        m_choose_k = scipy.special.comb(args.depth, k)
        rep_ar[i] = m_choose_k*args.width**k *args.alpha**(args.depth -k) * (1-alpha)**k * args.R0 / (args.c**(3**k-1))

        print(k, rep_ar[i])
##
def compute_rep_diff_alpha(args):
    """
    Compute path representation as function of k, diff alpha
    """
    rep_ar = [[] for _ in range(len(args.alpha_l) )] #np.zeros((args.depth,))
    compute_rep_rec(args, rep_ar, 0, 0, 1)
    
    print([sum(l) for l in rep_ar ])
    '''
    alpha = args.alpha
    for i, k in enumerate(range(args.depth)):
        m_choose_k = scipy.special.comb(args.depth, k)
        rep_ar[i] = m_choose_k*args.width**k *args.alpha**(args.depth -k) * (1-alpha)**k * args.R0 / (args.c**(3**k-1))

        print(k, rep_ar[i])
    '''
##
def compute_rep_rec(args,  rep_ar, cur_path_len, cur_idx, cur_alpha_prod):
    '''
    rep_ar is list of lists
    '''
    if cur_idx == len(args.alpha_l):
        if cur_path_len == 0:
            return
        try:
            cur_rep = args.R0 / (args.c**(3**cur_path_len-1)) * cur_alpha_prod
        except Exception as e:
            print(e)
            pdb.set_trace()
        rep_ar[cur_path_len-1].append(cur_rep)
        return
    #skip
    compute_rep_rec(args, rep_ar, cur_path_len, cur_idx+1, cur_alpha_prod*args.alpha_l[cur_idx ])
    #non skip
    compute_rep_rec(args, rep_ar, cur_path_len+1, cur_idx+1, cur_alpha_prod*(1-args.alpha_l[cur_idx ]))
    
    
def plot_scatter(plot_arg, fname, xlim=None, ylim=None,loc=None, bbox=None):
    #plt.plot(data_ar[0], data_ar[i], marker=markers[i-1], label=legend_l[i-1])
    plt.clf()
    markers = ['^', 'o', 'x', '.', '1', '3', '+', '4', '5']
    plt.rcParams.update({'font.size': 14.5})
    
    for i, y in enumerate(plot_arg.y):
        if len(plot_arg.x.shape) > 1:
            x_ar = plot_arg.x[i]
        else:
            x_ar = plot_arg.x
        if plot_arg.std[i].sum() == 0:
            plt.plot(x_ar, y, label=plot_arg.legend[i]) #,  linestyle="None")
        else:
            plt.errorbar(x_ar, y, yerr=plot_arg.std[i], marker=markers[i], label=plot_arg.legend[i]) #,  linestyle="None")
        
    plt.grid(True)
    if loc is not None:
        plt.legend(loc=loc, bbox_to_anchor=bbox)
    else:
        plt.legend()
    if xlim is not None:
        plt.xlim(xlim) #([-.5, .5])
    if ylim is not None:
        plt.ylim(ylim) #[-.5, .5])
    plt.xlabel(plot_arg.x_label)
    plt.ylabel(plot_arg.y_label)
    plt.title(plot_arg.title)
    
    fig_path = osp.join(res_dir, 'plot_{}.pdf'.format(fname )) #'baselines_{}{}{}.jpg'.format(opt.type, name, fname_append))
    plt.savefig(fig_path)
    print('figure saved under {}'.format(fig_path))    

def plot_plot(plot_arg, fname, xlim=None, ylim=None,loc=None, bbox=None):
    #plt.plot(data_ar[0], data_ar[i], marker=markers[i-1], label=legend_l[i-1])
    plt.clf()
    markers = ['^', 'o', 'x', '.', '1', '3', '+', '4', '5']
    plt.rcParams.update({'font.size': 14.5})
    
    for i, y in enumerate(plot_arg.y):
        if len(plot_arg.x.shape) > 1:
            x_ar = plot_arg.x[i]
        else:
            x_ar = plot_arg.x
        plt.errorbar(x_ar, y, yerr=plot_arg.std[i], marker=markers[i], label=plot_arg.legend[i]) #,  linestyle="None")
        
    plt.grid(True)
    if loc is not None:
        plt.legend(loc=loc, bbox_to_anchor=bbox)
    else:
        plt.legend()
    if xlim is not None:
        plt.xlim(xlim) #([-.5, .5])
    if ylim is not None:
        plt.ylim(ylim) #[-.5, .5])
    plt.xlabel(plot_arg.x_label)
    plt.ylabel(plot_arg.y_label)
    plt.title(plot_arg.title)
    
    fig_path = osp.join(res_dir, 'plot_{}.pdf'.format(fname )) #'baselines_{}{}{}.jpg'.format(opt.type, name, fname_append))
    plt.savefig(fig_path)
    print('figure saved under {}'.format(fig_path))    

def composite_norm(x):
    '''
    Composite l_1 l_\infty norm.
    Input: x, input tensor. Should be 2d.
    '''
    x = x.detach().cpu().numpy()
    norm = numpy.linalg.norm(x, float('inf'))*numpy.linalg.norm(x, 1)
    
    return norm
    
if __name__=='__main__':
    args = sort.parse_args()
    args.c = 1.0491046 #0.000491046376 #1.3
    args.R0 = 400 #48 #3
    args.alpha = .6
    args.alpha_l = [0.90140117, 0.90491132, 0.90742852, 0.88882977, 0.85639907, 0.77844384]
    #[0.88690354 0.90293881 0.66461398 0.79351534 0.6367262  0.82008575  0.81519226 0.81471389 0.87243967 0.57299129]
    print(args)
    compute_rep_diff_alpha(args)
