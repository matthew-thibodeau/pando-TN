# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:04:03 2022

@author: mthibode
"""

import numpy as np
import glob
from copy import copy
from collections import defaultdict
import matplotlib.pyplot as plt
from fractions import Fraction


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')


Dval = 3
Lvals = [40,80]
topdir = 'mps_data/data'


axtimes = [None,None]
figtime, axtimes = plt.subplots(constrained_layout=True, nrows=1, ncols=2, sharey=True)
figtime.set_size_inches(3.125, 1.75)

axerrs = [None,None]
figerr, axerrs = plt.subplots(constrained_layout=True, nrows=1, ncols=2, sharey=True)
figerr.set_size_inches(3.125, 1.75)

fig_corr, ax_corrs = plt.subplots(constrained_layout=True, nrows=1, ncols=2, sharey=True)
fig_corr.set_size_inches(3.125, 1.75)

plotmode = r'$\times$'
cmap = 'inferno'

for counter, (Lval, axerr, axtime, ax_corr) in enumerate(zip(Lvals, axerrs, axtimes, ax_corrs)):

    listing = glob.glob(f'{topdir}/time_data*_L*D{Dval}.npy')
    exactdict = defaultdict(list)
    approxdict = defaultdict(list)
    onesitedict = defaultdict(list)
    baddict = defaultdict(list)
    for fname in listing:
        x = fname[:-4]
        xl = x.split('_')
        dstr = xl[-1]
        dval = int(dstr[1:])
        lstr = xl[-2]
        lval = int(lstr[1:])
        data = np.load(fname)
    
        # data = (t1site, conv1, t_exactguided, conv_exactguided, t_nnguided, conv_nnguided, t2)
        
        exactval = data[6]/data[2]#
        approxval = data[6]/data[4]#(
        onesiteval = data[6]/data[0]#(
        if data[1] and data[3] and data[5]:
            exactdict[(dval, lval)].append(exactval)
            approxdict[(dval, lval)].append(approxval)
            onesitedict[(dval, lval)].append(onesiteval)
        else:
            baddict[(dval, lval)].append(exactval)
    
    energylisting = glob.glob(f'{topdir}/energy_data*_L*D{Dval}.npy')
    exact_e_dict = defaultdict(list)
    approx_e_dict = defaultdict(list)
    onesite_e_dict = defaultdict(list)
    for fname in energylisting:
        x = fname[:-4]
        xl = x.split('_')
        dstr = xl[-1]
        dval = int(dstr[1:])
        lstr = xl[-2]
        lval = int(lstr[1:])
        data = np.load(fname)
        exactval = np.real(data[0])
        approxval = np.real(data[1])
        osval = np.real(data[2])
        exact_e_dict[(dval, lval)].append(exactval)
        approx_e_dict[(dval, lval)].append(approxval)
        onesite_e_dict[(dval, lval)].append(osval)
    
    
    
    # avg and std of each bin
    exstatsdict = {}
    apstatsdict = {}
    exenergystatsdict = {}
    apenergystatsdict = {}
    
    for (statd,datad) in [(exstatsdict,exactdict), (apstatsdict, approxdict),
                          (exenergystatsdict,exact_e_dict), (apenergystatsdict, approx_e_dict)]:
        for k in datad.keys():
            statd[k[0]] = []
        for k,v in datad.items():
            m = np.mean(v)
            s = np.std(v)
            d = k[0]
            l = k[1]
            statd[d].append((l,m,s))
    
    keys = exstatsdict.keys()
    dims = list(set(keys))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    exdata = np.array(exactdict[Dval, Lval])
    apdata = np.array(approxdict[Dval, Lval])
    osdata = np.array(onesitedict[Dval, Lval])
    
    ex_approx_os_tuples = list(zip(exactdict[Dval, Lval], approxdict[Dval, Lval], onesitedict[Dval, Lval]))
    ex_approx_tuples = list(zip(exactdict[Dval, Lval], approxdict[Dval, Lval]))
    ex_os_tuples = list(zip(exactdict[Dval, Lval], onesitedict[Dval, Lval]))
    approx_os_tuples = list(zip(approxdict[Dval, Lval], onesitedict[Dval, Lval]))
    
    # ex_approx_os_tuples_normalized = [[x/t[0] for x in t] for t in ex_approx_os_tuples]
    # _, approxnorm, osnorm = zip(*ex_approx_os_tuples_normalized)
    
    
    
    
    if plotmode == 'percent':
        plabel = '%'
        exdata = (np.array(exactdict[Dval, Lval]) - 1) * 100
        apdata = (np.array(approxdict[Dval, Lval]) - 1) * 100
        osdata = (np.array(onesitedict[Dval, Lval]) - 1) * 100
        
        lineheight = 100
        
    elif plotmode == r'$\times$':
        plabel = '%'
        lineheight = 1
        
        axtime.set_yticks([2*k + 1 for k in range(10)])
        
    else:
        raise ValueError('plotting mode not recognized')
        
    binmin, binmax = 0.1, 3.9
    ax_corr.hist2d(exdata, apdata, bins = (50,50), range = [[binmin, binmax],[binmin, binmax]], 
                   density=True, cmap = cmap,  rasterized=True)
    ax_corr.plot(np.linspace(binmin, binmax, 2), np.linspace(binmin, binmax, 2), color='white', linewidth = 0.4)
    ax_corr.set_xticks([1,2,3])
    ax_corr.set_yticks([1,2,3])
    
    ax_corr.set_title(rf'$L = {Lval}$', loc='right', fontsize=12)
    if counter == 0:
        fig_corr.supxlabel(rf'Oracle-guided speedup, {plotmode}')#,labelpad=0)
    ax_corr.set_ylabel(rf'NN-guided speedup, {plotmode}' if counter == 0 else '')#,labelpad=0)
        
    axtime.boxplot([apdata, exdata], widths = 0.3, sym='')
    axtime.set_ylabel(f'Wall-time speedup, {plotmode}' if counter == 0 else '')#,labelpad=0)
    axtime.set_xticks([1,2], labels=['NN', 'oracle'])#, 'vanilla 1-site'])
    axtime.hlines(lineheight, axtime.get_xlim()[0],  axtime.get_xlim()[1], color='k', linestyle=':')
    axtime.set_title(rf'$L = {Lval}$', loc='right', fontsize=12)
    
    
    exedata = np.array(exact_e_dict[Dval, Lval]) / Lval
    apedata = np.array(approx_e_dict[Dval, Lval]) / Lval
    osedata = np.array(onesite_e_dict[Dval, Lval]) / Lval
    axerr.boxplot([apedata, exedata], widths = 0.3, sym='')
    axerr.set_ylabel(r'$E / E_{2-site} - 1$' if counter == 0 else '')#,labelpad=0)
    axerr.set_xticks([1,2,], labels=['NN', 'oracle',])# 'vanilla 1-site'])
    axerr.set_title(rf'$L = {Lval}$', loc='right', fontsize=12)
    
    
    # plt.figure()
    # for (sdict, label) in zip([exstatsdict, apstatsdict], ['exact', 'approx']):
    #     for d,v in sdict.items():
    #         color = colors.pop(0)
    #         vs = sorted(v, key = lambda t: t[0])
    #         x,m,s = zip(*vs)
    #         plt.errorbar(x, m, yerr=s, c=color, linestyle=None, capsize=5, label=label, marker='x')
    
    # plt.legend()
    # plt.xlabel("Length of spin chain (# of tensors)")
    # plt.ylabel("% reduction in wall time\nfor GS preparation")
    # plt.title("GS prep. is accelerated with entanglement knowledge")
