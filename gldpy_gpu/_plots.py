"""Plots of the NumPy `GLD` class. Importing this module imports matplotlib, so `_gld_module` loads it on first use."""
import matplotlib.pyplot as plt
import numpy as np


def plot_cdf(gld, param_list, data=None, ymin=0.01, ymax=0.99, n_points=100, names=None, color_emp='lightgrey', colors=None):
    param_list = np.array(param_list)
    if param_list.ndim==1:
        param_list = param_list.reshape(1,-1)
    if names is None:
        names = [str(x) for x in param_list]
    if colors is None:
        colors = [None]*len(param_list)
    plt.figure()
    plt.grid()
    if  not (data is None):
        data = data.ravel()
        plt.plot(np.sort(data), np.arange(len(data))/len(data),color = color_emp,lw = 2)
        names = np.hstack(['empirical data', names ])
    y = np.linspace(ymin,ymax,n_points)
    for i in range(param_list.shape[0]):
        param = param_list[i]
        plt.plot(gld.Q(y,param), y, color = colors[i])
    plt.ylim(ymin = 0)
    plt.legend(names,bbox_to_anchor=(1.0,  1.0 ))
    plt.title('CDF')


def plot_pdf(gld, param_list, data=None, ymin=0.01, ymax=0.99, n_points=100, bins=None, names=None, color_emp='lightgrey', colors=None):
    param_list = np.array(param_list)
    if param_list.ndim==1:
        param_list = param_list.reshape(1,-1)
    if names is None:
        names = [str(x) for x in param_list]
    plt.figure()
    plt.grid()
    pdf_max = 0
    if not data is None:
        data = data.ravel()
        p = plt.hist(data, bins = bins, color = color_emp, density = True)
        pdf_max = np.max(p[0])
    if colors is None:
        colors = [None]*len(param_list)
    y = np.linspace(ymin,ymax,n_points)

    for i in range(param_list.shape[0]):
        param = param_list[i]
        plt.plot(gld.Q(y,param), gld.PDF_Q(y,param),color = colors[i])
        pdf_max = np.max([pdf_max,np.max(gld.PDF_Q(y,param))])
    plt.ylim(ymin = 0,ymax = pdf_max * 1.05)
    plt.legend(names,bbox_to_anchor=(1.0,  1.0 ))
    plt.title('PDF')


def plot_fitting(gld, data, param, bins=None):
    data = data.ravel()
    fig,ax = plt.subplots(1,3,figsize = (15,3))
    ax[0].hist(data,bins = bins,density = True,color = 'skyblue')
    y = np.linspace(0.001,0.999,100)
    ax[0].plot(gld.Q(y,param),gld.PDF_Q(y,param),lw = 2,color = 'r')
    ax[0].set_title('PDF')
    ax[0].grid()
    ax[1].plot(np.sort(data), np.arange(len(data))/len(data))
    ax[1].plot(gld.Q(y,param), y)
    ax[1].grid()
    ax[1].set_title('CDF')
    x = np.sort(data)
    y = (np.arange(len(data))+0.5)/len(data)
    ax[2].plot(gld.Q(y,param), x,'bo',ms = 3)
    m1 = np.min([x,gld.Q(y,param)])
    m2 = np.max([x,gld.Q(y,param)])
    ax[2].plot([m1,m2], [m1,m2],'r')
    ax[2].grid()
    ax[2].set_title('Q-Q-plot')


def plot_curve_fit(gld, data, param1, param, x, y, c, shift_val, bins_hist=None):
    fig, ax = plt.subplots(1,2,figsize = (12,3.5))
    ax[0].grid()
    ax[0].hist(data,color = 'skyblue',density = True, bins = bins_hist)
    p = np.linspace(0.01,0.99,100)
    ax[0].plot(gld.Q(p,param1), gld.PDF_Q(p,param1),'r')
    ax[0].set_title('Simulation phase')
    y_fit = gld.PDF_num(x,param)*c + shift_val
    y = y + shift_val
    delta = np.max([0.05*np.abs(np.mean(y)), 10**(-5)])
    ax[1].grid()
    ax[1].plot(x,y,'b')
    ax[1].plot(x,y_fit,'r')
    ax[1].set_ylim(np.min([y,y_fit]) - delta,np.max([y,y_fit])+delta)
    ax[1].legend(['data', 'GLD'])
    ax[1].set_title('Result of curve fitting')
