'''
multiplots: routines for producing plots with shared axes and nice big labels
'''

import pylab as plt

def dofig(figno = 1, nx = 1, ny = 1, test = False, aspect = 0, \
              textsize = 1, useTex = True, scale = 1):
    '''Set up window for multiple plots with shared axes'''
    # First make the labels nice and big and enable latex if requested
    plotpar = {'axes.labelsize': 16 * textsize,
               'text.fontsize': 16 * textsize,
               'legend.fontsize': 14 * textsize,
               'xtick.labelsize': 14 * textsize,
               'ytick.labelsize': 14 * textsize, 
               'text.usetex': useTex}
    plt.rcParams.update(plotpar)
    figx = 8 * nx * scale
    figy = 2 * ny * scale
    if aspect != 0: figy *= 1.5 * aspect
    plt.close(figno)
    plt.figure(figno, (figx, figy))
    inches_per_pt = 1.0/72.27 
    xoff = 0.2 * 8. / figx
    xoffr = 0.02 * 8. / figx
    yoff = 0.13 * 6. / figy
    yoffr = 0.06 * 6. / figy
    edges = xoff, xoffr, yoff, yoffr
    if test == True:
        for i in scipy.arange(nx):
            for j in scipy.arange(ny):
                doaxes(edges, nx, ny, i, j)
                plt.plot(scipy.arange(10)/100.0)
                if i == nx - 1: plt.xlabel('this is x')
                if j == ny / 2: plt.ylabel('this is y')
    return edges

def doaxes(edges, nx, ny, ix, iy, sharex = None, sharey = None, extra = 0.05):
    '''Set up one of multiple plots with shared axes'''
    xoff, xoffr, yoff, yoffr = edges
    xwi = (1.0 - xoff - xoffr) / float(nx)
    ywi = (1.0 - yoff - yoffr) / float(ny)
    xll = xoff + ix * xwi
    yll = 1.0 - yoffr - (iy+1) * ywi
    if nx > 1:
        xoffr -= extra * xwi
        xwi = (1-extra) * (1.0 - xoff - xoffr) / float(nx)
    if ny > 1:
        yoffr -= extra * ywi
        ywi = (1-extra) * (1.0 - yoff - yoffr) / float(ny)
    axc = plt.axes([xll, yll, xwi, ywi], sharex = sharex, sharey = sharey)
    if ix > 0:
        plt.setp(axc.get_yticklabels(), visible=False)
    if iy < (ny -1):
        plt.setp(axc.get_xticklabels(), visible=False)
    axc.yaxis.set_major_locator(plt.MaxNLocator(5, prune = 'both'))
    axc.xaxis.set_major_locator(plt.MaxNLocator(5, prune = 'both'))
    return axc

