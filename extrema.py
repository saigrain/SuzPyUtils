import numpy as np
def extrema(x, do_max = True, do_min = True, strict = False, withend = False):
    '''
    Returns indices and values of the extrema of array x
    Options:
    do_max	If true, index maxima only
    do_min	If true, index minima only
    strict	If true, do not index changes to zero gradient
    withend	If true, always include x[0] and x[-1]
    '''	
    # Compute gradient
    dx =  x[1:] - x[:-1]
    dx = np.append(dx[0], dx)
    # Keep only the sign of the gradient (-1, 0 or 1)
    dx = np.sign(dx)
    # Define the threshold for whether to pick out changes to zero gradient
    threshold = 0
    if strict: threshold = 1
    # Now find changes of sign
    dx2 = dxs[1:] - dxs[:-1]
    if do_max and do_min:
	d2x = abs(d2x)
    elif do_max:
	d2x = -d2x
    # Take care of the two ends
    if withend:
	d2x[0] = 2
	d2x[-1] = 2
    # Sift out the list of extremas
    ind = np.nonzero(d2x > threshold)[0]
    return ind, x[ind]
