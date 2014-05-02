import scipy, pylab, numpy
import scipy.spatial as ssp
import scipy.optimize as sop
from planetc import transit

'''GP regression routines with a few different kernels, and
hyper-parameter tuning using max likelihood or a simple MCMC. Incudes
a transit mean function which requires Tom Evans's planetc module
(just comment out the relevant code if you don't have that module).'''

def GP_covmat(X1, X2, par, typ = 'SE', sigma = None):
    '''
    Compute covariance matrix with or without white noise for a range of
    GP kernels. Currently implemented:
    - SE (squared exponential 1D, default)
    - SE_ARD (squared exponential with separate length scales for each input dimension)
    - M32 (Matern 32, 1D)
    - QP (quasi-periodic SE, 1D)
    '''
    if typ == 'QP':
        DD = ssp.distance.cdist(X1, X2, 'euclidean')
        K = par[0]**2 * \
            scipy.exp(- (scipy.sin(scipy.pi * DD / par[1]))**2 / 2. / par[2]**2 \
                      - DD**2 / 2. / par[3]**2) 
    if typ == 'Per':
        DD = ssp.distance.cdist(X1, X2, 'euclidean')
        K = par[0]**2 * \
            scipy.exp(- (scipy.sin(scipy.pi * DD / par[1]))**2 / 2. / par[2]**2) 
    elif typ == 'M32':
        DD = ssp.distance.cdist(X1, X2, 'euclidean')
        arg = scipy.sqrt(3) * abs(DD) / par[1]
        K = par[0]**2 * (1 + arg) * scipy.exp(- arg)
    elif typ == 'SE_ARD':
        V = numpy.abs(numpy.matrix( numpy.diag( 1. / numpy.sqrt(2) / par[1:]) ))
        D2 = ssp.distance.cdist(X1 * V, X2 * V, 'sqeuclidean')
        K = par[0]**2 * numpy.exp( -D2 )
    else: # 'SE (radial)'
        D2 = ssp.distance.cdist(X1, X2, 'sqeuclidean')
        K = par[0]**2 * scipy.exp(- D2 / 2. / par[1]**2)
    if sigma != None:
        N = X1.shape[0]
        K += sigma**2 * scipy.identity(N)
    return scipy.matrix(K)

def transit_MF(p, x):
    '''Compute transit light curve for parameters p and times x'''
    pars = {}
    pars['P'] = p[0]
    pars['T0'] = p[1]
    pars['RpRs'] = p[2]
    pars['aRs'] = p[3]
    pars['incl'] = p[4]
    pars['ecc'] = p[5]
    pars['omega'] = p[6]
    pars['foot'] = p[7]
    pars['grad'] = p[8]
    if len(p) == 11:
        pars['ld'] = 'quad'
        pars['gam1'] = p[9]
        pars['gam2'] = p[10]
    else:
        pars['ld'] = 'nonlin'
        pars['c1'] = p[9]
        pars['c2'] = p[10]
        pars['c3'] = p[11]
        pars['c4'] = p[12]
    return transit.ma02_aRs(x, **pars)

def step_MF(p, x):
    y = numpy.zeros(len(x)) + p[0]
    l = x > p[1]
    y[l] = p[2]
    return y

def GP_negloglik(p, x, y, cov_func = None, cov_typ = 'SE', MF = None, n_MF_par = 0, \
                 MF_args = None, fixed = None, fixed_par = None, \
                 prior = None):
    '''
    Compute negative log likelihood for GP, optionally with:
    - fixed parameters (indices in fixed, values in fixed_par)
    - simple prior specs (mean and standard dev of Gaussian prior)
    - mean function MF
    The parameters of both covariance and mean function should be in
    the p array, MF pars last. The total number of MF pars
    (variable+fixed) should be given by n_MF_par. The separation
    between cov and MF pars is done after reinserting the fixed pars.
    '''
    if fixed == None:
        par = scipy.copy(p)
    else:
        par = scipy.zeros(len(fixed))
        par[fixed == True] = fixed_par
        par[fixed == False] = p
    if (MF != None):
        r = y - scipy.matrix([MF(par[-n_MF_par:], MF_args)]).T
    else:
        r = y[:]
    if cov_func == None:
        K = GP_covmat(x, x, par[:-n_MF_par-1], typ = cov_typ, sigma = par[-n_MF_par-1])
    else:
        if n_MF_par == 0:
            covpar_end = len(par)
        else:
            covpar_end = -n_MF_par
        K = cov_func(x, x, par[:covpar_end], white_noise = True)            
    try:
        L = scipy.linalg.cho_factor(K)
    except scipy.linalg.LinAlgError:
        pylab.clf()
        pylab.imshow(K, interpolation = 'nearest')
        print 'Warning: covariance matrix was not positive definite'
        raw_input('continue?')
        return 1e20
    a = numpy.log(numpy.diag(L[0])).sum()
    b = scipy.linalg.cho_solve(L, r)
    a += 0.5 * r.T * scipy.matrix(b)
    if prior != None:
        for i in scipy.arange(len(p)):
            if prior[i,1] > 0:
                a += ((par[i] - prior[i,0]) / prior[i,1])**2 
    return a

def GP_train(x, y, cov_par, cov_func = None, cov_typ ='SE', \
             cov_fixed = None, prior = None, \
             MF = None, MF_par = None, MF_args = None, \
             MF_fixed = None):
    '''    
    Max likelihood optimization of GP hyper-parameters. Calls
    GP_negloglik. Takes care of merging / splitting the fixed /
    variable and cov / MF parameters
    '''
    if MF != None:
        merged_par = scipy.append(cov_par, MF_par)
        n_MF_par = len(MF_par)
        fixed = scipy.append(scipy.zeros(len(cov_par), 'bool'), \
                             scipy.zeros(n_MF_par, 'bool'))
        if (cov_fixed != None): fixed[0:-n_MF_par] = cov_fixed
        if (MF_fixed != None): fixed[-n_MF_par:] = MF_fixed
        if MF_args == None: MF_args = x[:]
    else:
        merged_par = cov_par[:]
        n_MF_par = 0
        fixed = scipy.zeros(len(cov_par), 'bool')
        if cov_fixed != None: fixed[:] = cov_fixed
    var_par_in = merged_par[fixed == False]
    fixed_par = merged_par[fixed == True]
    args = (x, y, cov_func, cov_typ, MF, n_MF_par, MF_args, fixed, fixed_par, prior)
    var_par_out = \
        sop.fmin(GP_negloglik, var_par_in, args)
    par_out = scipy.copy(merged_par)
    par_out[fixed == False] = var_par_out
    par_out[fixed == True] = fixed_par
    if MF != None:
        return par_out[:-n_MF_par], par_out[-n_MF_par:]
    else:
        return par_out

def GP_train_MCMC(Nstep, x, y, cov_par, cov_scales, cov_func = None, \
                  cov_typ = 'SE', cov_prior = None, \
                  MF = None, MF_par = None, MF_scales = None, MF_args = None, \
                  MF_prior = None):
    '''    
    MCMC over GP hyper-parameters. Calls GP_negloglik. Takes care of
    merging / splitting the fixed / variable and cov / MF parameters
    Returns a Nstep x (M+1) array where M is the number of *variable*
    (scale > 0) hyper-parameters. The first column of the return array
    contains the neg log likelihood values, then the other columns the
    parameters that were varied along the chain.
    '''
    # Sort out the fixed / variable and cov / MF parameters
    if MF != None:
        params = scipy.append(cov_par, MF_par)
        scales = scipy.append(cov_scales, MF_scales)
        n_MF_par = len(MF_par)
        if MF_prior == None:
            if cov_prior == None:
                prior = None
            else:
                prior = numpy.copy(cov_prior)
        else:
            if cov_prior == None:
                prior = numpy.copy(MF_prior)
            else:
                prior = scipy.append(cov_prior, MF_prior)
        if MF_args == None: MF_args = x
    else:
        params = cov_par[:]
        scales = cov_scales[:]
        n_MF_par = 0
        prior = numpy.copy(cov_prior)
    # No do MCMC proper
    fixed = scales == 0
    var = scales > 0
    nvar = var.sum()
    var_par = params[scales > 0]
    var_scales = scales[scales > 0]
    fixed_par = scales[scales == 0]
    chain = scipy.zeros((Nstep, nvar+1)) - 1
    logL = - GP_negloglik(var_par, x, y, covfunc = cov_func, covtyp = cov_typ, \
                          MF = MF, n_MF_par = n_MF_par, \
                          MF_args = MF_args, fixed = fixed, fixed_par = params[fixed])
    randnos = scipy.log(scipy.rand(Nstep))
    for i in range(Nstep):
        shift = pylab.normal(0., 1., nvar) * scales[var]
        var_par_new = var_par + shift
        print var_par_new
        logL_new = \
            - GP_negloglik(var_par_new, x, y, covfunc = cov_func, \
                           covtyp = cov_typ, MF = MF, n_MF_par = n_MF_par, \
                           MF_args = MF_args, fixed = fixed, fixed_par = params[fixed], \
                           prior = prior)
        dlogL = logL_new - logL
        if (randnos[i] <= dlogL):
            print '%8.6f %11.6f %11.6f %8.6f %8.6f %1s' % \
                (i/float(Nstep), logL, logL_new, dlogL, randnos[i], 'A')
            var_par = var_par_new
            logL = logL_new
        else:
            print '%8.6f %11.6f %11.6f %8.6f %8.6f %1s' % \
                (i/float(Nstep), logL, logL_new, dlogL, randnos[i], 'R')
        # Store the new values of the parameters and the new merit function
        chain[i,0] = logL
        chain[i,1:] = scipy.array(var_par)
    return chain

def GP_predict(p, xpred, x, y, cov_func = None, cov_typ = 'SE', MF = None, n_MF_par = 0, \
               MF_args = None, MF_args_pred = None, \
               WhiteNoise = False, ReturnCov = False):
    '''
    Compute predictive distribution for GP with hyper-parameters p,
    conditioned on (x,y) at test inputs xpred, with (optional) mean
    function MF. The parameters of both covariance and mean function
    should be in the p array, MF pars last. The number of MF pars should
    be given by n_MF_par.
    '''
    if MF == None:
        r = y[:]
    else:
        if MF_args == None:
            MF_args = x
        r = y - scipy.matrix([MF(p[-n_MF_par:], MF_args)]).T
    if cov_func == None:
        K = GP_covmat(x, x, p[:-n_MF_par-1], typ = cov_typ, sigma = p[-n_MF_par-1])
        Ks = GP_covmat(xpred, x, p[:-n_MF_par-1], typ = cov_typ)
        if WhiteNoise == True:
            Kss = GP_covmat(xpred, xpred, p[:-n_MF_par-1], typ = cov_typ, \
                            sigma = p[-n_MF_par-1])
        else:
            Kss = GP_covmat(xpred, xpred, p[:-n_MF_par-1], typ = cov_typ)
    else:
        if n_MF_par == 0:
            covpar_end = len(p)
        else:
            covpar_end = -n_MF_par
        K = cov_func(x, x, p[:covpar_end], white_noise = True)
        Ks = cov_func(xpred, x, p[:covpar_end], white_noise = False)
        Kss = cov_func(xpred, xpred, p[:covpar_end], white_noise = WhiteNoise)
    L = scipy.linalg.cho_factor(K)
    b = scipy.linalg.cho_solve(L, r)
    prec_mean = scipy.array(Ks * scipy.matrix(b)).flatten()
    if MF != None:
        if MF_args_pred == None:
            MF_args_pred = xpred
        prec_mean += MF(p[-n_MF_par:], MF_args_pred)
    b = scipy.linalg.cho_solve(L, Ks.T)
    prec_cov = Kss - Ks * scipy.matrix(b)
    if ReturnCov == True:
        return prec_mean, scipy.array(prec_cov)
    else:
        return prec_mean, \
            scipy.array(scipy.sqrt(numpy.diag(prec_cov))).flatten()

def GP_plotpred(xpred, x, y, cov_par, cov_func = None, cov_typ = 'SE',
             MF = None, MF_par = None, MF_args = None, MF_args_pred = None, \
             WhiteNoise = False, plot_color = None):
    '''
    Wrapper for GP_predict that takes care of merging the
    covariance and mean function parameters, and (optionally) plots
    the predictive distribution (as well as returning it)
    '''
    if MF != None:
        merged_par = scipy.append(cov_par, MF_par)
        n_MF_par = len(MF_par)
    else:
        merged_par = cov_par[:]
        n_MF_par = 0
    fpred, fpred_err = GP_predict(merged_par, xpred, x, y, \
                                  cov_func = cov_func, cov_typ = cov_typ, \
                                  MF = MF, n_MF_par = n_MF_par, \
                                  MF_args = MF_args, MF_args_pred = MF_args_pred, \
                                  WhiteNoise = WhiteNoise)
    xpl = scipy.array(xpred[:,0]).flatten()
    if plot_color != None:
        pylab.fill_between(xpl, fpred + 2 * fpred_err, fpred - 2 * fpred_err, \
                           color = plot_color, alpha = 0.1)
        pylab.fill_between(xpl, fpred + fpred_err, fpred - fpred_err, \
                           color = plot_color, alpha = 0.1)
        pylab.plot(xpl, fpred, '-', color = plot_color)
    return fpred, fpred_err

def transit_test(N = 50):
    # Generate a fake dataset 
    xpred = scipy.r_[-0.12:0.1205:0.0005]
    MF = transit_MF
    MF_par = scipy.array([4.47, 0.0, 0.12, 9.85, \
                          # P, T0, Rp/Rs, a/Rs
                          85.634, 0.0, 0.0, 1.0, 0.0, \
                          # incl, ecc, omega, foot, grad
                          0.4397, 0.3754, 0.1005, -0.0622])
                          # limb-darkening parameters
    cov_typ = 'SE_ARD'
    cov_par = scipy.array([0.0003, 0.01, 0.0001])
    Xpred = scipy.matrix([xpred]).T
    x = scipy.copy(xpred)
    numpy.random.shuffle(x)
    x = scipy.sort(x[:N])
    X = scipy.matrix([x]).T
    K = GP_covmat(X, X, cov_par[:-1], typ = cov_typ, sigma = cov_par[-1])
    y = MF(MF_par, x)
    y += numpy.random.multivariate_normal(scipy.zeros(N), K).flatten()
    Y = scipy.matrix([y]).T
    
    # Alter some parameters away from true values, and make prediction
    # using those
    print 'True values:', MF_par[2], cov_par[1]
    MF_par[2] = 0.1
    cov_par[1] = 0.1
    print 'Altered values:', MF_par[2], cov_par[1]
    pylab.clf()
    pylab.plot(x, y, 'k.')
    GP_plotpred(Xpred, X, Y, cov_par, cov_typ = cov_typ,
             MF = MF, MF_par = MF_par, MF_args = x, MF_args_pred = xpred, \
             WhiteNoise = True, plot_color = 'b')

    # Now try to fit for the altered parameters, holding the others fixed
    cov_fixed = scipy.ones(len(cov_par), 'bool')
    cov_fixed[1] = False
    MF_fixed = scipy.ones(len(MF_par), 'bool')
    MF_fixed[2] = False
    cov_par, MF_par = GP_train(X, Y, cov_par, None, cov_typ, cov_fixed, \
                               MF = MF, MF_par = MF_par, MF_args = x, \
                               MF_fixed = MF_fixed)
    print 'Fitted values:', MF_par[2], cov_par[1]
    
    # and make a prediction using the fitted values
    GP_plotpred(Xpred, X, Y, cov_par, cov_typ = cov_typ,
             MF = MF, MF_par = MF_par, MF_args = x, MF_args_pred = xpred, \
             WhiteNoise = True, plot_color = 'r')
    return

