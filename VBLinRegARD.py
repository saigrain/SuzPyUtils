def meanvar(D):
   m = scipy.mean(D, axis = 0)
   s = scipy.shape(D)
   if len(s) == 1:
      c = D.var()
      if c == 0: c = 1.0
   elif (s[0] == 1) + (s[1] == 1):
      c = D.var()
      if c == 0: c = 1.0
   else:
      c = scipy.diag(scipy.cov(D, rowvar = False))
      c[c == 0] = 1
   return m, c

def normalis(X, D):
    m, c = meanvar(D)
    return (X - m) / scipy.sqrt(c)

def unnorm(X, D):
    m, c = meanvar(D)
    Y = scipy.multiply(X, scipy.ones([scipy.shape(X)[0], 1]) * scipy.sqrt(c))
    Y = Y + scipy.dot(scipy.ones([scipy.shape(X)[0], 1]), m)
    return Y

def logdet(a):
    if scipy.allclose(a.T,a) == False:
        print 'MATRIX NOT SYMMETRIC'
    # Second make sure that matrix is positive definite:
    eigenvalues = scipy.linalg.eigvalsh(a)
    if min(eigenvalues) <=0:
        print 'Matrix is NOT positive-definite'
        print '   min eigv = %.16f' % min(eigenvalues)        
    step1 = scipy.linalg.cholesky(a)
    step2 = scipy.diag(step1.T)
    out = 2. * scipy.sum(scipy.log(step2), axis=0)
    return out

def bayes_linear_fit_ard(X, y):
    # uninformative priors (TME: Need to ask Steve about the reasoning behind this stuff.)
    a0 = 1e-2
    b0 = 1e-4
    c0 = 1e-2
    d0 = 1e-4
    # pre-process data
    [N, D] = scipy.shape(X)
    X_corr = X.T * X
    Xy_corr = X.T * y
    an = a0 + N / 2.    
    gammaln_an = scipy.special.gammaln(an)
    cn = c0 + 1 / 2.    
    D_gammaln_cn = D * scipy.special.gammaln(cn)
    # iterate to find hyperparameters
    L_last = -sys.float_info.max
    max_iter = 500
    E_a = scipy.matrix(scipy.ones(D) * c0 / d0).T
    for iter in range(max_iter):
        # covariance and weight of linear model
        invV = scipy.matrix(scipy.diag(scipy.array(E_a)[:,0])) + X_corr   
        V = scipy.matrix(scipy.linalg.inv(invV))
        logdetV = -logdet(invV)    
        w = scipy.dot(V, Xy_corr)[:,0]
        # parameters of noise model (an remains constant)
        sse = scipy.sum(scipy.power(X*w-y, 2), axis=0)
        if scipy.imag(sse)==0:
            sse = scipy.real(sse)[0]
        else:
            print 'Something went wrong'
            pdb.set_trace()
        bn = b0 + 0.5 * (sse + scipy.sum((scipy.array(w)[:,0]**2) * scipy.array(E_a)[:,0], axis=0))
        E_t = an / bn 
        # hyperparameters of covariance prior (cn remains constant)
        dn = d0 + 0.5 * (E_t * (scipy.array(w)[:,0]**2) + scipy.diag(V))
        E_a = scipy.matrix(cn / dn).T
        # variational bound, ignoring constant terms for now
        L = -0.5 * (E_t*sse + scipy.sum(scipy.multiply(X,X*V))) + \
            0.5 * logdetV - b0 * E_t + gammaln_an - an * scipy.log(bn) + an + \
            D_gammaln_cn - cn * scipy.sum(scipy.log(dn))
        # variational bound must grow!
        if L_last > L:
            # if this happens, then something has gone wrong....
            file = open('ERROR_LOG','w')
            file.write('Last bound %6.6f, current bound %6.6f' % (L, L_last))
            file.close()
            raise Exception('Variational bound should not reduce - see ERROR_LOG')
            return
        # stop if change in variation bound is < 0.001%
        if abs(L_last - L) < abs(0.00001 * L):        
            break
        # print L, L_last
        L_last = L
    if iter == max_iter:    
        warnings.warn('Bayes:maxIter ... Bayesian linear regression reached maximum number of iterations.') 
    # augment variational bound with constant terms
    L = L - 0.5 * (N * scipy.log(2 * scipy.pi) - D) - scipy.special.gammaln(a0) + \
        a0 * scipy.log(b0) + D * (-scipy.special.gammaln(c0) + c0 * scipy.log(d0))
    return w, V, invV, logdetV, an, bn, E_a, L

def VB_linreg_ard(phi_tr, flux_tr, phi_new):
    ntr = scipy.shape(phi_tr)[0]
    nbasis = scipy.shape(phi_tr)[1]  
    nnew = scipy.shape(phi_new)[0]
    xn = normalis(phi_tr, phi_tr)
    xn = scipy.concatenate([xn, scipy.ones([ntr,1])], axis=1) # add in the bias term
    t = normalis(flux_tr, flux_tr)
    [w, v, invv, logdetv, an, bn, e_a, l] = bayes_linear_fit_ard(xn, t)
    phi_new_n = normalis(phi_new, phi_tr)
    phi_new_n = scipy.concatenate([phi_new_n, scipy.ones([nnew,1])], axis=1) #add in the bias term
    yp = phi_new_n * w # expectation of y
    yp = unnorm(yp, flux_tr)
    e_tau = an / bn # expectation of noise precision
    s2 = 1. / e_tau
    wu = scipy.matrix(scipy.zeros([scipy.shape(phi_new_n)[0],1]))
    for n in range(scipy.shape(phi_new_n)[0]):    
        wu[n, :] = phi_new_n[n,:] * v * phi_new_n[n,:].T
    sd = scipy.sqrt(s2 + wu)
    sd = scipy.multiply(sd, scipy.matrix(scipy.tile(scipy.std(flux_tr,axis=0),[scipy.shape(yp)[0],1])))
    return yp, sd, w
