#!/usr/bin/env python
import os
import ngmix
import fitsio

def set_priors(conf):
    """
    Sets priors for each model.
    Currently only separable priors can be set.
    """
    model_pars=conf['model_pars']
    
    for model,params in model_pars.iteritems():
        set_cen_prior(params)
        set_g_prior(params)
        set_T_prior(params)
        
        counts_prior_repeat=params.get('counts_prior_repeat',False)
        if counts_prior_repeat:
            print("repeating counts prior for model '%s'" % model)
        set_counts_prior(params, repeat=counts_prior_repeat)

        if 'fracdev_prior_file' in params:
            set_fracdev_prior(params)

def unpack_priors(conf,nband):    
    from ngmix.joint_prior import PriorSimpleSep
    from ngmix.priors import ZDisk2D

    g_prior_flat=ZDisk2D(1.0)

    model_pars=conf['model_pars']
    
    for model, params in model_pars.iteritems():
        print("loading prior for:",model)

        counts_prior_repeat=params.get('counts_prior_repeat',False)
        cp = params['counts_prior']
        if counts_prior_repeat:
            cp = [cp]*nband
        else:
            mess=("counts prior must be length "
                  "%d, bot %d" % (nband,len(cp)) )
            assert len(cp)==nband,mess
            
        print("    full")
        prior = PriorSimpleSep(params['cen_prior'],
                               params['g_prior'],
                               params['T_prior'],
                               cp)
        
        # for the exploration, for which we do not apply g prior during
        print("    gflat")
        gflat_prior = PriorSimpleSep(params['cen_prior'],
                                     g_prior_flat,
                                     params['T_prior'],
                                     cp)
        
        params['prior'] = prior
        params['gflat_prior'] = gflat_prior

def set_T_prior(params):
    typ=params['T_prior_type']
    if typ == 'flat':
        pars=params['T_prior_pars']
        params['T_prior']=ngmix.priors.FlatPrior(pars[0], pars[1])
    elif typ=='TwoSidedErf':
        pars=params['T_prior_pars']
        params['T_prior']=ngmix.priors.TwoSidedErf(*pars)        
    elif typ =='lognormal':
        pars=params['T_prior_pars']
        params['T_prior']=ngmix.priors.LogNormal(pars[0], pars[1])
    elif typ=="cosmos_exp":
        params['T_prior']=ngmix.priors.TPriorCosmosExp()
    elif typ=="cosmos_dev":
        params['T_prior']=ngmix.priors.TPriorCosmosDev()
    else:
        raise ValueError("bad T prior type: %s" % T_prior_type)

def set_counts_prior(params, repeat=False):
    typ=params['counts_prior_type']
    pars=params['counts_prior_pars']
    
    if typ == 'flat':
        pclass = ngmix.priors.FlatPrior
    elif typ=='TwoSidedErf':
        pclass = ngmix.priors.TwoSidedErf
    else:
        raise ValueError("bad counts prior type: %s" % typ)
    
    if repeat:
        # we assume this is one that will be repeated
        params['counts_prior']=pclass(*pars)
    else:
        # assume this is a list of lists
        plist=[]
        for tpars in pars:
            cp = pclass(*tpars)
            plist.append(cp)
        params['counts_prior'] = plist
            
def set_fracdev_prior(params):
    fname=os.path.expanduser( params['fracdev_prior_file'] )
    fname=os.path.expandvars( fname )
    print("reading fracdev_prior:",fname)
    data = fitsio.read(fname)
    
    weights=data['weights']
    means=data['means']
    covars=data['covars']
    
    if len(means.shape) == 1:
        means = means.reshape( (means.size, 1) )
        
    prior = ngmix.gmix.GMixND(weights,
                              means,
                              covars)

    params['fracdev_prior'] = prior
    
def set_g_prior(params):
    typ=params['g_prior_type']

    if typ =='exp':
        parr=numpy.array(params['g_prior_pars'],dtype='f8')
        g_prior = ngmix.priors.GPriorM(parr)
    elif typ=='cosmos-sersic':
        g_prior = ngmix.priors.make_gprior_cosmos_sersic(type='erf')
    elif typ=='cosmos-exp':
        g_prior = ngmix.priors.make_gprior_cosmos_exp()
    elif typ=='cosmos-dev':
        g_prior = ngmix.priors.make_gprior_cosmos_dev()
    elif typ =='ba':
        sigma=params['g_prior_pars']
        g_prior = ngmix.priors.GPriorBA(sigma)
    elif typ=='flat':
        g_prior=ngmix.priors.ZDisk2D(1.0)
    else:
        raise ValueError("implement gprior '%s'")
    params['g_prior']=g_prior
    
def set_cen_prior(params):
    width=params['cen_prior_pars'][0]
    p=ngmix.priors.CenPrior(0.0, 0.0, width, width)
    params['cen_prior'] = p