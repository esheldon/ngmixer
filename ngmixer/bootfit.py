from __future__ import print_function
import time
import numpy
import logging

# ngmix
import ngmix
from ngmix import Observation, ObsList, MultiBandObsList, GMixRangeError

# local imports
from .defaults import DEFVAL,LOGGERNAME,NO_ATTEMPT,PSF_FIT_FAILURE,GAL_FIT_FAILURE,LOW_PSF_FLUX
from .fitting import NGMixER
from .util import Namer

# logging
log = logging.getLogger(LOGGERNAME)

# ngmix imports
from ngmix import print_pars
from ngmix.fitting import EIG_NOTFINITE
from ngmix.gexceptions import BootPSFFailure, BootGalFailure

def get_bootstrapper(obs, type='boot', **keys):
    from ngmix.bootstrap import Bootstrapper
    from ngmix.bootstrap import CompositeBootstrapper
    from ngmix.bootstrap import BestBootstrapper

    use_logpars=keys.get('use_logpars',True)

    if type=='boot':
        boot=Bootstrapper(obs,use_logpars=use_logpars)
    elif type=='composite':
        fracdev_prior = keys.get('fracdev_prior',None)
        fracdev_grid  = keys.get('fracdev_grid',None)
        boot=CompositeBootstrapper(obs,
                                   fracdev_prior=fracdev_prior,
                                   fracdev_grid=fracdev_grid,
                                   use_logpars=use_logpars)
    else:
        raise ValueError("bad bootstrapper type: '%s'" % type)

    return boot

class BootNGMixER(NGMixER):
    """
    Use a ngmix bootstrapper
    """
    def __init__(self,*args,**kw):
        super(BootNGMixER,self).__init__(*args,**kw)        
        self['replace_cov'] = self.get('replace_cov',False)
        self['use_logpars'] = self.get('ise_logpars',False)
        self['fit_models'] = self.get('fit_models',list(self['model_pars'].keys()))
        self['min_psf_s2n'] = self.get('min_psf_s2n',-numpy.inf)
        
    def _get_good_mb_obs_list(self,mb_obs_list):
        new_mb_obs_list = MultiBandObsList()
        for obs_list in mb_obs_list:
            new_obs_list = ObsList()
            for obs in obs_list:
                if obs.meta['flags'] == 0:
                    new_obs_list.append(obs)                    
            new_mb_obs_list.append(new_obs_list)
        new_mb_obs_list.update_meta_data(mb_obs_list.meta)
        new_mb_obs_list.update_meta_data({'old_mb_obs_list':mb_obs_list})
        return new_mb_obs_list
    
    def fit_obs_list(self,mb_obs_list,coadd=False):
        """
        fit the obs list
        """

        # onyl fit stuff that is not flagged
        new_mb_obs_list = self._get_good_mb_obs_list(mb_obs_list)
        
        fit_flags = 0
        epoch_data = []
        make_epoch = True
        for model in self['fit_models']:
            log.info('    fitting: %s' % model)            

            model_flags, boot = self._run_boot(model,new_mb_obs_list)
            fit_flags |= model_flags

            if make_epoch:
                make_epoch = False

                # make the epoch data

                # fill in PSF stats

            if model_flags & PSF_FIT_FAILURE:
                break
                
        return fit_flags,epoch_data
                                                 
    def _get_bootstrapper(self, model, mb_obs_list):
        """
        get the bootstrapper for fitting psf through galaxy
        """
        
        if model == 'cm':
            fracdev_prior=self['model_pars']['cm']['fracdev_prior']
            boot=get_bootstrapper(mb_obs_list,
                                  type='composite',
                                  fracdev_prior=fracdev_prior,
                                  **self)
        else:
            boot=get_bootstrapper(mb_obs_list, **self)
        
        return boot
    
    def _run_boot(self, model, mb_obs_list):
        """
        run a boot strapper
        """
        
        flags=0
        boot=self._get_bootstrapper(model,mb_obs_list)
        self.boot=boot

        try:
            self._fit_psfs()
            flags |= self._fit_psf_flux()

            if flags == 0:
                dindex = self.curr_data_index
                s2n = self.curr_data['psf_flux'][dindex,:]/self.curr_data['psf_flux_err'][dindex,:]
                max_s2n = numpy.nanmax(s2n)
                if max_s2n < self['min_psf_s2n']:
                    flags |= LOW_PSF_FLUX
                
            if flags == 0:
                try:
                    self._fit_galaxy(model)
                    self._copy_galaxy_result(model)
                    self._print_galaxy_result()
                except (BootGalFailure,GMixRangeError):
                    log.info("    galaxy fitting failed")
                    flags = GAL_FIT_FAILURE

        except BootPSFFailure:
            log.info("    psf fitting failed")
            flags = PSF_FIT_FAILURE

        return flags, boot

    def _fit_psf_flux(self):
        self.boot.fit_gal_psf_flux()

        res=self.boot.get_psf_flux_result()

        dindex = self.curr_data_index
        n = Namer("psf")
        data = self.curr_data
        
        flagsall=0
        for band in xrange(self['nband']):
            flags=res['flags'][band]
            flagsall |= flags

            flux=res['psf_flux'][band]
            flux_err=res['psf_flux_err'][band]

            data[n('flags')][dindex,band] = flags
            data[n('flux')][dindex,band] = flux
            data[n('flux_err')][dindex,band] = flux_err

            log.info("        psf flux(%s): %g +/- %g" % (band,flux,flux_err))

        return flagsall

    def _fit_psfs(self):
        """
        fit the psf model to every observation's psf image
        """

        boot=self.boot

        psf_pars=self['psf_em_pars']

        Tguess = 0.0
        Nguess = 0.0

        for band,obs_list in enumerate(boot.mb_obs_list_orig):
            for obs in obs_list:
                if obs.meta['flags'] == 0:
                    if 'sigma_sky' in obs.get_psf().meta:
                        Tguess += obs.get_psf().meta['sigma_sky']**2
                        Nguess += 1.0
        Tguess /= Nguess
        if Tguess <= 0.0:
            Tguess = 10.0
        
        boot.fit_psfs(psf_pars['model'],
                      Tguess=Tguess,
                      ntry=psf_pars['ntry'])

    def _fit_galaxy(self, model):
        """
        over-ride for different fitters
        """
        raise RuntimeError("over-ride me")

 
    def _fit_max(self, model):
        """
        do a maximum likelihood fit

        note prior applied during
        """
        boot=self.boot

        max_pars=self['max_pars']
        cov_pars=self['cov_pars']
        prior=self['model_pars'][model]['prior']

        # now with prior
        boot.fit_max(model,
                     max_pars,
                     prior=prior,
                     ntry=max_pars['ntry'])

        if self['replace_cov']:
            log.info("        replacing cov")
            boot.try_replace_cov(cov_pars)
        
    def _copy_galaxy_result(self, model):
        """
        Copy from the result dict to the output array
        """

        dindex=self.curr_data_index

        res=self.gal_fitter.get_result()
        mres=self.boot.get_max_fitter().get_result()

        rres=self.boot.get_round_result()

        n=Namer(model)
        data=self.curr_data

        data[n('flags')][dindex] = res['flags']

        fname, Tname = self._get_lnames()

        if res['flags'] == 0:
            pars=res['pars']
            pars_cov=res['pars_cov']

            flux=pars[5:]
            flux_cov=pars_cov[5:, 5:]
            
            data[n('max_flags')][dindex] = mres['flags']
            data[n('max_pars')][dindex,:] = mres['pars']
            data[n('max_pars_cov')][dindex,:,:] = mres['pars_cov']

            data[n('pars')][dindex,:] = pars
            data[n('pars_cov')][dindex,:,:] = pars_cov

            data[n('g')][dindex,:] = res['g']
            data[n('g_cov')][dindex,:,:] = res['g_cov']

            data[n('flags_r')][dindex]  = rres['flags']
            data[n('s2n_r')][dindex]    = rres['s2n_r']
            data[n(Tname+'_r')][dindex] = rres['pars'][4]
            data[n('psf_T_r')][dindex]  = rres['psf_T_r']

            if self['do_shear']:
                if 'g_sens' in res:
                    data[n('g_sens')][dindex,:] = res['g_sens']

                if 'R' in res:
                    data[n('P')][dindex] = res['P']
                    data[n('Q')][dindex,:] = res['Q']
                    data[n('R')][dindex,:,:] = res['R']

    def _print_galaxy_result(self):
        res=self.gal_fitter.get_result()
        if 'pars' in res:
            print_pars(res['pars'],    front='    gal_pars: ')
        if 'pars_err' in res:
            print_pars(res['pars_err'],front='    gal_perr: ')

        mres=self.boot.get_max_fitter().get_result()
        if 's2n_w' in mres:
            rres=self.boot.get_round_result()
            tup=(mres['s2n_w'],rres['s2n_r'],mres['chi2per'])
            log.info("    s2n: %.1f s2n_r: %.1f chi2per: %.3f" % tup)
    
    def _get_lnames(self):
        if self['use_logpars']:
            fname='log_flux'
            Tname='log_T'
        else:
            fname='flux'
            Tname='T'

        return fname, Tname

    def _get_all_models(self):
        """
        get all model names, includeing the coadd_ ones
        """
        self['fit_models'] = self.get('fit_models',list(self['model_pars'].keys()))
        
        models=[]

        if self['fit_coadd_galaxy']:
            models = models + ['coadd_%s' % model for model in self['fit_models']]

        if self['fit_me_galaxy']:
            models = models + self['fit_models']
                                        
        return models
    
    def _get_dtype(self):
        dt=super(BootNGMixER,self)._get_dtype()
        
        nband=self['nband']
        bshape=(nband,)
        simple_npars=5+nband

        dt += [('nimage_tot','i4',bshape),
               ('nimage_use','i4',bshape),
               ('mask_frac','f8'),
               ('psfrec_T','f8'),
               ('psfrec_g','f8', 2)]
        
        names = []
        if self['fit_me_galaxy']:
            names.append('psf')
        if self['fit_coadd_galaxy']:
            names.append('coadd_psf')
        for name in names:
            n=Namer(name)
            dt += [(n('flags'),   'i4',bshape),
                   (n('flux'),    'f8',bshape),
                   (n('flux_err'),'f8',bshape)]
        
        if nband==1:
            fcov_shape=(nband,)
        else:
            fcov_shape=(nband,nband)
        
        fname, Tname=self._get_lnames()

        models=self._get_all_models()
        for model in models:
            n=Namer(model)
            np=simple_npars
            
            dt+=[(n('flags'),'i4'),
                 (n('pars'),'f8',np),
                 (n('pars_cov'),'f8',(np,np)),
                 (n('g'),'f8',2),
                 (n('g_cov'),'f8',(2,2)),

                 (n('max_flags'),'i4'),
                 (n('max_pars'),'f8',np),
                 (n('max_pars_cov'),'f8',(np,np)),

                 (n('s2n_w'),'f8'),
                 (n('chi2per'),'f8'),
                 (n('dof'),'f8'),

                 (n('flags_r'),'i4'),
                 (n('s2n_r'),'f8'),
                 (n(Tname+'_r'),'f8'),
                 (n('psf_T_r'),'f8')]
            
            if self['do_shear']:
                dt += [(n('g_sens'), 'f8', 2)]
            
        return dt

    def _make_struct(self,num=1):
        """
        make the output structure
        """
        data = super(BootNGMixER,self)._make_struct(num)
        
        data['mask_frac'] = DEFVAL
        data['psfrec_T'] = DEFVAL
        data['psfrec_g'] = DEFVAL

        names = []
        if self['fit_me_galaxy']:
            names.append('psf')
        if self['fit_coadd_galaxy']:
            names.append('coadd_psf')
        for name in names:
            n=Namer(name)
            data[n('flags')] = NO_ATTEMPT
            data[n('flux')] = DEFVAL
            data[n('flux_err')] = DEFVAL

        fname, Tname=self._get_lnames()

        models=self._get_all_models()
        for model in models:
            n=Namer(model)

            data[n('flags')] = NO_ATTEMPT
            
            data[n('pars')] = DEFVAL
            data[n('pars_cov')] = DEFVAL

            data[n('g')] = DEFVAL
            data[n('g_cov')] = DEFVAL

            data[n('s2n_w')] = DEFVAL
            data[n('chi2per')] = DEFVAL
            
            data[n('max_flags')] = NO_ATTEMPT
            data[n('max_pars')] = DEFVAL
            data[n('max_pars_cov')] = DEFVAL

            data[n('flags_r')] = NO_ATTEMPT
            data[n('s2n_r')] = DEFVAL
            data[n(Tname+'_r')] = DEFVAL
            data[n('psf_T_r')] = DEFVAL
            
            if self['do_shear']:
                data[n('g_sens')] = DEFVAL

        return data

class MaxBootNGMixER(BootNGMixER):
    def _fit_galaxy(self, model):
        self._fit_max(model)

        rpars=self['round_pars']
        self.boot.set_round_s2n(self['max_pars'],fitter_type=rpars['fitter_type'])
        
        """
        if self['make_plots']:
            fitter = self.boot.get_max_fitter()
            self._do_make_plots(fitter, model, fitter_type='lm')
        """
        
        self.gal_fitter=self.boot.get_max_fitter()

class ISampleBootNGMixER(BootNGMixER):
    def _fit_galaxy(self, model):
        self._fit_max(model)

        """
        if self['make_plots']:
            fitter = self.boot.get_max_fitter()
            self._do_make_plots(fitter, model, fitter_type='lm')
        """
        
        self._do_isample(model)

        """
        if self['make_plots']:
            fitter = self.boot.get_isampler()
            self._do_make_plots(fitter, model, fitter_type='isample')
        """
        
        self._add_shear_info(model)

        self.gal_fitter=self.boot.get_isampler()

    def _do_isample(self, model):
        """
        run isample on the bootstrapper
        """
        ipars=self['isample_pars']
        prior=self['model_pars'][model]['prior']
        self.boot.isample(ipars, prior=prior)

        rpars=self['round_pars']
        self.boot.set_round_s2n(self['max_pars'],fitter_type=rpars['fitter_type'])

    def _add_shear_info(self, model):
        """
        add shear information based on the gal_fitter
        """

        boot=self.boot
        max_fitter=boot.get_max_fitter()
        sampler=boot.get_isampler()

        # this is the full prior
        prior=self['model_pars'][model]['prior']
        g_prior=prior.g_prior

        iweights = sampler.get_iweights()
        samples = sampler.get_samples()
        g_vals=samples[:,2:2+2]

        res=sampler.get_result()

        # keep for later if we want to make plots
        self.weights=iweights

        # we are going to mutate the result dict owned by the sampler
        stats = max_fitter.get_fit_stats(res['pars'])
        res.update(stats)

        ls=ngmix.lensfit.LensfitSensitivity(g_vals,
                                            g_prior,
                                            weights=iweights,
                                            remove_prior=True)
        g_sens = ls.get_g_sens()
        g_mean = ls.get_g_mean()

        res['g_sens'] = g_sens
        res['nuse'] = ls.get_nuse()

    def _copy_galaxy_result(self, model):
        super(ISampleBootNGMixER,self)._copy_galaxy_result(model)

        res=self.gal_fitter.get_result()
        if res['flags'] == 0:

            dindex=self.curr_data_index
            res=self.gal_fitter.get_result()
            n=Namer(model)

            for f in ['efficiency','neff']:
                self.curr_data[n(f)][dindex] = res[f]

    def _get_dtype(self):
        dt=super(ISampleBootNGMixER,self)._get_dtype()

        for model in self._get_all_models():
            n=Namer(model)
            dt += [(n('efficiency'),'f4'),
                (n('neff'),'f4')]
            
        return dt

    def _make_struct(self,num=1):
        d = super(ISampleBootNGMixER,self)._make_struct(num)

        for model in self._get_all_models():
            n=Namer(model)
            d[n('efficiency')] = DEFVAL
            d[n('neff')] = DEFVAL

        return d

class CompositeISampleBootNGMixER(ISampleBootNGMixER):
    def _copy_galaxy_result(self, model):
        super(CompositeISampleBootNGMixER,self)._copy_galaxy_result(model)

        res=self.gal_fitter.get_result()
        if res['flags'] == 0:

            dindex=self.curr_data_index
            res=self.gal_fitter.get_result()
            n=Namer(model)

            for f in ['fracdev','fracdev_noclip','fracdev_err','TdByTe']:
                self.curr_data[n(f)][dindex] = res[f]

    def _get_dtype(self):
        dt=super(CompositeISampleBootNGMixER,self)._get_dtype()

        n=Namer('cm')
        dt += [(n('fracdev'),'f4'),
               (n('fracdev_noclip'),'f4'),
               (n('fracdev_err'),'f4'),
               (n('TdByTe'),'f4')]
        
        return dt

    def _make_struct(self,num=1):
        d = super(CompositeISampleBootNGMixER,self)._make_struct(num)
        n=Namer('cm')
        d[n('fracdev')] = DEFVAL
        d[n('fracdev_noclip')] = DEFVAL
        d[n('fracdev_err')] = DEFVAL
        d[n('TdByTe')] = DEFVAL

        return d

