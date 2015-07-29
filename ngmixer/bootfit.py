from __future__ import print_function
import time
import numpy
import logging
import os

# ngmix
import ngmix
from ngmix import Observation, ObsList, MultiBandObsList, GMixRangeError

# local imports
from .defaults import DEFVAL,LOGGERNAME,NO_ATTEMPT,PSF_FIT_FAILURE,GAL_FIT_FAILURE,LOW_PSF_FLUX
from .fitting import BaseFitter
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

class NGMixBootFitter(BaseFitter):
    """
    Use an ngmix bootstrapper
    """
    def __init__(self,*args,**kw):
        super(NGMixBootFitter,self).__init__(*args,**kw)        
        self['replace_cov'] = self.get('replace_cov',False)
        self['use_logpars'] = self.get('ise_logpars',False)
        self['fit_models'] = self.get('fit_models',list(self['model_pars'].keys()))
        self['min_psf_s2n'] = self.get('min_psf_s2n',-numpy.inf)

        self.made_psf_plots = False
        
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
    
    def __call__(self,mb_obs_list,coadd=False,make_epoch_data=True):
        """
        fit the obs list
        """

        # only fit stuff that is not flagged
        new_mb_obs_list = self._get_good_mb_obs_list(mb_obs_list)
        self.new_mb_obs_list = new_mb_obs_list
        
        mb_obs_list.update_meta_data({'fit_data':self._make_struct(coadd)})
        self.data = mb_obs_list.meta['fit_data']

        if self['make_plots']:
            self.plot_dir = './%d-plots' % new_mb_obs_list.meta['id']
            if not os.path.exists(self.plot_dir):
                os.mkdir(self.plot_dir)
                
        fit_flags = 0
        for model in self['fit_models']:
            log.info('    fitting: %s' % model)            

            model_flags, boot = self._run_boot(model,new_mb_obs_list,coadd)
            fit_flags |= model_flags

            # fill the epoch data
            self._fill_epoch_data(mb_obs_list,boot.mb_obs_list)

            # fill in PSF stats in data rows
            if (model_flags & PSF_FIT_FAILURE) == 0:
                self._do_psf_stats(mb_obs_list,coadd)
            else:
                break
            
            if (model_flags & PSF_FIT_FAILURE) != 0:
                break
        
        self._fill_nimage_used(mb_obs_list,boot.mb_obs_list,coadd)
            
        return fit_flags

    def _fill_epoch_data(self,mb_obs_list,new_mb_obs_list):
        for band,obs_list in enumerate(mb_obs_list):
            for obs in obs_list:
                used = False
                res = None

                if obs.meta['flags'] != 0:
                    obs.update_meta_data({'fit_flags':obs.meta['flags']})
                    continue

                if obs.meta['flags'] == 0 and obs.has_psf() and 'fit_data' not in obs.meta:
                    psf_obs = obs.get_psf()
                    
                    ed = self._make_epoch_struct()
                    ed['npix'] = obs.image.size
                    ed['wsum'] = obs.weight.sum()
                    ed['wmax'] = obs.weight.max()
                    ed['psf_counts'] = psf_obs.image.sum()
                    
                    if 'fitter' in psf_obs.meta:
                        res = obs.get_psf().meta['fitter'].get_result()
                        ed['psf_fit_flags'] = res['flags']
                        
                    if psf_obs.has_gmix():
                        used = True
                        psf_gmix = psf_obs.get_gmix()
                        g1,g2,T = psf_gmix.get_g1g2T()
                        pars = psf_gmix.get_full_pars()

                        ed['psf_fit_g'][0,0] = g1
                        ed['psf_fit_g'][0,1] = g1
                        ed['psf_fit_T'] = T
                        ed['psf_fit_pars'] = pars

                    if obs in new_mb_obs_list[band] and used:
                        obs.update_meta_data({'fit_flags':0})
                    else:
                        obs.update_meta_data({'fit_flags':PSF_FIT_FAILURE})

                    obs.update_meta_data({'fit_data':ed})

    def _do_psf_stats(self,mb_obs_list,coadd):
        if coadd:
            n = Namer('coadd')
        else:
            n = Namer('')
        
        Tsum = 0.0
        g1sum = 0.0
        g2sum = 0.0
        wsum = 0.0
        wrelsum = 0.0
        npix = 0.0
        did_one = False
        did_one_max = False
        for band,obs_list in enumerate(mb_obs_list):
            for obs in obs_list:
                if obs.meta['flags'] == 0 and 'fit_data' in obs.meta and obs.meta['fit_data']['psf_fit_flags'][0] == 0:
                    assert obs.meta['fit_flags'] == 0
                    assert obs.get_psf().has_gmix()
                    if obs.meta['fit_data']['wmax'][0] > 0.0:
                        did_one_max = True
                        npix += obs.meta['fit_data']['npix'][0]
                        wrelsum += obs.meta['fit_data']['wsum'][0]/obs.meta['fit_data']['wmax'][0]

                    did_one = True
                    wsum += obs.meta['fit_data']['wsum'][0]
                    Tsum += obs.meta['fit_data']['wsum'][0]*obs.meta['fit_data']['psf_fit_T'][0]
                    g1sum += obs.meta['fit_data']['wsum'][0]*obs.meta['fit_data']['psf_fit_g'][0,0]
                    g2sum += obs.meta['fit_data']['wsum'][0]*obs.meta['fit_data']['psf_fit_g'][0,1]

        if did_one_max:
            self.data[n('mask_frac')][0] = 1.0 - wrelsum/npix

        if did_one:
            self.data[n('psfrec_g')][0,0] = g1sum/wsum
            self.data[n('psfrec_g')][0,1] = g2sum/wsum
            self.data[n('psfrec_T')][0] = Tsum/wsum
    
    def _fill_nimage_used(self,mb_obs_list,new_mb_obs_list,coadd):
        nim = numpy.zeros(self['nband'],dtype='i4')
        nim_used = numpy.zeros_like(nim)
        for band in xrange(self['nband']):
            nim[band] = len(mb_obs_list[band])
            nim_used[band] = len(new_mb_obs_list[band])
            
        if coadd:
            n = Namer("coadd")
        else:
            n = Namer("")

        self.data[n('nimage_use')][0,:] = nim_used
    
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
    
    def _run_boot(self, model, mb_obs_list, coadd):
        """
        run a boot strapper
        """
        
        flags=0
        boot=self._get_bootstrapper(model,mb_obs_list)
        self.boot=boot

        if coadd:
            n = Namer('coadd_psf')
        else:
            n = Namer('psf')
        
        try:
            self._fit_psfs(coadd)
            flags |= self._fit_psf_flux(coadd)

            if flags == 0:
                dindex = 0
                s2n = self.data[n('flux')][dindex,:]/self.data[n('flux_err')][dindex,:]
                max_s2n = numpy.nanmax(s2n)
                if max_s2n < self['min_psf_s2n']:
                    flags |= LOW_PSF_FLUX
                
            if flags == 0:
                try:
                    self._fit_galaxy(model,coadd)                    
                    self._copy_galaxy_result(model,coadd)
                    self._print_galaxy_result()
                except (BootGalFailure,GMixRangeError):
                    log.info("    galaxy fitting failed")
                    flags = GAL_FIT_FAILURE

        except BootPSFFailure:
            log.info("    psf fitting failed")
            flags = PSF_FIT_FAILURE

        return flags, boot

    def _fit_psf_flux(self,coadd):
        self.boot.fit_gal_psf_flux()

        res=self.boot.get_psf_flux_result()

        if coadd:
            n = Namer("coadd_psf")
        else:
            n = Namer("psf")
            
        flagsall=0
        for band in xrange(self['nband']):
            flags=res['flags'][band]
            flagsall |= flags

            flux=res['psf_flux'][band]
            flux_err=res['psf_flux_err'][band]

            self.data[n('flags')][0,band] = flags
            self.data[n('flux')][0,band] = flux
            self.data[n('flux_err')][0,band] = flux_err

            log.info("        psf flux(%s): %g +/- %g" % (band,flux,flux_err))

        return flagsall

    def _fit_psfs(self,coadd):
        """
        fit the psf model to every observation's psf image
        """

        log.info('    fitting PSF')
        
        boot=self.boot

        psf_pars = {}
        for k,v in self['psf_pars'].iteritems():
            if k != 'model' and k != 'ntry':
                psf_pars.update({k:v})

        boot.fit_psfs(self['psf_pars']['model'],
                      None,
                      Tguess_key='Tguess',
                      ntry=self['psf_pars']['ntry'],
                      fit_pars=psf_pars)

        if self['make_plots'] and not self.made_psf_plots:
            self.made_psf_plots = True
            for band,obs_list in enumerate(boot.mb_obs_list):
                for obs in obs_list:
                    psf_obs = obs.get_psf()
                    if psf_obs.has_gmix():
                        self._do_psf_plot(psf_obs,obs.meta['id'],band,obs.meta['band_id'],coadd)

    def _do_psf_plot(self,obs,obs_id,band,band_id,coadd):
        """
        make residual plots for psf
        """
        import images

        title='%d band: %s' % (obs_id, band)
        if coadd:
            title='%s coadd' % title
        else:
            title='%s %d' % (title,band_id)

        im=obs.image

        gmix = obs.get_gmix()
        model_im=gmix.make_image(im.shape, jacobian=obs.jacobian)
        modflux=model_im.sum()
        if modflux <= 0:
            log.log("        psf model flux too low: %f" % modflux)
            return

        model_im *= ( im.sum()/modflux )

        ims = 1e3/numpy.max(im)
        plt=images.compare_images(im*ims, model_im*ims,
                                  label1='psf', label2='model',
                                  show=False, nonlinear=0.075)
        plt.title=title

        if coadd:
            fname=os.path.join(self.plot_dir,'%d-psf-resid-band%d-coadd.png' % (obs_id,band))
        else:
            fname=os.path.join(self.plot_dir,'%d-psf-resid-band%d-%d.png' % (obs_id,band,band_id))

        log.info("        making plot %s" % fname)
        plt.write_img(1920,1200,fname)
        
    def _fit_galaxy(self, model, coadd):
        """
        over-ride for different fitters
        """
        raise RuntimeError("over-ride me")

    def _plot_resids(self, obj_id, fitter, model, coadd, fitter_type):
        """
        make plots
        """
        import images

        if coadd:
            ptype='coadd'
        else:
            ptype='mb'

        ptype = '%s-%s-%s' % (ptype,model,fitter_type)
        title='%d %s' % (obj_id,ptype)
        try:
            res_plots = None
            if fitter_type != 'isample':
                res_plots=fitter.plot_residuals(title=title)
            if res_plots is not None:
                for band, band_plots in enumerate(res_plots):
                    for icut, plt in enumerate(band_plots):
                        fname=os.path.join(self.plot_dir,'%d-%s-resid-band%d-im%d.png' % (obj_id,ptype,band,icut))
                        log.info("        making plot %s" % fname)
                        plt.write_img(1920,1200,fname)

        except GMixRangeError as err:
            log.info("        caught error plotting resid: %s" % str(err))

    def _plot_trials(self, obj_id, fitter, model, coadd, fitter_type, wgts):
        """
        make plots
        """        
        if coadd:
            ptype='coadd'
        else:
            ptype='mb'

        ptype = '%s-%s-%s' % (ptype,model,fitter_type)
        title='%d %s' % (obj_id,ptype)
        
        try:
            pdict=fitter.make_plots(title=title,
                                    weights=wgts)
            
            pdict['trials'].aspect_ratio=1.5
            pdict['wtrials'].aspect_ratio=1.5
            
            trials_png=os.path.join(self.plot_dir,'%d-%s-trials.png' % (obj_id,ptype))
            wtrials_png=os.path.join(self.plot_dir,'%d-%s-wtrials.png' % (obj_id,ptype))
            
            log.info("        making plot %s" % trials_png)
            pdict['trials'].write_img(1200,1200,trials_png)
            
            log.info("        making plot %s" % wtrials_png)
            pdict['wtrials'].write_img(1200,1200,wtrials_png)
        except:
            log.info("        caught error plotting trials")

        try:
            from .util import plot_autocorr
            trials=fitter.get_trials()
            plt=plot_autocorr(trials)
            plt.title=title
            fname=os.path.join(self.plot_dir,'%d-%s-autocorr.png' % (obj_id,ptype))
            log.info("        making plot %s" % fname)
            plt.write_img(1000,1000,fname)
        except:
            log.info("        caught error plotting autocorr")

    def _plot_images(self, obj_id, model, coadd):
        import images
        imlist = []
        titles = []
        for band,obs_list in enumerate(self.new_mb_obs_list):
            for obs in obs_list: 
                imlist.append(obs.image*obs.weight)
                titles.append('band: %d %d' % (band,obs.meta['band_id']))
        if coadd:
            coadd_png=os.path.join(self.plot_dir,'%d-coadd-images.png' % (obj_id))
        else:
            coadd_png=os.path.join(self.plot_dir,'%d-mb-images.png' % (obj_id))
        plt=images.view_mosaic(imlist, titles=titles, show=False)
        log.info("        making plot %s" % coadd_png)
        plt.write_img(1200,1200,coadd_png)

    def _copy_galaxy_result(self, model, coadd):
        """
        Copy from the result dict to the output array
        """

        dindex=0

        res=self.gal_fitter.get_result()
        mres=self.boot.get_max_fitter().get_result()

        rres=self.boot.get_round_result()

        if coadd:
            n=Namer('coadd_%s' % model)
        else:
            n=Namer(model)
        data=self.data

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

            for sn in ['s2n_w','chi2per','dof']:
                data[n(sn)][dindex] = res[sn]
            
            if self['do_shear']:
                if 'g_sens' in res:
                    data[n('g_sens')][dindex,:] = res['g_sens']

                if 'R' in res:
                    data[n('P')][dindex] = res['P']
                    data[n('Q')][dindex,:] = res['Q']
                    data[n('R')][dindex,:,:] = res['R']

            for f in ['fracdev','fracdev_noclip','fracdev_err','TdByTe']:
                if f in res:
                    data[n(f)][dindex] = res[f]
                    
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

    def get_num_pars_psf(self):
        npdict = {'em1':6,
                  'em2':6*2,
                  'em3':6*3,
                  'turb':6,
                  'gauss':6}
        model = self['psf_pars']['model'].lower()
        assert model in npdict,"psf model %s not allowed in NGMixBootFitter" % model
        return npdict[model]
    
    def get_epoch_fit_data_dtype(self):
        npars = self.get_num_pars_psf()
        dt=[('npix','i4'),
            ('wsum','f8'),
            ('wmax','f8'),
            ('psf_fit_flags','i4'),
            ('psf_counts','f8'),
            ('psf_fit_g','f8',2),
            ('psf_fit_T','f8'),
            ('psf_fit_pars','f8',npars)]

        return dt

    def _make_epoch_struct(self,num=1):
        dt = self.get_epoch_fit_data_dtype()
        
        epoch_data = numpy.zeros(num, dtype=dt)
        epoch_data['npix'] = DEFVAL
        epoch_data['wsum'] = DEFVAL
        epoch_data['wmax'] = DEFVAL
        epoch_data['psf_fit_flags'] = NO_ATTEMPT
        epoch_data['psf_counts'] = DEFVAL
        epoch_data['psf_fit_g'] = DEFVAL
        epoch_data['psf_fit_T'] = DEFVAL
        epoch_data['psf_fit_pars'] = DEFVAL
        
        return epoch_data
            
    def get_fit_data_dtype(self,me,coadd):
        dt = []
        if me:
            dt += self._get_fit_data_dtype(False)
        if coadd:
            dt += self._get_fit_data_dtype(True)
        return dt
    
    def _get_lnames(self):
        if self['use_logpars']:
            fname='log_flux'
            Tname='log_T'
        else:
            fname='flux'
            Tname='T'

        return fname, Tname

    def _get_all_models(self,coadd):
        """
        get all model names, includeing the coadd_ ones
        """
        self['fit_models'] = self.get('fit_models',list(self['model_pars'].keys()))

        models=[]
        if coadd:
            models = models + ['coadd_%s' % model for model in self['fit_models']]
        else:
            models = models + self['fit_models']    
        
        return models
    
    def _get_fit_data_dtype(self,coadd):
        dt=[]
        
        nband=self['nband']
        bshape=(nband,)
        simple_npars=5+nband

        if coadd:
            n = Namer('coadd_psf')
        else:
            n = Namer('psf')

        dt += [(n('flags'),   'i4',bshape),
               (n('flux'),    'f8',bshape),
               (n('flux_err'),'f8',bshape)]
                
        if coadd:
            n = Namer('coadd')
        else:
            n = Namer('')

        dt += [(n('nimage_use'),'i4',bshape)]
        
        dt += [(n('mask_frac'),'f8'),
               (n('psfrec_T'),'f8'),
               (n('psfrec_g'),'f8', 2)]
        
        if nband==1:
            fcov_shape=(nband,)
        else:
            fcov_shape=(nband,nband)
        
        fname, Tname=self._get_lnames()

        models=self._get_all_models(coadd)
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

            if 'cm' in model:
                dt += [(n('fracdev'),'f4'),
                       (n('fracdev_noclip'),'f4'),
                       (n('fracdev_err'),'f4'),
                       (n('TdByTe'),'f4')]
                
        return dt

    def _make_struct(self,coadd):
        """
        make the output structure
        """
        num = 1        
        dt = self._get_fit_data_dtype(coadd=coadd)
        data = numpy.zeros(num,dtype=dt)

        if coadd:
            n = Namer('coadd_psf')
        else:
            n = Namer('psf')
        
        data[n('flags')] = NO_ATTEMPT
        data[n('flux')] = DEFVAL
        data[n('flux_err')] = DEFVAL

        if coadd:
            n = Nmer('coadd')
        else:
            n = Namer('')
            
        data[n('mask_frac')] = DEFVAL
        data[n('psfrec_T')] = DEFVAL
        data[n('psfrec_g')] = DEFVAL
        
        fname, Tname=self._get_lnames()

        models=self._get_all_models(coadd)
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

            if 'cm' in model:
                data[n('fracdev')] = DEFVAL
                data[n('fracdev_noclip')] = DEFVAL
                data[n('fracdev_err')] = DEFVAL
                data[n('TdByTe')] = DEFVAL

        return data

    def get_default_fit_data(self,me,coadd):                
        dt = self.get_fit_data_dtype(me,coadd)
        d = numpy.zeros(1,dtype=dt)
        if me:
            dme = self._make_struct(False)
            for tag in dme.dtype.names:                
                d[tag] = dme[tag]

        if coadd:
            dcoadd = self._make_struct(True)
            for tag in dcoadd.dtype.names:                
                d[tag] = dcoadd[tag]        

        return d

    def get_default_epoch_fit_data(self):                
        d = self._make_epoch_struct()
        return d
    
class MaxNGMixBootFitter(NGMixBootFitter):
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

    def _fit_galaxy(self, model, coadd):
        self._fit_max(model)

        rpars=self['round_pars']
        self.boot.set_round_s2n(self['max_pars'],fitter_type=rpars['fitter_type'])
        
        """
        if self['make_plots']:
            fitter = self.boot.get_max_fitter()
            self._do_make_plots(fitter, model, fitter_type='lm')
        """
        
        self.gal_fitter=self.boot.get_max_fitter()

        self._plot_resids(self.new_mb_obs_list.meta['id'], self.boot.get_max_fitter(), model, coadd, 'max')
        self._plot_images(self.new_mb_obs_list.meta['id'], model, coadd)
        
class ISampNGMixBootFitter(MaxNGMixBootFitter):
    def _fit_galaxy(self, model, coadd):
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

        self._plot_resids(self.new_mb_obs_list.meta['id'], self.boot.get_max_fitter(), model, coadd, 'max')
        self._plot_images(self.new_mb_obs_list.meta['id'], model, coadd)
        self._plot_trials(self.new_mb_obs_list.meta['id'], self.boot.get_isampler(), model, coadd, 'isample', self.boot.get_isampler().get_iweights())

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

    def _copy_galaxy_result(self, model, coadd):
        super(ISampNGMixBootFitter,self)._copy_galaxy_result(model,coadd)

        dindex=0
        res=self.gal_fitter.get_result()

        if coadd:
            n=Namer('coadd_%s' % model)
        else:
            n=Namer(model)

        if res['flags'] == 0:
            for f in ['efficiency','neff']:
                self.data[n(f)][dindex] = res[f]

    def _get_fit_data_dtype(self,coadd):
        dt=super(ISampNGMixBootFitter,self)._get_fit_data_dtype(coadd)

        for model in self._get_all_models(coadd):
            n=Namer(model)
            dt += [(n('efficiency'),'f4'),
                (n('neff'),'f4')]
            
        return dt

    def _make_struct(self,coadd):
        d = super(ISampNGMixBootFitter,self)._make_struct(coadd)

        for model in self._get_all_models(coadd):
            n=Namer(model)
            d[n('efficiency')] = DEFVAL
            d[n('neff')] = DEFVAL

        return d

