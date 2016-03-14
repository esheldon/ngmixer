from __future__ import print_function
import time
import numpy
import os
import scipy.stats

from copy import deepcopy

# local imports
from .defaults import DEFVAL, NO_ATTEMPT, \
    PSF_FIT_FAILURE, GAL_FIT_FAILURE, \
    LOW_PSF_FLUX, PSF_FLUX_FIT_FAILURE
from .fitting import BaseFitter
from .util import Namer, print_pars

# ngmix imports
import ngmix
from ngmix import Observation, ObsList, MultiBandObsList, GMixRangeError
from ngmix.fitting import EIG_NOTFINITE
from ngmix.gexceptions import BootPSFFailure, BootGalFailure
from ngmix.gmix import GMixModel, GMix, GMixCM
from ngmix.shape import Shape

from pprint import pprint

def get_bootstrapper(obs, type='boot', **keys):
    from ngmix.bootstrap import Bootstrapper
    from ngmix.bootstrap import CompositeBootstrapper

    if type=='boot':
        boot=Bootstrapper(obs, **keys)
    elif type=='composite':
        boot=CompositeBootstrapper(obs, **keys)
    else:
        raise ValueError("bad bootstrapper type: '%s'" % type)

    return boot

class NGMixBootFitter(BaseFitter):
    """
    Use an ngmix bootstrapper
    """
    def __init__(self,*args,**kw):
        super(NGMixBootFitter,self).__init__(*args,**kw)

    def _setup(self):
        """
        ngmix-specific setups for all ngmix fitters
        """
        # LM doesn't calculate a very good covariance matrix
        self['replace_cov'] = self.get('replace_cov',False)

        # in the fitters use log(flux) and log(T)
        self['use_logpars'] = self.get('use_logpars',False)

        # which models to fit
        self['fit_models'] = self.get('fit_models',list(self['model_pars'].keys()))

        # allow pre-selection based on psf flux
        self['min_psf_s2n'] = self.get('min_psf_s2n',-numpy.inf)

        # find the center and reset jacobians before doing model fits
        self['pre_find_center'] = self.get('pre_find_center',False)

        # do we normalize the psf to unity when doing the PSF mags?
        self['normalize_psf'] = self.get('normalize_psf',True)

        # how to mask unmodeled nbrs
        self['unmodeled_nbrs_masking_type'] = self.get('model_nrbs_unmodmask','nbrs-seg')

        # whether to use the coadd_ prefix for names when no me fit was done
        self['use_coadd_prefix'] = self.get('use_coadd_prefix',True)

        # add in estimate for intrinsic profile variance 
        self['intr_prof_var_fac'] = self.get('intr_prof_var_fac',0.0)

    def _get_namer(self, model, coadd):
        if coadd and (self['fit_me_galaxy'] or self['use_coadd_prefix']):
            n = Namer('coadd_%s' % model)
        else:
            n = Namer(model)

        return n

    def get_models_for_checking(self):
        models = [modl for modl in self['fit_models']]
        pars = [modl+'_max_pars' for modl in self['fit_models']]
        covs = [modl+'_max_pars_cov' for modl in self['fit_models']]

        coadd_models = ['coadd_'+modl for modl in models]
        coadd_pars = ['coadd_'+modl for modl in pars]
        coadd_covs = ['coadd_'+modl for modl in covs]

        return models,pars,covs,coadd_models,coadd_pars,coadd_covs,5+self['nband']

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

    def _add_extra_sim_noise(self, mb_obs_list):
        from numpy.random import normal

        target_noise=self['target_noise']
        target_var = target_noise**2
        target_ivar = 1.0/target_var

        print("    target noise:",target_noise)

        for obslist in mb_obs_list:
            for obs in obslist:

                obs.image_orig=obs.image.copy()
                obs.weight_orig=obs.weight.copy()

                wt=obs.weight
                w=numpy.where(wt > 0)
                if w[0].size > 0:

                    im = obs.image

                    extra_var_values = numpy.zeros(im.shape)
                    orig_var  = numpy.zeros(im.shape)
                    orig_var[w] = 1.0/wt[w]

                    wbad=numpy.where(orig_var > target_var)
                    if wbad[0].size > 0:
                        raise ValueError("attempting to set target var "
                                         "smaller than existing var")

                    # is zero where weight <= 0
                    extra_var_values[w] = target_var - orig_var[w]
                    extra_noise_values = numpy.sqrt(extra_var_values)

                    noise_image = normal(loc=0.0, scale=1.0, size=im.shape)
                    noise_image *= extra_noise_values
                    im += noise_image
                
                    wt[w] = target_ivar

                    obs.image = im
                    obs.weight = wt


    def __call__(self,mb_obs_list,coadd=False,make_epoch_data=True,nbrs_fit_data=None):
        """
        fit the obs list
        """

        if 'target_noise' in self:
            self._add_extra_sim_noise(mb_obs_list)

        # only fit stuff that is not flagged
        new_mb_obs_list = self._get_good_mb_obs_list(mb_obs_list)
        self.new_mb_obs_list = new_mb_obs_list
        self.mb_obs_list = mb_obs_list
        
        # FIXME - removed if, this might have sid effects?
        #if 'fit_data' not in mb_obs_list.meta:
        mb_obs_list.update_meta_data({'fit_data':self._make_struct(coadd)})
        self.data = mb_obs_list.meta['fit_data']

        if self['make_plots']:
            self.plot_dir = './%d-plots' % new_mb_obs_list.meta['id']
            if not os.path.exists(self.plot_dir):
                os.makedirs(self.plot_dir)

        fit_flags = 0
        for model in self['fit_models']:
            print('    fitting: %s' % model)

            if self['model_nbrs'] and nbrs_fit_data is not None:
                self._render_nbrs(model,new_mb_obs_list,coadd,nbrs_fit_data)

            model_flags, boot = self._guess_and_run_boot(model,
                                                         new_mb_obs_list,
                                                         coadd,
                                                         nbrs_fit_data=nbrs_fit_data)

            fit_flags |= model_flags

            # fill the epoch data
            self._fill_epoch_data(mb_obs_list,boot.mb_obs_list)

            # fill in PSF stats in data rows
            if (model_flags & PSF_FIT_FAILURE) == 0:
                self._do_psf_stats(mb_obs_list,coadd)
            else:
                break

        self._fill_nimage_used(mb_obs_list,boot.mb_obs_list,coadd)

        return fit_flags

    def _guess_and_run_boot(self,model,new_mb_obs_list,coadd,nbrs_fit_data=None):
        n=self._get_namer(model, coadd)

        guess = None
        guess_errs = None
        guess_TdbyTe = 1.0
 
        if nbrs_fit_data is not None:

            ind = new_mb_obs_list.meta['cen_ind']

            if (nbrs_fit_data[n('flags')][ind] == 0
                and nbrs_fit_data['flags'][ind] == 0):

                guess = nbrs_fit_data[n('pars')][ind]
                
                # lots of pain to get good guesses...
                # the ngmix ParsGuesser does this
                #    for pars 0 through 3 inclusive - uniform between -width to +width
                #    for pars 4 through the end - guess = pars*(1+width*uniform(low=-1,high=1))
                # thus for pars 4 through the end, I divide the error by the pars so that guess is
                #  between 1-frac_err to 1+frac_err where frac_err = err/pars
                # I also scale the errors by scale
                scale = 0.5

                # get the errors (cov in this case)
                guess_errs = numpy.diag(nbrs_fit_data[n('max_pars_cov')][ind]).copy()

                #if less than zero, set to zero
                w, = numpy.where(guess_errs < 0.0)
                if w.size > 0:
                    guess_errs[w[:]] = 0.0
                    
                # take sqrt
                guess_errs = numpy.sqrt(guess_errs)*scale
                    
                # get pars to scale by
                # don't divide by zero! - if zero set to 0.1 (default val in ngmix)
                w, = numpy.where(guess == 0.0)
                guess_scale = guess.copy()
                if w.size > 0:
                    guess_scale[w] = 0.1
                w = numpy.arange(4,guess.size,1)

                # final equation - need sqrt then apply scale and then divide by pars
                guess_errs[w[:]] = guess_errs[w]/numpy.abs(guess_scale[w])
                
                # don't guess to wide for the shear
                if guess_errs[2] > 0.1:
                    guess_errs[2] = 0.1
                
                if guess_errs[3] > 0.1:
                    guess_errs[3] = 0.1            
                    
                print_pars(guess,front='    guess pars:  ')
                print_pars(guess_errs,front='    guess errs:  ')
                
                if model == 'cm':
                    guess_TdbyTe = nbrs_fit_data[n('TdByTe')][ind]
            
        if model == 'cm':
            return self._run_boot(model,new_mb_obs_list,coadd,
                                  guess_TdbyTe=guess_TdbyTe,
                                  guess=guess,                                  
                                  guess_widths=guess_errs)
        else:
            return self._run_boot(model,new_mb_obs_list,coadd,
                                  guess=guess,
                                  guess_widths=guess_errs)

    def _render_single(self,model,band,obs,pars_tag,fit_data,psf_gmix,jac,coadd):
        """
        render a single image of model with pars_tag in fit_data and psf_gmix w/ jac
        """
        n = self._get_namer(model, coadd)

        pars_obj = fit_data[pars_tag][0].copy()
        band_pars_obj = numpy.zeros(6,dtype='f8')
        band_pars_obj[0:5] = pars_obj[0:5]
        band_pars_obj[5] = pars_obj[5+band]
        assert len(band_pars_obj) == 6

        for i in [1,2]:
            try:
                if model != 'cm':
                    gmix_sky = GMixModel(band_pars_obj, model)
                else:
                    gmix_sky = GMixCM(fit_data[n('fracdev')][0],
                                      fit_data[n('TdByTe')][0],
                                      band_pars_obj)
                gmix_image = gmix_sky.convolve(psf_gmix)
            except GMixRangeError:
                print('        setting T=0 for nbr!')
                band_pars_obj[4] = 0.0 # set T to zero and try again
        
        image = gmix_image.make_image(obs.image.shape, jacobian=jac)
        return image

    def _mask_nbr(self,mb_obs_list,nbr_ind,masked_pix,nbrs_fit_data):
        """
        mask a nbr in weight map of central using a seg map
        """
        mtype=self['unmodeled_nbrs_masking_type']
        if mtype == 'nbrs-seg':
            nbrs_number = nbrs_fit_data['number'][nbr_ind]
            for band,obs_list in enumerate(mb_obs_list):
                for obs in obs_list:
                    if obs.meta['flags'] != 0:
                        continue
                    q = numpy.where(obs.seg == nbrs_number)
                    if q[0].size > 0:
                        masked_pix[q] = 1
        else:
            raise ValueError("no support for unmodeled nbrs "
                             "masking type %s" % mtype)

    def _render_nbrs(self,model,mb_obs_list,coadd,nbrs_fit_data):
        """
        render nbrs
        """

        print('    rendering nbrs')

        if len(mb_obs_list.meta['nbrs_inds']) == 0:
            return

        n=self._get_namer(model, coadd)

        pars_name = 'max_pars'
        if n(pars_name) not in nbrs_fit_data.dtype.names:
            pars_name = 'pars'

        pars_tag = n(pars_name)
        assert pars_tag in nbrs_fit_data.dtype.names

        if 'max' in pars_tag:
            fit_flags_tag = n('max_flags')
        else:
            fit_flags_tag = n('flags')

        cen_ind = mb_obs_list.meta['cen_ind']
        for band,obs_list in enumerate(mb_obs_list):
            for obs in obs_list:
                if obs.meta['flags'] != 0:
                    continue

                # do central image first
                # FIXME - need to clean up all flags
                if (nbrs_fit_data[fit_flags_tag][cen_ind] == 0
                        and obs.has_psf_gmix()
                        and nbrs_fit_data['flags'][cen_ind] == 0):

                    cenim = self._render_single(model,band,obs,pars_tag,
                                                nbrs_fit_data[cen_ind:cen_ind+1],
                                                obs.get_psf_gmix(),obs.get_jacobian(),
                                                coadd)
                    sub_nbrs_from_cenim = False
                    print('        rendered central object')
                else:
                    if (nbrs_fit_data[fit_flags_tag][cen_ind] != 0
                            or (nbrs_fit_data['flags'][cen_ind] & GAL_FIT_FAILURE) != 0):

                        print('        bad fit data for central: FoF obj = %d' % (cen_ind+1))
                    elif (nbrs_fit_data['flags'][cen_ind] & LOW_PSF_FLUX) != 0:
                        print('        PSF flux too low for central: FoF obj = %d' % (cen_ind+1))
                    elif (nbrs_fit_data['flags'][cen_ind] & PSF_FLUX_FIT_FAILURE) != 0:
                        print('        bad PSF flux fit for central: FoF obj = %d' % (cen_ind+1))
                    elif (nbrs_fit_data['flags'][cen_ind] & PSF_FIT_FAILURE) != 0:
                        print('        bad PSF fit for central: FoF obj = %d' % (cen_ind+1))
                    else:
                        print('        central not rendered for unknown '
                              'reason: FoF obj = %d' % (cen_ind+1))
                    
                    cenim = obs.image_orig.copy()
                    sub_nbrs_from_cenim = True

                if self['intr_prof_var_fac'] > 0.0:
                    varim = numpy.zeros_like(cenim)  
                    if not sub_nbrs_from_cenim:
                        varim += self['intr_prof_var_fac']*cenim*cenim
                    
                # now do nbrs
                nbrsim = numpy.zeros_like(cenim)
                masked_pix = numpy.zeros_like(cenim)
                for nbrs_ind,nbrs_flags,nbrs_psf,nbrs_jac in zip(mb_obs_list.meta['nbrs_inds'],
                                                                 obs.meta['nbrs_flags'],
                                                                 obs.meta['nbrs_psfs'],
                                                                 obs.meta['nbrs_jacs']):
                    if (nbrs_flags == 0
                            and nbrs_fit_data[fit_flags_tag][nbrs_ind] == 0
                            and nbrs_fit_data['flags'][nbrs_ind] == 0
                            and nbrs_psf.has_gmix() ):

                        print('        rendered nbr: %d' % (nbrs_ind+1))                        
                        nbrs_psf_gmix = nbrs_psf.get_gmix()
                        curr_nbrsim = self._render_single(model,
                                                          band,
                                                          obs,
                                                          pars_tag,
                                                          nbrs_fit_data[nbrs_ind:nbrs_ind+1],
                                                          nbrs_psf_gmix,
                                                          nbrs_jac,
                                                          coadd)
                        nbrsim += curr_nbrsim
                        
                        if self['intr_prof_var_fac'] > 0.0:
                            varim += self['intr_prof_var_fac']*curr_nbrsim*curr_nbrsim

                    else:
                        print('        masked nbr: %d' % (nbrs_ind+1))
                        
                        if nbrs_flags != 0:
                            print('        nbrs_flags set to %d: FoF obj = %d' % (nbrs_flags,nbrs_ind+1))
                        elif nbrs_fit_data[fit_flags_tag][nbrs_ind] != 0 or (nbrs_fit_data['flags'][nbrs_ind] & GAL_FIT_FAILURE) != 0:
                            print('        bad fit data for nbr: FoF obj = %d' % (nbrs_ind+1))
                        elif (nbrs_fit_data['flags'][nbrs_ind] & LOW_PSF_FLUX) != 0:
                            print('        PSF flux too low for nbr: FoF obj = %d' % (nbrs_ind+1))
                        elif (nbrs_fit_data['flags'][nbrs_ind] & PSF_FLUX_FIT_FAILURE) != 0:
                            print('        bad PSF flux fit for nbr: FoF obj = %d' % (nbrs_ind+1))
                        elif (nbrs_fit_data['flags'][nbrs_ind] & PSF_FIT_FAILURE) != 0:
                            print('        bad PSF fit for nbr: FoF obj = %d' % (nbrs_ind+1))
                        elif not nbrs_psf.has_gmix():
                            # FIXME - better flagging of nbrs
                            if 'fit_flags' in obs.meta and obs.meta['fit_flags'] != 0:
                                print('        bad PSF fit data for nbr: FoF obj = %d' % (nbrs_ind+1))
                            else:
                                # FIXME - need to fit psf from off chip nbrs
                                print('        FIXME: need to fit PSF for '
                                      'off-chip nbr %d for cen %d' % (nbrs_ind+1,cen_ind+1))
                        else:
                            print('        nbr not rendered for unknown reason: FoF obj = %d' % (nbrs_ind+1))
                            
                        self._mask_nbr(mb_obs_list,nbrs_ind,masked_pix,nbrs_fit_data)
                            
                # get total image and adjust central if needed
                if sub_nbrs_from_cenim:
                    cenim -= nbrsim                    
                totim = cenim + nbrsim
                
                if self['model_nbrs_method'] == 'subtract':
                    obs.image = obs.image_orig - nbrsim
                elif self['model_nbrs_method'] == 'frac':
                    frac = numpy.zeros_like(totim)
                    frac[:,:] = 1.0
                    msk = totim > 0.0
                    frac[msk] = cenim[msk]/totim[msk]
                    obs.image = obs.image_orig*frac
                else:
                    assert False,'nbrs model method %s not implemented!' % self['model_nbrs_method']
                    
                # mask unmodeled nbrs
                new_weight = obs.weight_orig.copy()
                q = numpy.where(masked_pix == 1.0)
                if q[0].size > 0:
                    new_weight[q] = 0.0

                if self['intr_prof_var_fac'] > 0.0:
                    qnz = numpy.where(new_weight != 0.0)
                    if qnz[0].size > 0:
                        new_weight[qnz] = 1.0/(1.0/new_weight[qnz] + varim[qnz])

                obs.weight = new_weight
                    
                if self['make_plots']:
                    self._plot_nbrs_model(band,model,obs,totim,cenim,coadd)

    def _plot_nbrs_model(self,band,model,obs,totim,cenim,coadd):
        """
        plot nbrs model
        """
        if coadd:
            ptype='coadd'
        else:
            ptype='mb'

        obj_id = obs.meta['id']
        ptype = '%s-%s-%s' % (ptype,model,'max')
        title='%d %s' % (obj_id,ptype)

        def plot_seg(seg):
            seg_new = seg.copy()
            seg_new = seg_new.astype(float)
            uvals = numpy.unique(seg)
            if len(uvals) > 1:
                mval = 1.0*(len(uvals)-1)
                ind = 1.0
                for uval in uvals:
                    if uval > 0:
                        qx,qy = numpy.where(seg == uval)
                        seg_new[qx[:],qy[:]] = ind/mval
                        ind += 1
            else:
                seg_new[:,:] = 1.0

            return seg_new

        icut_cen = obs.meta['icut']

        import images
        import biggles
        width = 1920
        height = 1200
        biggles.configure('screen','width', width)
        biggles.configure('screen','height', height)
        tab = biggles.Table(2,3)
        tab.title = title

        tab[0,0] = images.view(obs.image_orig,title='original image',show=False,nonlinear=0.075)
        tab[0,1] = images.view(totim-cenim,title='models of nbrs',show=False,nonlinear=0.075)

        tab[0,2] = images.view(plot_seg(obs.seg),title='seg map',show=False)

        tab[1,0] = images.view(obs.image,title='corrected image',show=False,nonlinear=0.075)
        msk = totim != 0
        frac = numpy.zeros(totim.shape)
        frac[msk] = cenim[msk]/totim[msk]
        tab[1,1] = images.view(frac,title='fraction of flux due to central',show=False)
        tab[1,2] = images.view(obs.weight,title='weight map',show=False)

        try:
            if icut_cen > 0:
                fname = os.path.join(self.plot_dir,'%s-nbrs-band%d-icut%d.png' % (ptype,band,icut_cen))
            else:
                fname = os.path.join(self.plot_dir,'%s-nbrs-band%d-coadd.png' % (ptype,band))
            print("        making plot %s" % fname)
            tab.write_img(1920,1200,fname)
        except:
            print("        caught error plotting nbrs")
            pass

    def _fill_epoch_data(self,mb_obs_list,new_mb_obs_list):
        print('    filling PSF data')
        for band,obs_list in enumerate(mb_obs_list):
            for obs in obs_list:
                used = False
                res = None

                if obs.meta['flags'] != 0:
                    obs.update_meta_data({'fit_flags':obs.meta['flags']})
                    ed = self._make_epoch_struct()
                    ed['psf_fit_flags'] = obs.meta['flags']
                    obs.update_meta_data({'fit_data':ed})
                    continue

                if obs.meta['flags'] == 0 and obs.has_psf():
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
        print('    doing PSF stats')

        n=self._get_namer('', coadd)

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

                fdata=obs.meta['fit_data']
                if (obs.meta['flags'] == 0
                        and 'fit_data' in obs.meta
                        and fdata['psf_fit_flags'][0] == 0):

                    assert obs.meta['fit_flags'] == 0
                    assert obs.get_psf().has_gmix()
                    if fdata['wmax'][0] > 0.0:
                        did_one_max = True
                        npix += fdata['npix'][0]
                        wrelsum += fdata['wsum'][0]/fdata['wmax'][0]

                    did_one = True

                    wsum += fdata['wsum'][0]
                    Tsum += fdata['wsum'][0]*fdata['psf_fit_T'][0]
                    g1sum += fdata['wsum'][0]*fdata['psf_fit_g'][0,0]
                    g2sum += fdata['wsum'][0]*fdata['psf_fit_g'][0,1]

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

        n=self._get_namer('', coadd)

        self.data[n('nimage_use')][0,:] = nim_used

    def _get_bootstrapper(self, model, mb_obs_list):
        """
        get the bootstrapper for fitting psf through galaxy
        """
        
        find_cen=self.get('pre_find_center',False)
        if model == 'cm':
            fracdev_prior=self['model_pars']['cm']['fracdev_prior']
            boot=get_bootstrapper(mb_obs_list,
                                  type='composite',
                                  fracdev_prior=fracdev_prior,
                                  find_cen=find_cen,
                                  **self)
        else:
            boot=get_bootstrapper(mb_obs_list, find_cen=find_cen, **self)

        return boot

    def _run_boot(self, model, mb_obs_list, coadd, guess=None, **kwargs):
        """
        run a boot strapper
        """

        flags=0
        boot=self._get_bootstrapper(model,mb_obs_list)
        self.boot=boot

        n=self._get_namer('psf', coadd)

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
                    self._fit_galaxy(model,coadd,guess=guess,**kwargs)
                    self._copy_galaxy_result(model,coadd)
                    self._print_galaxy_result()
                except (BootGalFailure,GMixRangeError) as err:
                    print("    galaxy fitting failed: %s" % err)
                    flags = GAL_FIT_FAILURE

        except BootPSFFailure:
            print("    psf fitting failed")
            flags = PSF_FIT_FAILURE

        return flags, boot

    def _fit_psf_flux(self,coadd):
        self.boot.fit_gal_psf_flux(normalize_psf=self['normalize_psf'])

        res=self.boot.get_psf_flux_result()

        n=self._get_namer('psf', coadd)

        flagsall=0
        for band in xrange(self['nband']):
            flags=res['flags'][band]
            flagsall |= flags

            flux=res['psf_flux'][band]
            flux_err=res['psf_flux_err'][band]

            self.data[n('flags')][0,band] = flags
            self.data[n('flux')][0,band] = flux
            self.data[n('flux_err')][0,band] = flux_err

            print("        psf flux(%s): %g +/- %g" % (band,flux,flux_err))

        return flagsall

    def _fit_psfs(self,coadd, boot=None):
        """
        fit the psf model to every observation's psf image
        """

        print('    fitting the PSFs')

        if boot is None:
            boot=self.boot

        psf_pars = {}
        for k,v in self['psf_pars'].iteritems():
            if k != 'model' and k != 'ntry':
                psf_pars.update({k:v})

        boot.fit_psfs(self['psf_pars']['model'],
                      None,
                      Tguess_key='Tguess',
                      ntry=self['psf_pars']['ntry'],
                      fit_pars=psf_pars,
                      norm_key='psf_norm')

        # check for no obs in a band if PSF fit fails
        for band,obs_list in enumerate(boot.mb_obs_list):
            if len(obs_list) == 0:
                raise BootPSFFailure("psf fitting failed - band %d has no obs" % band)

        if (self['make_plots']
            and (('made_psf_plots' not in self.mb_obs_list.meta) or
                 ('made_psf_plots' in self.mb_obs_list.meta and
                  self.mb_obs_list.meta['made_psf_plots'] == False)) ):

            self.mb_obs_list.update_meta_data({'made_psf_plots':True})
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

        print("        making plot %s" % fname)
        plt.write_img(1920,1200,fname)

    def _fit_galaxy(self, model, coadd, guess=None, **kwargs):
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
                        print("        making plot %s" % fname)
                        plt.write_img(1920,1200,fname)

        except GMixRangeError as err:
            print("        caught error plotting resid: %s" % str(err))

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
            if fitter_type == 'isample':
                pdict=fitter.make_plots(title=title)
            else:
                pdict=fitter.make_plots(title=title,weights=wgts)
        
            pdict['trials'].aspect_ratio=1.5
            pdict['wtrials'].aspect_ratio=1.5
            
            trials_png=os.path.join(self.plot_dir,'%d-%s-trials.png' % (obj_id,ptype))
            wtrials_png=os.path.join(self.plot_dir,'%d-%s-wtrials.png' % (obj_id,ptype))
            
            print("        making plot %s" % trials_png)
            pdict['trials'].write_img(1200,1200,trials_png)
            
            print("        making plot %s" % wtrials_png)
            pdict['wtrials'].write_img(1200,1200,wtrials_png)
        except:
            print("        caught error plotting trials")

        try:
            from .util import plot_autocorr
            trials=fitter.get_trials()
            plt=plot_autocorr(trials)
            plt.title=title
            fname=os.path.join(self.plot_dir,'%d-%s-autocorr.png' % (obj_id,ptype))
            print("        making plot %s" % fname)
            plt.write_img(1000,1000,fname)
        except:
            print("        caught error plotting autocorr")

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
        print("        making plot %s" % coadd_png)
        plt.write_img(1200,1200,coadd_png)

    def _copy_galaxy_result(self, model, coadd):
        """
        Copy from the result dict to the output array
        """

        dindex=0

        res=self.gal_fitter.get_result()
        mres=self.boot.get_max_fitter().get_result()

        rres=self.boot.get_round_result()

        n=self._get_namer(model, coadd)

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
            tup=(mres['s2n_w'],
                 rres['s2n_r'],
                 mres['chi2per'],
                 mres['chi2per']*mres['dof'],
                 mres['dof'],
                 scipy.stats.chi2.sf(mres['chi2per']*mres['dof'],mres['dof']))
            print("    s2n: %.1f s2n_r: %.1f chi2per: %.3f (chi2: %.3g dof: %.3g pval: %0.3g)" % tup)

    def get_num_pars_psf(self):
        npdict = {'em1':6,
                  'em2':6*2,
                  'em3':6*3,
                  'coellip2':6*2, # this is the "full pars" count, not that used in fit
                  'coellip3':6*3,
                  'turb':6*3,
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
        #if coadd:
        if coadd and (self['fit_me_galaxy'] or self['use_coadd_prefix']):
            models = models + ['coadd_%s' % model for model in self['fit_models']]
        else:
            models = models + self['fit_models']

        return models

    def _get_fit_data_dtype(self,coadd):
        dt=[]

        nband=self['nband']
        bshape=(nband,)
        simple_npars=5+nband

        n=self._get_namer('psf', coadd)

        dt += [(n('flags'),   'i4',bshape),
               (n('flux'),    'f8',bshape),
               (n('flux_err'),'f8',bshape)]

        n=self._get_namer('', coadd)

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

        n=self._get_namer('psf', coadd)

        data[n('flags')] = NO_ATTEMPT
        data[n('flux')] = DEFVAL
        data[n('flux_err')] = DEFVAL

        n=self._get_namer('', coadd)

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
    def _fit_max(self, model, guess=None, boot=None, **kwargs):
        """
        do a maximum likelihood fit

        note prior applied during
        """

        if boot is None:
            boot=self.boot

        max_pars=self['max_pars']
        prior=self['model_pars'][model]['prior']

        guess_widths = kwargs.get('guess_widths',None)

        # now with prior
        if model == 'cm' and 'guess_TdbyTe' in kwargs:
            boot.fit_max(model,
                         max_pars,
                         prior=prior,
                         ntry=max_pars['ntry'],
                         guess=guess,
                         guess_TdbyTe=kwargs['guess_TdbyTe'],
                         guess_widths=guess_widths)
        else:
            boot.fit_max(model,
                         max_pars,
                         prior=prior,
                         ntry=max_pars['ntry'],
                         guess=guess,
                         guess_widths=guess_widths)


        if self['replace_cov']:
            print("        replacing cov")
            cov_pars=self['cov_pars']
            boot.try_replace_cov(cov_pars)

    def _fit_galaxy(self, model, coadd, guess=None, **kwargs):
        self._fit_max(model,guess=guess,**kwargs)

        rpars=self['round_pars']
        self.boot.set_round_s2n(fitter_type=rpars['fitter_type'])

        self.gal_fitter=self.boot.get_max_fitter()

        if self['make_plots']:
            self._plot_resids(self.new_mb_obs_list.meta['id'],
                              self.boot.get_max_fitter(),
                              model, coadd, 'max')
            self._plot_images(self.new_mb_obs_list.meta['id'], model, coadd)

class ISampNGMixBootFitter(MaxNGMixBootFitter):
    def _setup(self):
        super(ISampNGMixBootFitter,self)._setup()
        
        # verbose True so we can see isamp output
        self['verbose'] = True

    def _fit_galaxy(self, model, coadd, guess=None, **kwargs):
        self._fit_max(model,guess=guess)
        self._do_isample(model)
        self._add_shear_info(model)

        self.gal_fitter=self.boot.get_isampler()

        if self['make_plots']:
            self._plot_resids(self.new_mb_obs_list.meta['id'],
                              self.boot.get_max_fitter(),
                              model,
                              coadd,
                              'max')

            self._plot_images(self.new_mb_obs_list.meta['id'], model, coadd)

            self._plot_trials(self.new_mb_obs_list.meta['id'],
                              self.boot.get_isampler(),
                              model,
                              coadd,
                              'isample',
                              self.boot.get_isampler().get_iweights())

    def _do_isample(self, model):
        """
        run isample on the bootstrapper
        """
        ipars=self['isample_pars']
        prior=self['model_pars'][model]['prior']
        self.boot.isample(ipars, prior=prior)

        rpars=self['round_pars']
        self.boot.set_round_s2n(fitter_type=rpars['fitter_type'])

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

        n = self._get_namer(model, coadd)

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

class MetacalNGMixBootFitter(MaxNGMixBootFitter):
    def _setup(self):
        super(MetacalNGMixBootFitter,self)._setup()

        self['nrand'] = self.get('nrand',1)
        if self['nrand'] is None:
            self['nrand']=1

    def _fit_galaxy(self, model, coadd, guess=None,**kwargs):
        mb_obs_list = self.boot.mb_obs_list

        super(MetacalNGMixBootFitter,self)._fit_galaxy(model,
                                                       coadd,
                                                       guess=guess,
                                                       **kwargs)
        self._do_metacal(model, self.boot)

        metacal_res = self.boot.get_metacal_max_result()

        res=self.gal_fitter.get_result()
        res.update(metacal_res)

    def _do_metacal(self,
                    model,
                    boot,
                    metacal_obs=None):
        """
        the basic fitter for this class
        """

        prior=self['model_pars'][model]['prior']

        Tguess=4.0
        ppars=self['psf_pars']
        psf_fit_pars = ppars.get('fit_pars',None)
        max_pars=self['max_pars']

        # need to make this more general, rather than a single extra noise value
        if self['nrand'] is not None and self['nrand'] > 1:
            print("    nrand:",self['nrand'])

        try:
            boot.fit_metacal_max(
                ppars['model'],
                model,
                max_pars,
                Tguess,
                psf_fit_pars=psf_fit_pars,
                prior=prior,
                ntry=max_pars['ntry'],
                metacal_pars=self['metacal_pars'],
                nrand=self['nrand'],
                metacal_obs=metacal_obs
            )


        except BootPSFFailure as err:
            # the _run_boot code catches this one
            raise BootGalFailure(str(err))



    def _get_fit_data_dtype(self,coadd):
        dt=super(MetacalNGMixBootFitter,self)._get_fit_data_dtype(coadd)

        dt_mcal = self._get_metacal_dtype(coadd)
        dt += dt_mcal
        return dt

    def _copy_galaxy_result(self, model, coadd):
        super(MetacalNGMixBootFitter,self)._copy_galaxy_result(model,coadd)

        dindex=0
        res=self.gal_fitter.get_result()

        tmodel='%s_mcal' % model
        n = self._get_namer(tmodel, coadd)

        if res['flags'] == 0:

            for f in self.mcal_flist:
                front='%s_' % (model)
                mf = f.replace(front,'')
                #print("copying %s -> %s" % (f,mf))
                self.data[f][dindex] = res[mf]

    def _get_metacal_dtype(self, coadd):

        nband=self['nband']
        simple_npars=5+nband
        np=simple_npars

        dt=[]
        for model in self._get_all_models(coadd):
            front='%s_mcal' % (model)
            n=Namer(front)
            dt += [
                (n('pars'),'f8',np),
                (n('pars_cov'),'f8',(np,np)),
                (n('g'),'f8',2),
                (n('g_cov'),'f8', (2,2) ),
                #(n('c'),'f8',2),
                (n('s2n_r'),'f8'),
                (n('R'),'f8',(2,2)),
                (n('Rpsf'),'f8',2),
                (n('gpsf'),'f8',2),
            ]

        self.mcal_flist = [d[0] for d in dt]

        return dt


    def _make_struct(self,coadd):
        data=super(MetacalNGMixBootFitter,self)._make_struct(coadd)

        models=self._get_all_models(coadd)
        for model in models:
            for f in self.mcal_flist:
                data[f] = DEFVAL

        return data

def _add_noise_to_obs(obs, noise_image, noise):
    """
    parameters
    ----------
    obs: Observation, ObsList, MultiBandObsList
        The obs
    noise_image: ndarray
        Array of noise to add
        if obs is a MultiBandObsList, noise_image can also be a list
    noise: float
        sigma of noise, for modifying the weight
        if obs is a MultiBandObsList, noise can also be a list
    """
    if isinstance(obs, MultiBandObsList):
        new_mb_obs = MultiBandObsList()
        for i,obslist in enumerate(obs):

            if isinstance(noise_image,list):
                use_noise_image = noise_image[i]
                use_noise = noise[i]
            else:
                use_noise_image = noise_image
                use_noise = noise

            new_obslist=_add_noise_to_obs(obslist,use_noise_image,use_noise)
            new_mb_obs.append(new_obslist)

        return new_mb_obs

    elif isinstance(obs, ObsList):
        new_obslist = ObsList()
        for i, tobs in enumerate(obs):
            new_obs = _add_noise_to_obs(tobs, noise_image, noise)
            new_obslist.append(new_obs)

        return new_obslist

    elif isinstance(obs, Observation):

        new_weight = obs.weight.copy()
        w=numpy.where(new_weight > 0)
        new_weight[w] = 1.0/(1.0/new_weight[w] + noise**2)

        new_obs = Observation(obs.image + noise_image,
                              weight=new_weight,
                              jacobian=obs.jacobian.copy(),
                              psf=deepcopy(obs.psf) )

        return new_obs

    else:
        raise ValueError("obs should be an Observation,ObsList,MultiBandObsList")

class MetacalDetrendNGMixFitter(MetacalNGMixBootFitter):
    def _fit_galaxy(self, model, coadd, guess=None,**kw):

        # this runs the psfs, max fitter, and metacal
        super(MetacalDetrendNGMixFitter,self)._fit_galaxy(
            model,
            coadd,
            guess=guess,
            **kw
        )

        boot=self.boot
        obs_dict_orig=boot.get_metacal_max_result()['obs_dict']

        mb_obs_list=boot.mb_obs_list
        if len(mb_obs_list) > 1 or len(mb_obs_list[0]) > 1:
            raise NotImplementedError("fix to work with multiple obs/bands")

        obs = mb_obs_list[0][0]
        im=obs.image
        wt=obs.weight

        noise_image1 = numpy.random.normal(loc=0.0,
                                           scale=1.0,
                                           size=im.shape)
        w=numpy.where(wt > 0)
        base_noise = numpy.sqrt( numpy.median(1.0/wt[w]) )

        #
        # now add extra noise before and after metacal
        #

        Rnoise_types=['1p','1m','2p','2m']
        #Rnoise_types=None
        new_results=[]

        detrend_noises = self['target_noise']*numpy.array(self['detrend_factors'])
        print("    doing detrend noise")

        for i, dtnoise in enumerate(detrend_noises):
            extra_noise = numpy.sqrt(dtnoise**2 - base_noise**2)

            #print("    doing detrend noise: %.3f "
            #      "extra noise: %.3f" % (dtnoise,extra_noise))


            # same noise image, just scaled
            noise_image = noise_image1*extra_noise

            #
            # adding noise *before* metacal
            # new psf observations are generated, psf models are refit currently
            #
            mb_obs_before = _add_noise_to_obs(mb_obs_list, noise_image, extra_noise)
            mcal_obs_before = ngmix.metacal.get_all_metacal(
                mb_obs_before,
                types=Rnoise_types,
                **self['metacal_pars']
            )
            self._do_metacal(model, boot, metacal_obs=mcal_obs_before)
            res_before = boot.get_metacal_max_result()

            #
            # adding noise *after* metacal
            # psf models get copied over
            #
            mcal_obs_after = {}
            for key,tobs in obs_dict_orig.iteritems():
                if key in Rnoise_types:
                    new_obs = _add_noise_to_obs(tobs, noise_image, extra_noise)
                    mcal_obs_after[key] = new_obs

            self._do_metacal(model, boot, metacal_obs=mcal_obs_after)
            res_after = boot.get_metacal_max_result()

            Rnoise = res_before['mcal_R']    - res_after['mcal_R']
            #Rnoise_psf = res_before['mcal_Rpsf']    - res_after['mcal_Rpsf']

            new_res={
                'mcal_Rnoise':Rnoise,
                #'mcal_Rnoise_psf':Rnoise_psf,
            }
            new_results.append(new_res)

        res=self.gal_fitter.get_result()
        res['mcal_dt_results'] = new_results

    def _copy_galaxy_result(self, model, coadd):
        """
        copy parameters specific to this class
        """
        super(MetacalDetrendNGMixFitter,self)._copy_galaxy_result(model, coadd)

        dindex=0
        res=self.gal_fitter.get_result()

        if res['flags'] == 0:
            tmodel = '%s_mcal' % model
            n = self._get_namer(tmodel, coadd)

            d=self.data

            for idt,dtres in enumerate(res['mcal_dt_results']):

                f=n('dt_Rnoise')
                d[f][dindex,idt,:,:] = dtres['mcal_Rnoise']

                #f=n('dt_Rnoise_psf')
                #d[f][dindex,idt,:,:] = dtres['mcal_Rnoise_psf']


    def _get_metacal_dtype(self, coadd):
        dt=super(MetacalDetrendNGMixFitter,self)._get_metacal_dtype(coadd)

        ndetrend = len(self['detrend_factors'])

        for model in self._get_all_models(coadd):
            front='%s_mcal' % (model)
            n=Namer(front)
            dt += [
                (n('dt_Rnoise'),'f8',(ndetrend,2,2)),
                #(n('dt_Rnoise_psf'),'f8',(ndetrend,2)),
            ]

        # not adding these to mcal_flist
        #self.mcal_flist = [d[0] for d in dt]

        return dt


class PostcalNGMixBootFitter(MetacalNGMixBootFitter):
    def _fit_galaxy(self, model, coadd, guess=None,**kwargs):
        mb_obs_list = self.boot.mb_obs_list

        super(MetacalNGMixBootFitter,self)._fit_galaxy(model,
                                                       coadd,
                                                       guess=guess,
                                                       **kwargs)
        postcal_res=self._do_postcal(model)

        res=self.gal_fitter.get_result()
        res.update(postcal_res)

    def _get_postcal_obsdict(self, mb_obs_list):
        import galsim

        obs = mb_obs_list[0][0]
        psf_obs = obs.psf

        im = obs.image.copy()
        psf_im = psf_obs.image.copy()
        gs_im = galsim.Image(im, scale=1.0)
        gs_psf = galsim.Image(psf_im, scale=1.0)

        i_im = galsim.InterpolatedImage(gs_im)
        i_psf = galsim.InterpolatedImage(gs_psf)

        step=self['postcal_pars']['step']

        shts = [('1p',Shape( step, 0.0)),
                ('1m',Shape(-step, 0.0)),
                ('2p',Shape( 0.0,  step)),
                ('2m',Shape( 0.0, -step))]

        odict={}
        for t in shts:
            name = t[0]
            shear = t[1]

            s_i_im = i_im.shear(g1=shear.g1, g2=shear.g2)
            s_i_psf = i_psf.shear(g1=shear.g1, g2=shear.g2)

            s_im = s_i_im.drawImage(ny=im.shape[0],
                                    nx=im.shape[1],
                                    scale=1.0,
                                    method='no_pixel')
            s_psf_im = s_i_psf.drawImage(ny=psf_im.shape[0],
                                         nx=psf_im.shape[1],
                                         scale=1.0,
                                         method='no_pixel')

            spsf_obs = Observation(
                s_psf_im.array,
                weight=psf_obs.weight.copy(),
                jacobian=psf_obs.jacobian.copy()
            )
            sobs = Observation(
                s_im.array,
                weight=obs.weight.copy(),
                jacobian=obs.jacobian.copy(),
                psf=spsf_obs
            )

            odict[name] = sobs

        return odict

    def _do_postcal(self,
                    model,
                    boot=None):
        """
        the basic fitter for this class
        """

        print("    doing postcal")

        if boot is None:
            boot=self.boot

        odict = self._get_postcal_obsdict(boot.mb_obs_list)

        Tguess = boot.mb_obs_list[0][0].psf.meta['Tguess']

        psf_pars = {}
        for k,v in self['psf_pars'].iteritems():
            if k != 'model' and k != 'ntry':
                psf_pars.update({k:v})

        fits={}
        for key in odict:
            obs = odict[key]

            tboot=self._get_bootstrapper(model, obs)
            
            tboot.fit_psfs(self['psf_pars']['model'],
                           Tguess,
                           ntry=self['psf_pars']['ntry'],
                           fit_pars=psf_pars)


            self._fit_max(model, boot=tboot)

            tboot.set_round_s2n()
            res=tboot.get_max_fitter().get_result()
            rres=tboot.get_round_result()
            res['s2n_r'] = rres['s2n_r']
            res['T_r'] = rres['T_r']

            fits[key] = res

        res = self._extract_postcal_responses(fits)
        return res

    def _extract_postcal_responses(self, fits):
        """
        pars pars_cov gpsf, s2n_r, T_r, psf_T_r required

        expect the shape to be in pars[2] and pars[3]
        """
        step = self['postcal_pars']['step']

        res1p = fits['1p']
        res1m = fits['1m']
        res2p = fits['2p']
        res2m = fits['2m']

        pars_mean = (res1p['pars']+
                     res1m['pars']+
                     res2p['pars']+
                     res2m['pars'])/4.0

        pars_cov_mean = (res1p['pars_cov']+
                         res1m['pars_cov']+
                         res2p['pars_cov']+
                         res2m['pars_cov'])/4.0

        pars_mean[2] = 0.5*(fits['1p']['pars'][2] + fits['1m']['pars'][2])
        pars_mean[3] = 0.5*(fits['2p']['pars'][3] + fits['2m']['pars'][3])

        s2n_r_mean = (res1p['s2n_r']
                      + res1m['s2n_r']
                      + res2p['s2n_r']
                      + res2m['s2n_r'])/4.0

        if self['verbose']:
            print_pars(pars_mean, front='    parsmean:   ')

        R=numpy.zeros( (2,2) ) 
        Rpsf=numpy.zeros(2)

        fac = 1.0/(2.0*step)

        R[0,0] = (fits['1p']['pars'][2]-fits['1m']['pars'][2])*fac
        R[0,1] = (fits['1p']['pars'][3]-fits['1m']['pars'][3])*fac
        R[1,0] = (fits['2p']['pars'][2]-fits['2m']['pars'][2])*fac
        R[1,1] = (fits['2p']['pars'][3]-fits['2m']['pars'][3])*fac

        #Rpsf[0] = (pars['1p_psf'][2]-pars['1m_psf'][2])*fac
        #Rpsf[1] = (pars['2p_psf'][3]-pars['2m_psf'][3])*fac


        #gpsf_name = 'pcal_%spsf' % shape_type
        #raw_gpsf_name = '%spsf' % shape_type
        res = {
            'pcal_pars':pars_mean,
            'pcal_pars_cov':pars_cov_mean,
            'pcal_g':pars_mean[2:2+2],
            'pcal_g_cov':pars_cov_mean[2:2+2, 2:2+2],
            'pcal_R':R,
            #'pcal_Rpsf':Rpsf,
            #'pcal_gpsf':fits['gpsf'],
            'pcal_s2n_r':s2n_r_mean,
        }
        return res




    def _get_fit_data_dtype(self,coadd):
        dt=super(MetacalNGMixBootFitter,self)._get_fit_data_dtype(coadd)

        dt_pcal = self._get_postcal_dtype(coadd)
        dt += dt_pcal
        return dt

    def _copy_galaxy_result(self, model, coadd):
        super(MetacalNGMixBootFitter,self)._copy_galaxy_result(model,coadd)

        dindex=0
        res=self.gal_fitter.get_result()

        tmodel='%s_pcal' % model
        n = self._get_namer(tmodel, coadd)

        if res['flags'] == 0:

            for f in self.pcal_flist:
                front='%s_' % (model)
                mf = f.replace(front,'')
                #print("copying %s -> %s" % (f,mf), res[mf])
                self.data[f][dindex] = res[mf]

    def _get_postcal_dtype(self, coadd):

        nband=self['nband']
        np=5+nband

        dt=[]
        for model in self._get_all_models(coadd):
            front='%s_pcal' % (model)
            n=Namer(front)
            dt += [
                (n('pars'),'f8',np),
                (n('pars_cov'),'f8',(np,np)),
                (n('g'),'f8',2),
                (n('g_cov'),'f8', (2,2) ),
                #(n('c'),'f8',2),
                (n('s2n_r'),'f8'),
                (n('R'),'f8',(2,2)),
                #(n('Rpsf'),'f8',2),
                #(n('gpsf'),'f8',2),
            ]

        self.pcal_flist = [d[0] for d in dt]

        return dt


    def _make_struct(self,coadd):
        data=super(MetacalNGMixBootFitter,self)._make_struct(coadd)

        models=self._get_all_models(coadd)
        for model in models:
            for f in self.pcal_flist:
                data[f] = DEFVAL

        return data


class PostcalNGMixSimFitter(PostcalNGMixBootFitter):
    def _do_postcal(self, model):

        mb_obs_list = boot.mb_obs_list
        if len(mb_obs_list) > 1 or len(mb_obs_list[0]) > 1:
            raise NotImplementedError("only a single obs for now")


        res=super(PostcalNGMixSimFitter,self)._do_postcal(model)


        print("    Calculating Rnoise")

        fitter = self.boot.get_max_fitter()
        gmix_list = []
        for band in xrange(self['nband']):
            gm = fitter.get_gmix(band=band)
            gmix_list.append(gm)


        # for noise added *before* metacal steps
        mobs_before = ngmix.simobs.simulate_obs(
            gmix_list,
            mb_obs_list,
            add_noise=True,
            convolve_psf=True
        )
        # for noise added *after* metacal steps
        mobs_after = ngmix.simobs.simulate_obs(
            gmix_list,
            mb_obs_list,
            add_noise=False,
            convolve_psf=True
        )


        boot_model_before = self._get_bootstrapper(model,mobs_before)
        boot_model_after = self._get_bootstrapper(model,mobs_after)

        mcal_obs_after = ngmix.metacal.get_all_metacal(
            mobs_after[0][0],
            self['metacal_pars']['step'],
        )

        # now add noise after creating the metacal observations
        # using the same noise image!

        noise = mobs_before[0][0].noise_image
        for key in mcal_obs_after:
            obs=mcal_obs_after[key]
            obs.image = obs.image + noise

        res_before=self._do_postcal(
            model,
            boot=boot_model_before
        )
        res_after=self._do_postcal(
            model,
            boot=boot_model_after,
            metacal_obs=mcal_obs_after
        )

        res_before = boot_model_before.get_metacal_max_result()
        res_after = boot_model_after.get_metacal_max_result()

        gnoise = res_before['mcal_g'] - res_after['mcal_g']
        Rnoise = res_before['mcal_R'] - res_after['mcal_R']
        Rpsf_noise = res_before['mcal_Rpsf'] - res_after['mcal_Rpsf']

        res = self.boot.get_max_fitter().get_result()
        res['mcal_gnoise'] = gnoise
        res['mcal_Rnoise'] = Rnoise
        res['mcal_Rpsf_noise'] = Rpsf_noise





    def _get_postcal_dtype(self, coadd):
        dt=super(PostcalNGMixSimFitter,self)._get_postcal_dtype(coadd)

        nband=self['nband']
        np=5+nband

        for model in self._get_all_models(coadd):
            front='%s_pcal' % (model)
            n=Namer(front)
            dt += [
                (n('Rnoise'),'f8',(2,2)),
            ]

        self.pcal_flist += [d[0] for d in dt]

        return dt




class MetacalSubnNGMixBootFitter(MetacalNGMixBootFitter):
    def _get_subn_metacal_obs(self):
        """
        subtract a correlated noise image sheared by
        negative of the shear applied to the real obs
        """
        from ngmix.metacal import get_all_metacal

        print("    subtracting sheared correlated noise")
        mb_obs_list = self.boot.mb_obs_list

        # simulated noise for these observations

        step = self['metacal_pars']['step']
        # currentlly only works for single obs
        mcal_obs = get_all_metacal(mb_obs_list, step)

        nrand = self.get('subn_nrand',1)

        for irand in xrange(nrand):
            tnoise_mb_obs = ngmix.simobs.simulate_obs(None,
                                                      mb_obs_list)
            tmcal_noise_obs = get_all_metacal(tnoise_mb_obs, step)

            if irand==0:
                mcal_noise_obs = tmcal_noise_obs 
            else:
                #print("    adding extra realization:",irand)
                for key in mcal_noise_obs:

                    # these are MultiBandObsLists
                    mobs = mcal_obs[key]
                    nmobs = mcal_noise_obs[key]
                    tnmobs = tmcal_noise_obs[key] 

                    assert len(mobs)==len(nmobs)

                    # loop over bands
                    for band in xrange(len(mobs)):
                        # mobs[band] etc are ObsLists

                        # append copies of the original, for adding
                        # the noise images to later
                        nn = len(tnmobs[band])

                        mobs[band].extend( deepcopy(mobs[band][0:nn] ) )

                        # append more observations from the ObsList
                        nmobs[band].extend( tnmobs[band] )

                        assert len(mobs[band])==len(nmobs[band])


        # add the noise sheared by negative of the shear
        # applied to observation

        for ipairs in [('1p','1m'),
                       ('1m','1p'),
                       ('2p','2m'),
                       ('2m','2p'),
                       ('1p_psf','1m_psf'),
                       ('1m_psf','1p_psf'),
                       ('2p_psf','2m_psf'),
                       ('2m_psf','2p_psf')]:


            mk=ipairs[0]
            nk=ipairs[1]

            imbobs = mcal_obs[mk]
            nmbobs = mcal_noise_obs[nk]

            for imb in xrange(len(imbobs)):
                iolist=imbobs[imb]
                nolist=nmbobs[imb]

                for iobs in xrange(len(iolist)):

                    obs  = iolist[iobs]
                    nobs = nolist[iobs]

                    im  = obs.image
                    nim = nobs.image

                    obs.image = im + nim
                    obs.weight = 0.5*obs.weight

        return mcal_obs

    def _fit_galaxy(self, model, coadd, guess=None,**kwargs):
        mb_obs_list = self.boot.mb_obs_list

        super(MetacalNGMixBootFitter,self)._fit_galaxy(model,
                                                       coadd,
                                                       guess=guess,
                                                       **kwargs)
        metacal_obs = self._get_subn_metacal_obs()
        self._do_metacal(model, self.boot, metacal_obs=metacal_obs)

        metacal_res = self.boot.get_metacal_max_result()

        res=self.gal_fitter.get_result()
        res.update(metacal_res)


class MetacalSimnNGMixBootFitter(MetacalNGMixBootFitter):

    def _fit_galaxy(self, model, coadd, guess=None,**kwargs):
        mb_obs_list = self.boot.mb_obs_list
        if len(mb_obs_list) > 1 or len(mb_obs_list[0]) > 1:
            raise NotImplementedError("only a single obs for now")


        super(MetacalSimnNGMixBootFitter,self)._fit_galaxy(
            model,
            coadd,
            guess=guess,
            **kwargs
        )

        print("    Calculating Rnoise")

        fitter = self.boot.get_max_fitter()
        gmix_list = []
        for band in xrange(self['nband']):
            gm = fitter.get_gmix(band=band)
            gmix_list.append(gm)

        # for noise added *before* metacal steps
        mobs_before = ngmix.simobs.simulate_obs(
            gmix_list,
            mb_obs_list,
            add_noise=True,
            convolve_psf=True
        )
        # for noise added *after* metacal steps
        mobs_after = ngmix.simobs.simulate_obs(
            gmix_list,
            mb_obs_list,
            add_noise=False,
            convolve_psf=True
        )


        boot_model_before = self._get_bootstrapper(model,mobs_before)
        boot_model_after = self._get_bootstrapper(model,mobs_after)

        mcal_obs_after = ngmix.metacal.get_all_metacal(
            mobs_after[0][0],
            self['metacal_pars']['step'],
        )

        # now add noise after creating the metacal observations
        # using the same noise image!

        noise = mobs_before[0][0].noise_image
        for key in mcal_obs_after:
            obs=mcal_obs_after[key]
            obs.image = obs.image + noise

        self._do_metacal(
            model,
            boot_model_before,
        )
        self._do_metacal(
            model,
            boot_model_after,
            metacal_obs=mcal_obs_after
        )

        res_before = boot_model_before.get_metacal_max_result()
        res_after = boot_model_after.get_metacal_max_result()

        gnoise = res_before['mcal_g'] - res_after['mcal_g']
        Rnoise = res_before['mcal_R'] - res_after['mcal_R']
        Rpsf_noise = res_before['mcal_Rpsf'] - res_after['mcal_Rpsf']

        res = self.boot.get_max_fitter().get_result()
        res['mcal_gnoise'] = gnoise
        res['mcal_Rnoise'] = Rnoise
        res['mcal_Rpsf_noise'] = Rpsf_noise

    def _get_metacal_dtype(self, coadd):
        dt = super(MetacalSimnNGMixBootFitter,self)._get_metacal_dtype(coadd)

        dt_extra=[]
        for model in self._get_all_models(coadd):
            tmodel='%s_pcal' % model
            n = self._get_namer(tmodel, coadd)
            dt_extra += [
                (n('gnoise'), 'f8', 2),
                (n('Rnoise'), 'f8', (2,2)),
                (n('Rpsf_noise'), 'f8', 2),
            ]

        self.mcal_flist += [d[0] for d in dt_extra]

        dt += dt_extra

        return dt

class MetacalAddnNGMixBootFitter(MetacalSimnNGMixBootFitter):
    def _get_noisier_mobs(self):

        fac=self['simnoise']['noise_fac']
        ofacsq=1.0/(1.0 + fac**2)
        #ofacsq=1.0

        mobs=self.mb_obs_list

        new_mobs=MultiBandObsList()
        for olist in mobs:

            new_olist=ObsList()
            for obs in olist:
                noise_image = ngmix.simobs.get_noise_image(obs.weight)

                noise_image *= fac
                new_weight = obs.weight.copy()
                new_weight *= ofacsq

                new_image = obs.image.copy()
                new_image += noise_image

                new_obs = Observation(
                    new_image,
                    weight=new_weight,
                    jacobian=obs.jacobian.copy(),
                    psf=obs.psf
                )

                new_obs.noise_image=noise_image

                new_olist.append(new_obs)
            new_mobs.append(new_olist)

        return new_mobs

    def _fit_galaxy(self, model, coadd, guess=None,**kwargs):
        # note jumping over parent here
        super(MetacalSimnNGMixBootFitter,self)._fit_galaxy(model,
                                                           coadd,
                                                           guess=guess,
                                                           **kwargs)

        if len(self.mb_obs_list) > 1 or len(self.mb_obs_list[0]) > 1:
            raise NotImplementedError("only a single obs for now")

        print("    Calculating Rnoise by adding noise to image")

        mobs_before = self._get_noisier_mobs()

        boot_model_before = self._get_bootstrapper(model,mobs_before)
        boot_model_after = self._get_bootstrapper(model,self.mb_obs_list)

        mcal_obs_after = ngmix.metacal.get_all_metacal(
            self.mb_obs_list[0][0],
            self['metacal_pars']['step'],
        )

        # now add noise after creating the metacal observations
        # using the same noise image!

        noise_image = mobs_before[0][0].noise_image
        new_weight = mobs_before[0][0].weight
        for key in mcal_obs_after:
            obs=mcal_obs_after[key]
            obs.image = obs.image + noise_image
            obs.weight = new_weight.copy()

        self._do_metacal(model, boot_model_before)
        self._do_metacal(model, boot_model_after,
                         metacal_obs=mcal_obs_after)

        res_before = boot_model_before.get_metacal_max_result()
        res_after = boot_model_after.get_metacal_max_result()

        fac2inv=1.0/self['simnoise']['noise_fac']**2
        #print("        multiplying by:",fac2inv)
        Rnoise = (res_before['mcal_R'] - res_after['mcal_R'])*fac2inv
        Rpsf_noise = (res_before['mcal_Rpsf'] - res_after['mcal_Rpsf'])*fac2inv

        print("    Rnoise[0,0]: %g" % Rnoise[0,0])

        res = self.boot.get_max_fitter().get_result()
        res['mcal_Rnoise'] = Rnoise
        res['mcal_Rpsf_noise'] = Rpsf_noise


class MetacalRegaussBootFitter(MaxNGMixBootFitter):

    def _guess_and_run_boot(self,model,mb_obs_list,coadd,**kw):

        flags=0

        boot=get_bootstrapper(mb_obs_list, find_cen=False, **self)
        self.boot=boot

        res={}
        ppars=self['psf_pars']
        rpars=self['regauss_pars']

        try:
            boot.fit_metacal_regauss(
                ppars['Tguess'],
                rpars['Tguess'],
                psf_ntry=ppars['ntry'],
                ntry=rpars['ntry'],
                metacal_pars=self['metacal_pars'],
            )
            mres=boot.get_metacal_regauss_result()
            res.update(mres)

            # yikes, monkey patching
            # TODO figure out how to fix this
            boot._result = res

            self._copy_galaxy_result()
            self._print_galaxy_result()

        except BootPSFFailure as err:
            print("    psf fitting failed")
            flags = PSF_FIT_FAILURE
        except (BootGalFailure,GMixRangeError) as err:
            print("    galaxy fitting failed")
            flags = GAL_FIT_FAILURE

        except BootPSFFailure:
            print("    psf fitting failed")
            flags = PSF_FIT_FAILURE

        res['flags'] = flags
        return flags, boot

    def _print_galaxy_result(self):
        print("implement print")

    def _copy_galaxy_result(self):
        dindex=0
        res=self.boot.get_metacal_regauss_result()

        data=self.data

        if res['flags'] == 0:
            for f in ['pars','pars_cov','e','e_cov',
                      'c','R', 'Rpsf','epsf']:

                mf = 'mcal_%s' % f
                data[f][dindex] = res[mf]

    def _get_fit_data_dtype(self, coadd):

        np=6

        nband=self['nband']
        bshape=(nband,)

        dt = [
            ('nimage_use','i4',bshape),
            ('pars','f8',np),
            ('pars_cov','f8',(np,np)),
            ('e','f8',2),
            ('e_cov','f8', (2,2) ),
            ('c','f8',2),
            ('R','f8',(2,2)),
            ('Rpsf','f8',2),
            ('epsf','f8',2),
        ]

        return dt

    def _make_struct(self,coadd):

        dt=self._get_fit_data_dtype(coadd)
        num=1
        data=numpy.zeros(num, dtype=dt)

        for n in data.dtype.names:
            if n != 'flags':

                data[n] = DEFVAL

        return data
