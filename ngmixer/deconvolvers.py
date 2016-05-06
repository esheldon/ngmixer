from __future__ import print_function
import time
import numpy
from numpy import array
import os
import scipy.stats

# local imports
from .defaults import DEFVAL, NO_ATTEMPT, \
    PSF_FIT_FAILURE, GAL_FIT_FAILURE, \
    LOW_PSF_FLUX, PSF_FLUX_FIT_FAILURE, \
    NBR_HAS_NO_PSF_FIT, METACAL_FAILURE
from .fitting import BaseFitter
from .util import Namer, print_pars
from .render_ngmix_nbrs import RenderNGmixNbrs

# ngmix imports
import ngmix
from ngmix import Observation, ObsList, MultiBandObsList, GMixRangeError
from ngmix.gexceptions import BootPSFFailure, BootGalFailure
from ngmix.gmix import GMix
from ngmix.jacobian import Jacobian

from pprint import pprint

from .bootfit import NGMixBootFitter


try:
    import deconv
except ImportError:
    pass

class Deconvolver(NGMixBootFitter):
    def __init__(self,*args,**kw):
        super(Deconvolver,self).__init__(*args,**kw)

    def _setup(self):
        """
        ngmix-specific setups for all ngmix fitters
        """

        # whether to use the coadd_ prefix for names when no me fit was done
        self['use_coadd_prefix'] = self.get('use_coadd_prefix',True)

    def __call__(self,mb_obs_list,coadd=False, **kw):
        """
        fit the obs list
        """

        # only fit stuff that is not flagged
        new_mb_obs_list = self._get_good_mb_obs_list(mb_obs_list)
        self.new_mb_obs_list = new_mb_obs_list
        self.mb_obs_list = mb_obs_list

        mb_obs_list.update_meta_data({'fit_data':self._make_struct(coadd)})
        self.data = mb_obs_list.meta['fit_data']

        if self['make_plots']:
            self.plot_dir = './%d-plots' % new_mb_obs_list.meta['id']
            if not os.path.exists(self.plot_dir):
                os.makedirs(self.plot_dir)

        '''
        cres=self._find_center(mb_obs_list)
        if cres['flags'] != 0:
            print("could not find center")
            return GAL_FIT_FAILURE
        '''

        flags=0
        res=None
        try:
            # this sets self.boot, which is only used for psf stuff
            self._fit_psfs(new_mb_obs_list)
            flags |= self._fit_psf_flux(coadd)

            try:
                # to get a center
                self._fit_gauss()

                meas = self._do_deconv(new_mb_obs_list)
                self._copy_galaxy_result(meas, coadd)
                res=meas.get_result()

                if res['flags'] != 0:
                    print("    deconv failed with flags:",res['flags'])
                    flags |= GAL_FIT_FAILURE

            except BootGalFailure:
                print("    galaxy fitting failed: %s" % err)
                flags = GAL_FIT_FAILURE

        except BootPSFFailure:
            print("    psf fitting failed")
            flags = PSF_FIT_FAILURE

        # fill the epoch data
        self._fill_epoch_data(mb_obs_list,new_mb_obs_list)

        # fill in PSF stats in data rows
        if (flags & PSF_FIT_FAILURE) == 0:
            self._do_psf_stats(mb_obs_list,coadd)

        self._fill_nimage_used(res, coadd)

        return flags

    def _do_deconv(self, mb_obs_list):

        mfitter=self.boot.get_max_fitter()
        mres=mfitter.get_result()

        cen=mres['pars'][0:0+2].copy()

        mbo=self._trim_images(mb_obs_list, cen)

        meas=deconv.measure.calcmom_ksigma_obs(
            mbo,
            self['sigma_weight'],  # arcsec
            dk=self['dk'],         # 1/arcsec or None
        )

        if self['make_plots']:
            self._do_deconv_plots(meas)

        return meas

    def _trim_images(self, mbo_input, censky):
        """
        cen in sky coords, relative to jacobian
        center
        """

        mbo = MultiBandObsList()
        for obslist in mbo_input:

            new_obslist=ObsList()
            for obs in obslist:

                j=obs.jacobian
                scale=j.get_scale()
                cenpix=array( j.get_cen() )

                new_cen = cenpix + censky/scale
                print("cen0:",cenpix,"newcen:",new_cen)

                new_im=_trim_image(obs.image, new_cen)
                new_wt=_trim_image(obs.image, new_cen)
                new_j = j.copy()
                new_j.set_cen(row=new_cen[0], col=new_cen[1])

                newobs = Observation(
                    new_im,
                    weight=new_wt,
                    jacobian=new_j,
                    psf=obs.psf,
                )

                new_obslist.append( newobs )
            mbo.append( new_obslist )

        return mbo

    def _find_center(self, mb_obs_list):
        scale=mb_obs_list[0][0].jacobian.get_scale()
        Tguess=4.0 * scale**2

        max_pars=self['max_pars']
        fit_pars=max_pars['fit_pars']

        guesser=ngmix.guessers.TFluxGuesser(
            Tguess,
            Fguess,
            scaling='linear'
        )
        runner=ngmix.bootstrap.MaxRunner(
            mb_obs_list,
            'gauss',
            Tguess,
            fit_pars,
        )
        runner.go(ntry=max_pars['ntry'])
        res=runner.fitter.get_result()
        return res

    def _do_deconv_plots(self, meas):
        import images

        mlist=meas.get_meas_list()

        for iband,blist in enumerate(mlist):
            for icut,m in enumerate(blist):
                pngfile=os.path.join(
                    self.plot_dir,
                    'deconv-%d-%02d.png' % (iband,icut),
                )
                imlist=[
                    self._scale_image(m.gal_image.array),
                    self._scale_image(m.psf_image.array),
                    self._scale_image(m.kimage),
                    self._scale_image(m.kimage*m.kweight),
                ]
                titles=['image','psf','kimage','weighted k image']

                print("writing plot:",pngfile)
                images.view_mosaic(
                    imlist,
                    titles=titles,
                    width=1000,
                    height=1000,
                    file=pngfile,
                )

    def _scale_image(self, imin):
        im=imin.copy()

        maxval=im.max()
        #im = numpy.log10(im.clip(min=1.0e-4*maxval))
        im = numpy.log10(im.clip(min=1.0e-3*maxval))
        im -= im.min()
        im /= im.max()

        return im


    def _fill_nimage_used(self,res,coadd):
        nim_used = numpy.zeros(self['nband'],dtype='i4')
        for band in xrange(self['nband']):
            nim_used[band] = res['nimage_use_band'][band]

        n=self._get_namer('', coadd)

        self.data[n('nimage_use')][0,:] = nim_used

    def _fit_psfs(self, mb_obs_list):
        """
        fit the psf model to every observation's psf image
        """

        print('    fitting the PSFs')
        boot=ngmix.bootstrap.Bootstrapper(mb_obs_list)

        psf_pars=self['psf_pars']
        fit_pars=psf_pars['fit_pars']

        boot.fit_psfs(psf_pars['model'],
                      None,
                      Tguess_key='Tguess',
                      ntry=psf_pars['ntry'],
                      fit_pars=fit_pars,
                      norm_key='psf_norm')

        # check for no obs in a band if PSF fit fails
        for band,obs_list in enumerate(boot.mb_obs_list):
            if len(obs_list) == 0:
                raise BootPSFFailure("psf fitting failed - band %d has no obs" % band)

        self.boot=boot

    def _fit_gauss(self):
        """
        to get a center
        """

        prior=self['model_pars']['gauss']['prior']

        max_pars=self['max_pars']

        self.boot.fit_max(
            'gauss',
            max_pars,
            ntry=max_pars['ntry'],
            prior=prior,
        )

        res=self.boot.get_max_fitter().get_result()
        if res['flags'] != 0:
            raise BootGalFailure("failure fitting gauss")


    def _copy_galaxy_result(self, meas, coadd):
        """
        Copy from the result dict to the output array
        """

        dindex=0
        data=self.data

        res=meas.get_result()

        data['dc_flags'][dindex] = res['flags']
        data['dc_orflags'][dindex] = res['orflags']
        data['dc_flags_band'][dindex] = res['flags_band']
        data['dc_orflags_band'][dindex] = res['orflags_band']

        data['dk'][dindex] = res['dk']

        data['T'][dindex] = res['T']
        data['e'][dindex] = res['e']
        data['wflux'][dindex] = res['wflux']
        data['wflux_band'][dindex] = res['wflux']

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

    def get_fit_data_dtype(self,me,coadd):
        dt = []
        if me:
            dt += self._get_fit_data_dtype(False)
        if coadd:
            dt += self._get_fit_data_dtype(True)
        return dt

    def _get_fit_data_dtype(self,coadd):
        dt=[]

        nband=self['nband']
        bshape=(nband,)
        simple_npars=5+nband

        n=self._get_namer('psf', coadd)

        dt += [(n('flags'),   'i4',bshape),
               (n('flux'),    'f8',bshape),
               (n('flux_err'),'f8',bshape),
               (n('flux_s2n'),'f8',bshape)]

        n=self._get_namer('', coadd)

        dt += [(n('nimage_use'),'i4',bshape)]

        dt += [(n('mask_frac'),'f8'),
               (n('psfrec_T'),'f8'),
               (n('psfrec_g'),'f8', 2)]

        if nband==1:
            fcov_shape=(nband,)
        else:
            fcov_shape=(nband,nband)

        dt += [
            ('dc_flags','i4'),
            ('dc_orflags','i4'),
            ('dc_flags_band','i4',bshape),
            ('dc_orflags_band','i4',bshape),

            ('dk','f8'),

            ('T','f8'),
            ('e','f8',2),
            ('wflux','f8',bshape),
            ('wflux_band','f8',bshape),
        ]

        return dt

    def _make_struct(self,coadd):
        """
        make the output structure
        """
        num = 1
        dt = self._get_fit_data_dtype(coadd)
        data = numpy.zeros(num,dtype=dt)

        n=self._get_namer('psf', coadd)

        data[n('flags')] = NO_ATTEMPT
        data[n('flux')] = DEFVAL
        data[n('flux_err')] = DEFVAL
        data[n('flux_s2n')] = DEFVAL

        n=self._get_namer('', coadd)

        data[n('mask_frac')] = DEFVAL
        data[n('psfrec_T')] = DEFVAL
        data[n('psfrec_g')] = DEFVAL

        # the deconvolve parameters
        data['dc_flags']        = NO_ATTEMPT
        data['dc_orflags']      = NO_ATTEMPT
        data['dc_flags_band']   = NO_ATTEMPT
        data['dc_orflags_band'] = NO_ATTEMPT
        data['dk']              = DEFVAL
        data['T']               = DEFVAL
        data['e']               = DEFVAL
        data['wflux']           = DEFVAL
        data['wflux_band']      = DEFVAL

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

def _trim_image(im, cen):

    drow_low = cen[0]
    drow_high = im.shape[0] - cen[0] - 1

    dcol_low = cen[1]
    dcol_high = im.shape[1] - cen[1] - 1

    drow = min(drow_low,drow_high)
    dcol = min(dcol_low,dcol_high)

    frlow=cen[0]-drow
    frhigh=cen[0]+drow
    fclow=cen[1]-dcol
    fchigh=cen[1]+dcol


    rlow=int(frlow)
    rhigh=int(frhigh)
    clow=int(fclow)
    chigh=int(fchigh)

    '''
    print(frlow,rlow)
    print(frhigh,rhigh)
    print(fclow,clow)
    print(fchigh,chigh)
    '''


    return im[
        rlow:rhigh+1,
        clow:chigh+1,
    ]



