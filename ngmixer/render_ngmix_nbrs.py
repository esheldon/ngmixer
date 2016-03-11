from __future__ import print_function
import time
import numpy
import os

from copy import deepcopy

# local imports
from .defaults import DEFVAL, NO_ATTEMPT, \
    PSF_FIT_FAILURE, GAL_FIT_FAILURE, \
    LOW_PSF_FLUX, PSF_FLUX_FIT_FAILURE
from .util import Namer, print_pars

# ngmix imports
import ngmix
from ngmix import Observation, ObsList, MultiBandObsList, GMixRangeError
from ngmix.fitting import EIG_NOTFINITE
from ngmix.gmix import GMixModel, GMix, GMixCM
from ngmix.shape import Shape
from ngmix.jacobian import Jacobian

from pprint import pprint

class RenderNGmixNbrs(object):
    """
    class to render nbrs in stamps from ngmix MOF
    
    The procedures to render the nbrs follow exaclty those used for the actual 
    MOF. (In fact, the MOF uses the static methods of this class.) 
    
    Example
    -------    
    import fitsio
    from ngmixer import RenderNGmixNbrs
    
    # read the data
    fit_data = fitsio.read('/path/to/ngmix/output.fits')
    nbrs_data = fitsio.read('/path/to/ngmix/output.fits',ext='nbrs_data')
    
    # render the nbrs
    conf = {'unmodeled_nbrs_masking_type':'nbrs-seg'}
    renderer = RenderNGmixNbrs(fit_data, nbrs_data, **conf)
    
    # this returns a list of images and masks (but some elements could be none)
    # the central image could be None too
    band = 0 # g band for griz MOF
    model = 'cm' # cmodel fit
    object_segmap = ... # object seg map obtained from MEDS via 
                        #  interpolating coadd seg onto single epoch
    cen_img, nbrs_imgs, nbrs_masks = renderer.render_nbrs(coadd_object_id,
                                                          meds_cutout_index,
                                                          object_segmap,
                                                          model,
                                                          band)
                                         
    # you can do things like add the images up (and AND the masks)
    nbrs_img = numpy.zeros(object_seg.shape,dtype='f8')
    for img in nbrs_imgs:
        if img is not None:
            nbrs_img += img
    
    nbr_mask = numpy.ones(object_seg.shape,dtype='f8')
    for msk in nbrs_masks:
        if msk is not None:
            nbr_mask *= msk 

    # you can also have the code do this for you by sending total=True
    cen_img, nbrs_img, nbr_mask = renderer.render_nbrs(coadd_object_id,
                                                       meds_cutout_index,
                                                       object_segmap,
                                                       model,
                                                       band,
                                                       total=True)
    
    Parameters
    ----------
    fit_data: the ngmix fit data from the MOF
    nbrs_data: the nbrs data from the MOF
    
    Optional Arguments
    ------------------
    unmodeled_nbrs_masking_type: string, method to use to mask unmodeled nbrs (default: 'nbrs-seg')
        see _mask_nbr_seg method docs for details
        
    Methods
    -------
    render_nbrs: method to render nbrs of a given object    
    """

    def __init__(self,fit_data,nbrs_data,**kwargs):
        self._conf = {}
        self._conf.update(kwargs)
        
        self.fit_data = fit_data
        self.nbrs_data = nbrs_data

    def _set_defaults(self):
        self._conf['unmodeled_nbrs_masking_type'] = self._conf.get('unmodeled_nbrs_masking_type','nbrs-seg')
        self._conf['prematch'] = self._conf.get('prematch',False)
        
    def render_nbrs(self,cen_id,
                    cutout_index,
                    segmap,
                    model,
                    band,
                    total=False,
                    unmodeled_nbrs_masking_type='nbrs-seg',
                    verbose=True):
        """
        render nbr images for a given object
        
        Parameters
        ----------
        cen_id: (coadd_objects) id for central object
        cutout_index: index of cutout in the MEDS file for this object
        segmap: coadd segmap around central object interpolated to the SE image (see the MEDS 
            method 'interpolate_coadd_seg')
        model: string with ngmix model to render (e.g., 'cm' for cmodel fits)
        band: integer gigivng band (e.g., 0 for g with griz MOF)
        
        Optional Parameters
        -------------------
        total: if set to True, return the sum of all nbrs images (and the and of any masks)
        verbose: bool indicating if the code should tell you things that happen (default: True)
        
        Returns
        -------
        cen_img: image of central (if passes all flags and cuts in _render_nbrs)
        nbr_imgs: list of images of the nbrs (if they pass all flags and cuts, otherwise None)
        nbr_masks: list of nbrs masks, if any nbr does not have an image in nbr_imgs, then its pixels 
            to be masked (according to unmodeled_nbrs_masking_type, see _mask_nbrs_seg) are set to 0 
            in this array, othwerwise this array is all ones
        unmodeled_nbrs_masking_type: string indicating which type of masking to use (default: 'nbrs-seg')
            
        If total=True is sent, then nbr_imgs is just a single image and nbr_masks is just a single array.
        
        If nbrs cannot be rendered, None is returned.
        """
        
        res = self._get_nbrs_data(cen_id,band,cutout_index)
        if res is None:
            return None
        cen_ind, cen_flags, cen_psf_gmix, cen_jac, nbrs_inds, nbrs_flags, nbrs_psf_gmixes, nbrs_jacs, fit_data = res
        
        pars_tag = '%s_max_pars' % model
        fit_flags_tag = '%s_max_flags' % model
        
        if model == 'cm':
            fracdev_tag = 'cm_fracdev'
            TdByTe_tag = 'cm_TdByTe'
        else:
            fracdev_tag=None
            TdByTe_tag=None
            
        img_shape = segmap.shape

        res = self._render_nbrs(model, band, img_shape,
                                cen_ind, cen_psf_gmix, cen_jac, segmap,
                                nbrs_inds, nbrs_flags,
                                nbrs_jacs, nbrs_psf_gmixes,
                                pars_tag, fit_flags_tag, fit_data,
                                unmodeled_nbrs_masking_type=unmodeled_nbrs_masking_type,
                                verbose=verbose,
                                fracdev_tag=fracdev_tag,TdByTe_tag=TdByTe_tag)
        
        if total:
            cen_img, nbrs_imgs, nbrs_masks = res
            
            nbrs_img = numpy.zeros(img_shape,dtype='f8')
            for img in nbrs_imgs:
                if img is not None:
                    nbrs_img += img
    
            nbr_mask = numpy.ones(img_shape,dtype='f8')
            for msk in nbrs_masks:
                if msk is not None:
                    nbr_mask *= msk 
                    
            return cen_img, nbr_img, nbr_mask
        else:
            return res
        
    def _get_nbrs_data(self,cen_id,band,cutout_index):
        """
        return cen and nbrs info for a given cen_id
        
        Parameters
        ----------
        cen_id: id of central object
        band: integer for band (e.g., 0 for griz MOF)
        cutout_index: index of epoch in MEDS file for nbrs
        
        Returns
        -------
        cen_ind: index of central intp returned fit_data
        cen_flags: flags for central from nbrs_data
        cen_psf_gmix: an ngmix GMix for PSF of central
        cen_jac: an ngmix Jacobian for the central
        nbrs_inds: list of indexes into returned fit_data of nbrs
        nbrs_flags: flags of nbrs (only use flags == 0)
        nbrs_psf_gmixes: a list of ngmix GMixes of the PSF models for the nbrs
        nbrs_jacs: a list of ngmix Jacobians for the nbrs
        fit_data: entries of input fit_data corresponding to cen_ind and nbrs_inds
        
        will return None if thete is an error or if not all quantities are found
        """
        
        fit_inds = []
        
        # get cen stuff
        q, = numpy.where(self.fit_data['id'] == cen_id)
        if len(q) != 1:
            return None        
        
        cen_ind = 0        
        fit_inds.append(q[0])
        
        q, = numpy.where((self.nbrs_data['id'] == cen_id) 
                         & (self.nbrs_data['nbr_id'] == cen_id) 
                         & (self.nbrs_data['band'] == band) 
                         & (self.nbrs_data['cutout_index'] == cutout_index))
        if len(q) != 1:
            return None
        
        ind = q[0]
        cen_flags = self.nbrs_data['nbr_flags'][ind]
        if cen_flags == 0:
            cen_jac = Jacobian(self.nbrs_data['nbr_jac_row0'][ind],
                               self.nbrs_data['nbr_jac_col0'][ind],
                               self.nbrs_data['nbr_jac_dudrow'][ind],
                               self.nbrs_data['nbr_jac_dudcol'][ind],
                               self.nbrs_data['nbr_jac_dvdrow'][ind],
                               self.nbrs_data['nbr_jac_dvdcol'][ind])
            cen_psf_gmix = GMix(pars=self.nbrs_data['nbr_psf_fit_pars'][ind,:])
        else:
            cen_psf_gmix = None
            cen_jac = None

        # get nbr ids
        q, = numpy.where((self.nbrs_data['id'] == cen_id) 
                         & (self.nbrs_data['nbr_id'] != cen_id) 
                         & (self.nbrs_data['band'] == band) 
                         & (self.nbrs_data['cutout_index'] == cutout_index))
        if len(q) == 0:
            return None
        
        assert len(numpy.unique(self.nbrs_data['nbr_id'][q])) == len(self.nbrs_data['nbr_id'][q]), \
            "nbrs list for given band %d and cutout_index %d for object %d is not unique!" % (band,cutout_index,cen_id)
        
        nbrs_inds = [i+1 for i in xrange(len(q))]
        
        # find location of nbrs in fit_data
        for nbr_id in self.nbrs_data['nbr_id'][q]:
            qq, = numpy.where(self.fit_data['id'] == nbr_id)
            if len(qq) != 0:
                return cen_ind,[],self.fit_data[fit_inds]
            fit_inds.append(qq[0])            

        # doing fit data
        fit_data = self.fit_data[fit_inds]
        
        # build flag, psf and jacobian lists
        nbrs_flags = []
        nbrs_psf_gmixes = []
        nbrs_jacs = []
        for i,ind in enumerate(q):
            assert fit_data['id'][i+1] == self.nbrs_data['nbr_id'][ind],"Matching of nbr_ids to ids in fit_data did not work!"
            nbrs_flags.append(self.nbrs_data['nbr_flags'][ind])
            
            if nbrs_flags[-1] == 0:
                nbrs_jac.append(Jacobian(self.nbrs_data['nbr_jac_row0'][ind],
                                         self.nbrs_data['nbr_jac_col0'][ind],
                                         self.nbrs_data['nbr_jac_dudrow'][ind],
                                         self.nbrs_data['nbr_jac_dudcol'][ind],
                                         self.nbrs_data['nbr_jac_dvdrow'][ind],
                                         self.nbrs_data['nbr_jac_dvdcol'][ind]))
                nbrs_psf_gmixes.append(GMix(pars=self.nbrs_data['nbr_psf_fit_pars'][ind,:]))
            else:
                nbrs_jac.append(None)
                nbrs_psf_gmixes.append(None)
        
        return cen_ind, cen_flags, cen_psf_gmix, cen_jac, nbrs_inds, nbrs_flags, nbrs_psf_gmixes, nbrs_jacs, fit_data

    @staticmethod
    def _render_nbrs(model, band, img_shape,
                     cen_ind, cen_psf_gmix, cen_jac, cen_seg,
                     nbrs_inds, nbrs_flags,
                     nbrs_jacs, nbrs_psf_gmixes,
                     pars_tag, fit_flags_tag, nbrs_fit_data,
                     unmodeled_nbrs_masking_type='nbrs-seg',
                     verbose=True,
                     fracdev_tag=None,TdByTe_tag=None):
        """
        render or mask nbrs around a central object given a set of nbr flags, jacobians and PSF GMixes
        
        Parameters
        ----------
        model: string indicating which ngmix model (e.g., 'cm' for cmodel, 'exp', etc.)
        band: integer for which band (depends on input fit_data, for griz g=0, r=1, etc.)
        img_shape: tuple with image shape (e.g., (48,48), should be same as box_size in MEDS)
        cen_ind: index of central object in nbrs_fit_data
        cen_psf_gmix: an ngmix GMix representing the PSF for the central object (could be None)
        cen_jac: an ngmix Jacobian for the central object
        cen_seg: numpy array with (probably coadd) seg map around central
        nbrs_inds: list indexes of nbrs in nbrs_fit_data
        nbrs_flags: list flags indicating whether or not to render a nbr (other cuts may also be made),
            list should have the nbrs in the as they are in nbrs_inds
        nbrs_jacs: ngmix Jacobians of nbrs (should have the nbrs in the same order as they are in nbrs_inds)
        nbrs_psf_gmixes: ngmix GMixes for the PSFs of nbrs (should have the nbrs in the same order as 
            they are in nbrs_inds)
        pars_tag: string inidcating which tag in nbrs_fit_data to use to render the nbrs
        fit_flags_tags: fit flags corresponding to pars_tag (i.e., for pars_tag = 'cm_pars', 
            fit_flags_tag = 'cm_flags')
        unmodeled_nbrs_masking_type: string indicating which type of masking to use (default: 'nbrs-seg')           
        
        Optional Parameters
        -------------------
        verbose: bool indicating if the code should tell you things that happen (default: True)
        fracdev_tag: tag to use for fracdev if model == 'cm' (default: None)
        TdByTe_tag: tag to use for TdByTed if model == 'cm' (default: None)
        
        Returns
        -------
        cen_img: image of central (if passes all flags and cuts below) 
        nbr_imgs: list of images of the nbrs (if they pass all flags and cuts, otherwise None)
        nbr_masks: list of nbrs masks, if any nbr does not have an image in nbr_imgs, then its pixels 
            to be masked (according to unmodeled_nbrs_masking_type, see _mask_nbrs_seg) are set to 0 
            in this array, othwerwise this array is all ones
        """

        assert pars_tag in nbrs_fit_data.dtype.names, "pars_tag '%s' is not in nbrs_fit_data!" % pars_tag
        assert fit_flags_tag in nbrs_fit_data.dtype.names, "fit_flags_tag '%s' is not in nbrs_fit_data!" % fit_flags_tag
        assert 'flags' in nbrs_fit_data.dtype.names, "'flags' is not in nbrs_fit_data!"
        
        if fracdev_tag is not None:
            assert fracdev_tag in nbrs_fit_data.dtype.names, "fracdev_tag '%s' is not in nbrs_fit_data!" % fracdev_tag
            assert TdByTe_tag in nbrs_fit_data.dtype.names, "TdByTe_tag '%s' is not in nbrs_fit_data!" % TdByTe_tag
        
        # do central image first
        if (nbrs_fit_data[fit_flags_tag][cen_ind] == 0
            and cen_psf_gmix is not None
            and cen_jac is not None
            and nbrs_fit_data['flags'][cen_ind] == 0):
            
            if verbose:
                print('        rendered central')

            cen_img = RenderNGmixNbrs._render_single(model, band, img_shape,
                                                   pars_tag,
                                                   nbrs_fit_data[cen_ind:cen_ind+1],
                                                   cen_psf_gmix, cen_jac,
                                                   fracdev_tag=fracdev_tag,TdByTe_tag=TdByTe_tag)
        else:
            cen_img = None

            if verbose:
                print('        central not rendered')
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
                
        # now do nbrs
        nbrs_imgs = []
        nbrs_masks = []
        for nbr_ind,nbr_flags,nbr_psf_gmix,nbr_jac in zip(nbrs_inds,
                                                          nbrs_flags,
                                                          nbrs_psf_gmixes,
                                                          nbrs_jacs):
            if (nbr_flags == 0
                and nbrs_fit_data[fit_flags_tag][nbr_ind] == 0
                and nbrs_fit_data['flags'][nbr_ind] == 0
                and nbr_psf_gmix is not None
                and nbr_jac is not None):

                print('        rendered nbr: %d' % (nbr_ind+1))
                
                curr_nbrsim = RenderNGmixNbrs._render_single(model, band, img_shape,
                                                             pars_tag, 
                                                             nbrs_fit_data[nbr_ind:nbr_ind+1],
                                                             nbr_psf_gmix, nbr_jac,
                                                             fracdev_tag=fracdev_tag, TdByTe_tag=TdByTe_tag)
                nbrs_imgs.append(curr_nbrsim)
                nbrs_masks.append(numpy.ones(img_shape))
                
            else:
                if verbose:
                    print('        masked nbr: %d' % (nbr_ind+1))
                    
                msk = numpy.ones(img_shape)
                RenderNGmixNbrs._mask_nbr_seg(cen_seg,
                                              nbrs_fit_data['number'][nbr_ind],
                                              msk,
                                              unmodeled_nbrs_masking_type=unmodeled_nbrs_masking_type)
                nbrs_imgs.append(None)
                nbrs_masks.append(msk)
                
                if verbose:
                    if nbr_flags != 0:
                        print('        nbrs_flags set to %d: FoF obj = %d' % (nbr_flags,nbr_ind+1))
                    elif nbrs_fit_data[fit_flags_tag][nbr_ind] != 0 or (nbrs_fit_data['flags'][nbr_ind] & GAL_FIT_FAILURE) != 0:
                        print('        bad fit data for nbr: FoF obj = %d' % (nbr_ind+1))
                    elif (nbrs_fit_data['flags'][nbr_ind] & LOW_PSF_FLUX) != 0:
                        print('        PSF flux too low for nbr: FoF obj = %d' % (nbr_ind+1))
                    elif (nbrs_fit_data['flags'][nbr_ind] & PSF_FLUX_FIT_FAILURE) != 0:
                        print('        bad PSF flux fit for nbr: FoF obj = %d' % (nbr_ind+1))
                    elif (nbrs_fit_data['flags'][nbr_ind] & PSF_FIT_FAILURE) != 0:
                        print('        bad PSF fit for nbr: FoF obj = %d' % (nbr_ind+1))
                    elif not nbr_psf_gmix is None:
                        print('        no PSF fit data for nbr: FoF obj = %d' % (nbr_ind+1))
                    else:
                        print('        nbr not rendered for unknown reason: FoF obj = %d' % (nbr_ind+1))

        return cen_img, nbrs_imgs, nbrs_masks
        
    @staticmethod
    def _render_single(model,band,img_shape,pars_tag,fit_data,psf_gmix,jac,fracdev_tag=None,TdByTe_tag=None):
        """
        render a single image of model for band with pars_tag in fit_data and psf_gmix w/ jac
        
        Parameters
        ----------
        model: string indicating which ngmix model (e.g., 'cm' for cmodel, 'exp', etc.)
        band: integer for which band (depends on input fit_data, for griz g=0, r=1, etc.)
        img_shape: tuple with image shape (e.g., (48,48), should be same as box_size in MEDS)
        pars_tag: which pars in fit_data to use to render the image (e.g., 'cm_max_pars')
        fit_data: numpy recarray with ngmix outputs
        psf_gmix: an ngmix GMix with the PSF model in it
        jac: an ngmix Jacobian with the jacobian

        Optional Parameters
        -------------------
        fracdev_tag: tag to use for fracdev if model == 'cm'
        TdByTe_tag: tag to use for TdByTed if model == 'cm'
        
        Returns
        -------
        img: image of the nbr as numpy array
        """
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
                    assert fracdev_tag is not None,"You must give fracdev_tag for model == 'cm'!"
                    assert TdByTe_tag is not None,"You must give TdByTe_tag for model == 'cm'!"
                    gmix_sky = GMixCM(fit_data[fracdev_tag][0],
                                      fit_data[TdByTe_tag][0],
                                      band_pars_obj)
                gmix_image = gmix_sky.convolve(psf_gmix)
            except GMixRangeError:
                print('        setting T=0 for nbr!')
                band_pars_obj[4] = 0.0 # set T to zero and try again
        
        image = gmix_image.make_image(img_shape, jacobian=jac)
        return image
    
    @staticmethod
    def _mask_nbr_seg(seg,nbr_number,masked_pix,unmodeled_nbrs_masking_type='nbrs-seg'):
        """
        mask a nbr in weight map of central using a seg map
        
        the maked_pix array should have all ones and gets set to zero where things are masked
        
        Parameters
        ----------
        seg: seg map as numpy array
        nbr_number: number in seg map of nbr
        masked_pix: numpy array of same shape as seg map to record where things get masked
        unmodeled_nbrs_masking_type: string indicating which type of masking to use, can be
            'nbrs-seg' - mask pixels in seg map marked as the nbr
        """

        if unmodeled_nbrs_masking_type == 'nbrs-seg':
            q = numpy.where(seg == nbr_number)
            if q[0].size > 0:
                masked_pix[q] = 0
        else:
            raise ValueError("no support for unmodeled nbrs "
                             "masking type %s" % unmodeled_nbrs_masking_type)
        
