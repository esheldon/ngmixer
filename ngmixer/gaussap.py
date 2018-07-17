"""
synthesize gaussian aperture fluxes
"""
from __future__ import print_function
import numpy as np
import ngmix
import esutil as eu
from . import defaults

def add_gauss_aper_flux_cat(cat,
                            model,
                            pixel_scale,
                            weight_fwhm,
                            psf_fwhm=None):
    """
    Measure synthesized gaussian weighted apertures for a simple ngmix model,
    and add entries to the input ngmixer output catalog.  If the entries
    already exist in the catalog they are over-riddent, otherwise new fields
    are dded

    parameters
    ----------
    cat: array
        Output array from an ngmixer run.
    model: string
        e.g. exp,dev,gauss,cm
    pixel_scale: float
        Pixel scale for images
    weight_fwhm: float
        FWHM of the gaussian weight function in the same units as the pixel
        scale.
    psf_fwhm: float, optional
        Size of the small psf with which to convolve the profile. Default
        is the pixel scale.  This psf helps avoid issues with resolution
    """


    output=get_gauss_aper_flux_cat(
        cat=cat,
        model=model,
        pixel_scale=pixel_scale,
        weight_fwhm=weight_fwhm,
        psf_fwhm=psf_fwhm,
    )

    pname='%s_pars' % model
    npars=cat[pname][0].size
    nband=_get_nband(npars,model)

    gap_flux_name, gap_flags_name=_get_names(model)
    extra_dt=[]
    if gap_flux_name not in cat.dtype.names:
        extra_dt += [(gap_flux_name,'f8',nband)]
    if gap_flags_name not in cat.dtype.names:
        extra_dt += [(gap_flags_name,'i4')]

    if len(extra_dt) != 0:
        new_data=eu.numpy_util.add_fields(cat, extra_dt)
    else:
        new_data=cat
    
    new_data[gap_flux_name] = output[gap_flux_name]
    new_data[gap_flags_name] = output[gap_flags_name]
    return new_data

def get_gauss_aper_flux_cat(cat,
                            model,
                            pixel_scale,
                            weight_fwhm,
                            psf_fwhm=None):
    """
    Measure synthesized gaussian weighted apertures for a simple ngmix
    model, for all entries in an ngmixer output catalog

    parameters
    ----------
    cat: array
        Output array from an ngmixer run.
    model: string
        e.g. exp,dev,gauss,cm
    pixel_scale: float
        Pixel scale for the images.
    weight_fwhm: float
        FWHM of the weight function in the same units as the
        pixel scale.
    psf_fwhm: float, optional
        Size of the small psf with which to convolve the profile. Default
        is the pixel scale.  This psf helps avoid issues with resolution
    """

    pname='%s_pars' % model
    if model=='cm':
        fracdev_name='%s_fracdev' % model
        TdByTe_name='%s_TdByTe' % model
        gapmeas=GaussAperCM(
            pixel_scale=pixel_scale,
            weight_fwhm=weight_fwhm,
            psf_fwhm=psf_fwhm,
        )
    else:
        gapmeas=GaussAperSimple(
            pixel_scale=pixel_scale,
            weight_fwhm=weight_fwhm,
            psf_fwhm=psf_fwhm,
        )

    gap_flux_name, gap_flags_name=_get_names(model)

    tpars=cat[pname][0]
    nband=len(tpars)-6+1
    dt=[
        (gap_flags_name,'i4'),
        (gap_flux_name,'f8',nband),
    ]
    output=np.zeros(cat.size, dtype=dt)
    output[gap_flags_name] = defaults.NO_ATTEMPT
    output[gap_flux_name] = defaults.DEFVAL

    for i in range(cat.size):
        if ((i+1) % 10) == 0:
            print("%d/%d" % (i+1,cat.size))

        if cat['flags'][i] != 0:
            continue

        try:
            bsize=cat['box_size'][i]
            dims=[bsize,bsize]

            if model=='cm':
                output[gap_flux_name][i] = gapmeas.get_aper_flux(
                    cat[fracdev_name][i],
                    cat[TdByTe_name][i],
                    cat[pname][i],
                    dims,
                )
            else:
                output[gap_flux_name][i] = gapmeas.get_aper_flux(
                    cat[pname][i],
                    model,
                    dims,
                )

        except ngmix.GMixRangeError as err:
            print(str(err))
            output[gap_flags_name][i] = defaults.GAL_FIT_FAILURE

    return output


class GaussAperSimple(object):
    def __init__(self, pixel_scale, weight_fwhm, psf_fwhm=None):
        """
        measure synthesized gaussian weighted apertures for a simple ngmix
        model

        parameters
        ----------
        pixel_scale: float
            Pixel scale for images.
        weight_fwhm: float
            FWHM of the weight function in the same units as the
            pixel scale.
        psf_fwhm: float, optional
            Size of the small psf with which to convolve the profile. Default
            is the pixel scale.  This psf helps avoid issues with resolution
        """

        self.pixel_scale=pixel_scale
        self._set_weight(weight_fwhm)

        if psf_fwhm is None:
            psf_fwhm=pixel_scale
        self._set_psf(psf_fwhm)

    def get_aper_flux(self, pars, model, dims):
        """
        get the aperture flux for the specified parameters and model

        parameters
        ----------
        pars: 6-element sequence
            The parameters [cen1,cen2,g1,g2,T,flux1,flux2,...]
            Note multiple bands may be present. The center is
            not used, the model will always be centered in
            the image.
        model: string
            Simple model such as gauss,exp,dev
        dims: 2-element sequence
            Dimensions of the images to simulate.

        returns
        -------
        fluxes: array
            gaussian weighted fluxes in the aperture with length
            nband
        """

        assert len(dims)==2

        nband=_get_nband(len(pars),model)

        fluxes=np.zeros(nband)-9999.0

        jacobian=self._get_jacobian(dims)
        for band in range(nband):
            band_pars=self._get_band_pars(pars, band)
            fluxes[band] = self._get_flux(
                band_pars,
                model,
                dims,
                jacobian,
            )

        return fluxes

    def _get_default_flux(self):
        return np.zeros(nband)-9999.0

    def _get_flux(self, pars, model, dims, jacobian):
        gm=self._get_gmix(pars)
        im = gm.make_image(dims, jacobian=jacobian)
        wtim = self._get_weight_image(dims, jacobian)
        return self._do_get_flux(im, wtim)

    def _do_get_flux(self, im, wtim):

        weighted_im = wtim*im
        flux = weighted_im.sum()*self.pixel_scale**2
        return flux

    def _get_gmix(self, pars):
        gm0 = ngmix.GMixModel(pars, model)
        gm = gm0.convolve(self.psf)
        return gm

    def _get_band_pars(self, pars, band):
        band_pars=np.zeros(6)
        band_pars[0:5] = pars[0:5]
        band_pars[5] = pars[5+band]
        return band_pars

    def _get_jacobian(self, dims):
        cen=(np.array(dims)-1.0)/2.0
        return ngmix.DiagonalJacobian(
            row=cen[0],
            col=cen[1],
            scale=self.pixel_scale,
        )

    def _set_weight(self, weight_fwhm):
        self.weight_fwhm=weight_fwhm
        weight=self._get_gauss_model(weight_fwhm)
        self.weight=weight

    def _set_psf(self, psf_fwhm):
        self.psf_fwhm=psf_fwhm
        self.psf=self._get_gauss_model(psf_fwhm)

    def _get_gauss_model(self, fwhm):
        sigma = ngmix.moments.fwhm_to_sigma(fwhm)
        T = 2*sigma**2

        pars = [
            0.0,
            0.0,
            0.0,
            0.0,
            T,
            1.0,
        ]
        return ngmix.GMixModel(pars, "gauss")


class GaussAperCM(GaussAperSimple):
    def get_aper_flux(self, fracdev, TdByTe, pars, dims):
        """
        get the aperture flux for the specified parameters and model

        parameters
        ----------
        fracdev: float
            Fraction of light in the dev component
        TdByTe: float
            Ratio of bulge T to disk T Tdev/Texp
        pars: 6-element sequence
            The parameters [cen1,cen2,g1,g2,T,flux1,flux2,...]
            Note multiple bands may be present. The center is
            not used, the model will always be centered in
            the image.
        dims: 2-element sequence
            Dimensions of the images to simulate.

        returns
        -------
        fluxes: array
            gaussian weighted fluxes in the aperture with length
            nband
        """

        assert len(dims)==2

        nband=_get_nband(len(pars),"cm")

        fluxes=np.zeros(nband)-9999.0

        jacobian=self._get_jacobian(dims)
        for band in range(nband):
            band_pars=self._get_band_pars(pars, band)
            fluxes[band] = self._get_flux(
                fracdev,
                TdByTe,
                band_pars,
                dims,
                jacobian,
            )

        return fluxes

    def _get_flux(self, fracdev, TdByTe, pars, dims, jacobian):
        gm=self._get_gmix(fracdev, TdByTe, pars)
        im = gm.make_image(dims, jacobian=jacobian)
        wtim = self._get_weight_image(dims, jacobian)
        return self._do_get_flux(im, wtim)

    def _get_weight_image(self, dims, jacobian):
        wtim = self.weight.make_image(dims, jacobian=jacobian)
        wtim *= 1.0/wtim.max()
        return wtim

    def _get_gmix(self, fracdev, TdByTe, pars):
        gm0=ngmix.gmix.GMixCM(
            fracdev,
            TdByTe,
            pars,
        )

        gm = gm0.convolve(self.psf)
        return gm

def _get_names(model):
    gap_flux_name='%s_gap_flux' % model
    gap_flags_name='%s_gap_flags' % model
    return gap_flux_name,gap_flags_name

def _get_nband(npars, model):
    nband = npars-6+1
    if nband < 1:
        raise ValueError("pars must be length "
                         "5+nband, got %d" % len(pars))

    return nband


