"""
simple MEDS simulation file
"""
from __future__ import print_function
import os
import numpy
import copy
import fitsio

# meds and ngmix imports
import meds

# local imports
from .medsio import MEDSImageIO


class SimpSimMEDSImageIO(MEDSImageIO):
    def _set_defaults(self):
        super(SimpSimMEDSImageIO, self)._set_defaults()

        pconf=self.conf['imageio']['psfs']
        assert pconf['type'] == 'infile'


        self.conf['center_psf'] = self.conf.get('center_psf',False)

    def _get_psf_image(self, band, mindex, icut):
        """
        Get an image representing the psf
        """
        
        im = self.meds_list[band].get_psf(mindex,icut)
        pfile = self.meds_files[band]

        im /= im.sum()
        cen = ( numpy.array(im.shape) - 1.0)/2.0
        sigma_pix = 2.5
            
        return im, cen, sigma_pix, pfile
