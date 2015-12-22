#!/usr/bin/env python
from __future__ import print_function
import os
import numpy
import copy
import fitsio

# meds and ngmix imports
import meds

# local imports
from .medsio import MEDSImageIO

PSF_IND_FIELD='ind_psf'
PSF_IM_FIELD='psf_im'

class SimpSimMEDSImageIO(MEDSImageIO):
    def _set_defaults(self):
        super(SimpSimMEDSImageIO, self)._set_defaults()
        self.conf['psf_ind_field'] = self.conf.get('psf_ind_field',PSF_IND_FIELD)
        self.conf['psf_im_field'] = self.conf.get('psf_im_field',PSF_IM_FIELD)
        self.conf['psfs_in_file'] = self.conf.get('psfs_in_file',False)

    def _load_psf_data(self):
        if not self.conf['psfs_in_file']:
            if 'psf_file' in self.extra_data:
                self.psf_file = self.extra_data['psf_file']
            else:
                pth,bname = os.path.split(self.meds_files_full[0])
                bname = bname.replace('meds','psf')
                self.psf_file = os.path.join(pth,bname)
            print('psf file: %s' % self.psf_file)
            self.psf_data = fitsio.read(self.psf_file)

    def _get_psf_image(self, band, mindex, icut):
        """
        Get an image representing the psf
        """
        
        if self.conf['psfs_in_file']:
            im = self.meds_list[band].get_psf(mindex,icut)
            pfile = self.meds_files[band]
        else:
            meds=self.meds_list[band]
            psf_ind_field=self.conf['psf_ind_field']
            
            ind_psf = meds[psf_ind_field][mindex,icut]
            
            psf_im_field=self.conf['psf_im_field']
            im = self.psf_data[psf_im_field][ind_psf].copy()
            
            pfile = self.psf_file

        im /= im.sum()
        cen = ( numpy.array(im.shape) - 1.0)/2.0
        sigma_pix = 2.5
            
        return im, cen, sigma_pix, pfile
