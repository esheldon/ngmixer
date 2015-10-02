#!/usr/bin/env python
from __future__ import print_function
import os
import numpy
import copy
import fitsio

from .medsio import MEDSImageIO
from .. import nbrsfofs
from ..util import print_with_verbosity

import meds

# flagging
IMAGE_FLAGS_SET=2**0

# SVMEDS
class SVDESMEDSImageIO(MEDSImageIO):

    def get_file_meta_data(self):
        meds_meta_list = self.meds_meta_list
        dt = meds_meta_list[0].dtype.descr

        if 'config_file' in self.conf:
            tmp,config_file = os.path.split(self.conf['config_file'])
            clen=len(config_file)
            dt += [('ngmixer_config','S%d' % clen)]

        flen=max([len(mf.replace(os.environ['DESDATA'],'${DESDATA}')) for mf in self.meds_files_full] )
        dt += [('meds_file','S%d' % flen)]

        mydesdata = os.environ['DESDATA']
        dt += [('ngmixer_DESDATA','S%d' % len(mydesdata))]

        nband=len(self.meds_files_full)
        meta=numpy.zeros(nband, dtype=dt)

        for band in xrange(nband):
            meds_file = self.meds_files_full[band]
            meds_meta=meds_meta_list[band]
            mnames=meta.dtype.names
            for name in meds_meta.dtype.names:
                if name in mnames:
                    meta[name][band] = meds_meta[name][0]

            if 'config_file' in self.conf:
                meta['ngmixer_config'][band] = config_file
            meta['meds_file'][band] = meds_file.replace(os.environ['DESDATA'],'${DESDATA}')
            meta['ngmixer_DESDATA'][band] = mydesdata

        return meta

    def _get_image_flags(self, band, mindex):
        """
        find images associated with the object and get the image flags
        Also add in the psfex flags, eventually incorporated into meds
        """
        meds=self.meds_list[band]
        ncutout=meds['ncutout'][mindex]

        file_ids = meds['file_id'][mindex, 0:ncutout]
        image_flags = self.all_image_flags[band][file_ids]

        return image_flags

    def _get_meds_orig_filename(self, meds, mindex, icut):
        """
        Get the original filename
        """
        file_id=meds['file_id'][mindex, icut]
        ii=meds.get_image_info()
        return ii['image_path'][file_id]

    def _get_band_observations(self, band, mindex):
        coadd_obs_list, obs_list = super(SVDESMEDSImageIO, self)._get_band_observations(band,mindex)

        # divide by jacobian scale^2 in order to apply zero-points correctly
        for olist in [coadd_obs_list,obs_list]:
            for obs in olist:
                if obs.meta['flags'] == 0:
                    pixel_scale2 = obs.jacobian.get_det()
                    pixel_scale4 = pixel_scale2*pixel_scale2
                    obs.image /= pixel_scale2
                    obs.weight *= pixel_scale4
                    if obs.weight_raw is not None:
                        obs.weight_raw *= pixel_scale4
                    if obs.weight_us is not None:
                        obs.weight_raw *= pixel_scale4

        return coadd_obs_list, obs_list

    def get_epoch_meta_data_dtype(self):
        dt = super(SVDESMEDSImageIO, self).get_epoch_meta_data_dtype()
        dt += [('image_id','i8')]  # image_id specified in meds creation, e.g. for image table
        return dt

    def _fill_obs_meta_data(self,obs, band, mindex, icut):
        """
        fill meta data to be included in output files
        """
        super(SVDESMEDSImageIO, self)._fill_obs_meta_data(obs, band, mindex, icut)
        meds=self.meds_list[band]
        file_id  = meds['file_id'][mindex,icut].astype('i4')
        image_id = meds._image_info[file_id]['image_id']
        obs.meta['meta_data']['image_id'][0]  = image_id

    def _load_psf_data(self):
        self.psfex_lists = self._get_psfex_lists()

    def _get_psf_image(self, band, mindex, icut):
        """
        Get an image representing the psf
        """

        meds=self.meds_list[band]
        file_id=meds['file_id'][mindex,icut]

        pex=self.psfex_lists[band][file_id]

        row=meds['orig_row'][mindex,icut]
        col=meds['orig_col'][mindex,icut]

        im=pex.get_rec(row,col).astype('f8', copy=False)
        cen=pex.get_center(row,col)
        sigma_pix=pex.get_sigma()

        return im, cen, sigma_pix, pex['filename']

    def _get_psfex_lists(self):
        """
        Load psfex objects for each of the SE images
        include the coadd so we get  the index right
        """
        print('loading psfex')
        desdata=os.environ['DESDATA']
        meds_desdata=self.meds_list[0]._meta['DESDATA'][0]

        psfex_lists=[]
        for band in self.iband:
            meds=self.meds_list[band]

            psfex_list = self._get_psfex_objects(meds,band)
            psfex_lists.append( psfex_list )

        return psfex_lists

    def _psfex_path_from_image_path(self, meds, image_path):
        """
        infer the psfex path from the image path.
        """
        desdata=os.environ['DESDATA']
        meds_desdata=meds._meta['DESDATA'][0]

        psfpath=image_path.replace('.fits.fz','_psfcat.psf')
        if desdata not in psfpath:
            psfpath=psfpath.replace(meds_desdata,desdata)

        if self.conf['use_psf_rerun'] and 'coadd' not in psfpath:
            psfparts=psfpath.split('/')
            psfparts[-6] = 'EXTRA' # replace 'OPS'
            psfparts[-3] = 'psfex-rerun/%s' % self.conf['psf_rerun_version'] # replace 'red'
            psfpath='/'.join(psfparts)

        return psfpath

    def _get_psfex_objects(self, meds, band):
        """
        Load psfex objects for all images, including coadd
        """
        from psfex import PSFExError, PSFEx

        psfex_list=[]

        info=meds.get_image_info()
        nimage=info.size
        for i in xrange(nimage):
            pex=None

            # don't even bother if we are going to skip this image
            flags = self.all_image_flags[band][i]
            if (flags & self.conf['image_flags2check']) == 0:

                impath=info['image_path'][i].strip()
                psfpath = self._psfex_path_from_image_path(meds, impath)

                if not os.path.exists(psfpath):
                    print("warning: missing psfex: %s" % psfpath)
                    self.all_image_flags[band][i] |= self.conf['image_flags2check']
                else:
                    print_with_verbosity("loading: %s" % psfpath,verbosity=2)
                    try:
                        pex=PSFEx(psfpath)
                    except PSFExError as err:
                        print("problem with psfex file: %s " % str(err))
                        pex=None

            psfex_list.append(pex)

        return psfex_list

    def _get_replacement_flags(self, filenames):
        from .util import CombinedImageFlags

        if not hasattr(self,'_replacement_flags'):
            fname=os.path.expandvars(self.conf['replacement_flags'])
            print("reading replacement flags: %s" % fname)
            self._replacement_flags=CombinedImageFlags(fname)

        default=self.conf['image_flags2check']
        return self._replacement_flags.get_flags_multi(filenames,default=default)

    def _load_meds_files(self):
        """
        Load all listed meds files
        We check the flags indicated by image_flags2check.  the saved
        flags are 0 or IMAGE_FLAGS_SET
        """

        self.meds_list=[]
        self.meds_meta_list=[]
        self.all_image_flags=[]

        for i,funexp in enumerate(self.meds_files):
            f = os.path.expandvars(funexp)
            print('band %d meds: %s' % (i,f))
            medsi=meds.MEDS(f)
            medsi_meta=medsi.get_meta()
            image_info=medsi.get_image_info()

            if i==0:
                nobj_tot=medsi.size
            else:
                nobj=medsi.size
                if nobj != nobj_tot:
                    raise ValueError("mismatch in meds "
                                     "sizes: %d/%d" % (nobj_tot,nobj))
            self.meds_list.append(medsi)
            self.meds_meta_list.append(medsi_meta)
            image_flags=image_info['image_flags'].astype('i8')

            if 'replacement_flags' in self.conf and self.conf['replacement_flags'] is not None and image_flags.size > 1:
                print("    replacing image flags")
                image_flags[1:] = \
                    self._get_replacement_flags(image_info['image_path'][1:])

            # now we reduce the flags to zero or IMAGE_FLAGS_SET
            # copy out and check image flags just for cutouts
            cimage_flags=image_flags[1:].copy()
            w,=numpy.where( (cimage_flags & self.conf['image_flags2check']) != 0)
            print("    flags set for: %d/%d" % (w.size,cimage_flags.size))
            cimage_flags[:] = 0
            if w.size > 0:
                cimage_flags[w] = IMAGE_FLAGS_SET

            # copy back in reduced flags
            image_flags[1:] = cimage_flags
            self.all_image_flags.append(image_flags)

        self.nobj_tot = self.meds_list[0].size



# SV multifit with one-off WCS
class MOFSVDESMEDSImageIO(SVDESMEDSImageIO):
    def __init__(self,conf,meds_files):
        super(MOFSVDESMEDSImageIO,self).__init__(conf,meds_files)

        read_wcs = self.config.get('read_wcs',False)
        if read_wcs:
            self.wcs_transforms = self._get_wcs_transforms()

    def _get_wcs_transforms(self):
        """
        Load the WCS transforms for each meds file
        """
        import json
        from esutil.wcsutil import WCS

        print('loading WCS')
        wcs_transforms = {}
        for band in self.iband:
            mname = self.conf['meds_files_full'][band]
            wcsname = mname.replace('-meds-','-meds-wcs-').replace('.fits.fz','.fits').replace('.fits','.json')
            print('loading: %s' % wcsname)
            try:
                with open(wcsname,'r') as fp:
                    wcs_list = json.load(fp)
            except:
                assert False,"WCS file '%s' cannot be read!" % wcsname

            wcs_transforms[band] = []
            for hdr in wcs_list:
                wcs_transforms[band].append(WCS(hdr))

        return wcs_transforms

# alias for now
Y1DESMEDSImageIO = SVDESMEDSImageIO
