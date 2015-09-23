#!/usr/bin/env python
from __future__ import print_function
import os
import numpy
import logging
import copy
import fitsio

from .medsio import MEDSImageIO
from ..defaults import LOGGERNAME
from .. import nbrsfofs

# logging
log = logging.getLogger(LOGGERNAME)

# simple alias for SV
SVMEDSImageIO = MEDSImageIO

# SV multifit with one-off WCS
class SVMOFMEDSImageIO(SVMEDSImageIO):
    def __init__(self,conf,meds_files):
        super(SVMOFMEDSImageIO,self).__init__(conf,meds_files)

        read_wcs = self.config.get('read_wcs',False)
        if read_wcs:
            self.wcs_transforms = self._get_wcs_transforms()

    def _get_wcs_transforms(self):
        """
        Load the WCS transforms for each meds file
        """
        import json
        from esutil.wcsutil import WCS

        log.info('loading WCS')
        wcs_transforms = {}
        for band in self.iband:
            mname = self.conf['meds_files_full'][band]
            wcsname = mname.replace('-meds-','-meds-wcs-').replace('.fits.fz','.fits').replace('.fits','.json')
            log.info('loading: %s' % wcsname)
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
Y1MEDSImageIO = MEDSImageIO
