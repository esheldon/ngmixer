#!/usr/bin/env python
import os
import meds
import numpy

# logging
import logging
from .defaults import LOGGERNAME
log = logging.getLogger(LOGGERNAME)

from .ngmixit import NGMixIt

class MEDSNGMixIt(NGMixIt):
    def read_config(self):
        super(MEDSNGMixIt,self).read_config()
        self.conf['nband'] = len(self.data_files)
        self.conf['imageio_type'] = 'meds'
                
    def _get_sub_fname(self,fname):
        rng_string = '%s-%s' % (self.fof_range[0], self.fof_range[1])
        bname = os.path.basename(fname)
        bname = bname.replace('.fits.fz','').replace('.fits','')
        bname = '%s-%s.fits' % (bname, rng_string)
        newf = os.path.expandvars(os.path.join(self.work_dir, bname))
        return newf

    def _get_sub(self):
        """
        Local files will get cleaned up
        """
        extracted=[]
    
        if self.fof_file is None:
            #FoF data defaults to 1 per object so range is old range
            for f in self.data_files:
                log.info(f)
                newf = self._get_sub_fname(f)
                ex=meds.MEDSExtractor(f, self.fof_range[0], self.fof_range[1], newf, cleanup=True)
                extracted.append(ex)
            extracted.append(None)
        else:
            # do the fofs first
            log.info(self.fof_file)
            newf = self._get_sub_fname(fof_file)
            fofex = nbrsfofs.NbrsFoFExtractor(fof_file, self.fof_range[0], self.fof_range[1], newf, cleanup=True)
            
            # now do the meds
            for f in self.data_files:
                log.info(f)
                newf = self._get_sub_fname(f)
                ex=meds.MEDSNumberExtractor(f, fofex.numbers, newf, cleanup=True)
                extracted.append(ex)
            extracted.append(fofex)
    
        return extracted

    def setup_work_files(self):
        """
        Set up local, possibly sub-range meds files
        """
        self.data_files_full = self.data_files
        self.fof_file_full = self.fof_file
        self.extracted=None
        if self.fof_range is not None:
            # note variable extracted is cleaned up when MedsExtractors get
            # garbage collected
            extracted=self._get_sub()
            meds_files=[ex.sub_file for ex in extracted if ex is not None]
            if extracted[-1] is not None:
                fof_file = extracted[-1].sub_file        
                self.fof_file = fof_file
                meds_files = meds_files[:-1]
            self.data_files = meds_files
            self.extracted = extracted
        
    def get_file_meta_data(self):
        meds_meta_list = self.ngmixer.get_file_meta_data()
        dt = meds_meta_list[0].dtype.descr

        config_file = self.conf_file

        clen=len(config_file)
        flen=max([len(mf) for mf in self.data_files_full] )
        
        dt += [('gmix_meds_config','S%d' % clen),
               ('meds_file','S%d' % flen)]

        nband=len(self.data_files_full)
        meta=numpy.zeros(nband, dtype=dt)

        for band in xrange(nband):            
            meds_file = self.data_files_full[band]
            meds_meta=meds_meta_list[band]
            mnames=meta.dtype.names
            for name in meds_meta.dtype.names:
                if name in mnames:
                    meta[name][band] = meds_meta[name][0]

            meta['gmix_meds_config'][band] = config_file
            meta['meds_file'][band] = meds_file
            
        return meta
    


