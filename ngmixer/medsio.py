#!/usr/bin/env python
from __future__ import print_function
import os
import numpy
import logging
import copy
import fitsio

# meds and ngmix imports
import meds
from ngmix import Jacobian
from ngmix import Observation, ObsList, MultiBandObsList

# local imports
from .imageio import ImageIO
from .defaults import DEFVAL,IMAGE_FLAGS,LOGGERNAME
from . import nbrsfofs

# internal flagging
IMAGE_FLAGS_SET=2**0

# logging
log = logging.getLogger(LOGGERNAME)

class MEDSImageIO(ImageIO):
    """
    Class for MEDS image I/O.
    """
    
    def __init__(self,*args,**kwargs):
        super(MEDSImageIO, self).__init__(*args,**kwargs)
        
        self.conf = args[0]
        self.conf['min_weight'] = self.conf.get('min_weight',-numpy.inf)
        
        meds_files = args[1]        
        if not isinstance(meds_files, list):
	    self.meds_files = [meds_files]
	else:
	    self.meds_files =  meds_files

        # do sub range files if needed
        self._setup_work_files()
        
	# load meds files and image flags array
	self._load_meds_files()

        # set extra config
        self.iband = range(len(self.meds_list))
	self.conf['nband'] = len(self.meds_list)

        self.conf['reject_outliers'] = self.conf.get('reject_outliers',True) # from cutouts
        if self.conf['reject_outliers']:
            log.info("will reject outliers")

	self._set_and_check_index_lookups()
        
	self.psfex_lists = self._get_psfex_lists()

        # deal with extra data
        if 'extra_data' in kwargs:
            self.extra_data = kwargs['extra_data']            
        else:
            self.extra_data = None

        if self.extra_data is None:
            self.extra_data = {}

        # make sure if we are doing nbrs we have the info we need
        if 'model_nbrs' in self.conf:
            assert 'extra_data' in kwargs            
            assert 'nbrs' in self.extra_data

    def get_file_meta_data(self):
        meds_meta_list = self.meds_meta_list
        dt = meds_meta_list[0].dtype.descr

        if 'config_file' in self.conf:
            clen=len(self.conf['config_file'])
            dt += [('ngmixer_config','S%d' % clen)]
            
        flen=max([len(mf) for mf in self.meds_files_full] )
        dt += [('meds_file','S%d' % flen)]

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
                meta['ngmixer_config'][band] = self.conf['config_file']
            meta['meds_file'][band] = meds_file

        return meta

    def _get_sub_fname(self,fname):
        rng_string = '%s-%s' % (self.fof_range[0], self.fof_range[1])
        bname = os.path.basename(fname)
        bname = bname.replace('.fits.fz','').replace('.fits','')
        bname = '%s-%s.fits' % (bname, rng_string)
        newf = os.path.expandvars(os.path.join(self.conf['work_dir'], bname))
        return newf

    def _get_sub(self):
        """
        Local files will get cleaned up
        """
        extracted=[]

        if self.fof_file is None:
            for f in self.meds_files:
                log.info(f)
                newf = self._get_sub_fname(f)
                ex=meds.MEDSExtractor(f, self.fof_range[0], self.fof_range[1], newf, cleanup=True)
                extracted.append(ex)
            extracted.append(None)
        else:
            # do the fofs first
            log.info(self.fof_file)
            newf = self._get_sub_fname(self.fof_file)
            fofex = nbrsfofs.NbrsFoFExtractor(self.fof_file, self.fof_range[0], self.fof_range[1], newf, cleanup=True)

            # now do the meds
            for f in self.meds_files:
                log.info(f)
                newf = self._get_sub_fname(f)
                ex=meds.MEDSNumberExtractor(f, fofex.numbers, newf, cleanup=True)
                extracted.append(ex)
            extracted.append(fofex)

        return extracted

    def _setup_work_files(self):
        """
        Set up local, possibly sub-range meds files
        """
        self.meds_files_full = self.meds_files
        self.fof_file_full = self.fof_file
        self.extracted=None
        if self.fof_range is not None:
            extracted=self._get_sub()
            meds_files=[ex.sub_file for ex in extracted if ex is not None]
            if extracted[-1] is not None:
                self.fof_file = extracted[-1].sub_file
                meds_files = meds_files[:-1]
            self.meds_files = meds_files
            self.extracted = extracted
        
    def _set_and_check_index_lookups(self):
	"""
	Deal with common indexing issues in one place
	"""

	"""
	Indexing notes:
	
	Each MEDS file consists of a list of objects to be fit, which are indexed by 
	
	   self.mindex = 0-offset from start of file
	   
	The code runs by processing groups of objects, which we call FoFs. Note however
	that these groupings can be arbitrary. The FoFs are specified by a numpy array 
	(read from a FITS table) which has two columns
	
	    fofid - the ID of the fof group, 0-offset
	    number - the ID number of the object in the coadd detection tile seg map
	    
	We require that the FoF file number column matches the MEDS file, line-by-line.
	
	We do however build some lookup tables to enable easy translation. They are
    
	    self.fofid2mindex = this is a dict keyed on the fofid - it returns the set of mindexes
		that give the members of the FoF group
	    self.number2mindex = lookup table for converting numbers to mindexes
	These are both dictionaries which use python's hashing of keys. There may be a performance 
	issue here, in terms of building the dicts, for large numbers of objects.
	"""

        # warn the user
        log.info('making fof indexes')
        
        if self.fof_file is not None:
            read_fofs = True
            self.fof_data = fitsio.read(self.fof_file)
        else:
            read_fofs = False
            nobj = len(self.meds_list[0]['number'])
	    self.fof_data = numpy.zeros(nobj,dtype=[('fofid','i8'),('number','i8')])
            self.fof_data['fofid'][:] = numpy.arange(nobj)
            self.fof_data['number'][:] = self.meds_list[0]['number'][:]
        
	# first, we error check
	for band,meds in enumerate(self.meds_list):
	    assert numpy.array_equal(meds['number'],self.fof_data['number']),"FoF number is not the same as MEDS number for band %d!" % band

	#set some useful stuff here
	self.fofids = numpy.unique(self.fof_data['fofid'])
        self.fofids = numpy.sort(self.fofids)
        self.num_fofs = len(self.fofids)

        if read_fofs:
            #build the fof hash
	    self.fofid2mindex = {}
	    for fofid in self.fofids:
	        q, = numpy.where(self.fof_data['fofid'] == fofid)
	        assert len(q) > 0, 'Found zero length FoF! fofid = %ld' % fofid
                assert numpy.array_equal(self.fof_data['number'][q],self.meds_list[0]['number'][q])
                self.fofid2mindex[fofid] = q.copy()
        else:
            #use a list of lists since is much faster to make
            self.fofid2mindex = []
            for fofid in self.fofids:
                self.fofid2mindex.append([fofid])
                assert self.fofid2mindex[fofid][0] == fofid

    def __next__(self):
        if self.fofindex >= self.num_fofs:
            raise StopIteration
        else:
            fofid = self.fofids[self.fofindex]
            mindexes = self.fofid2mindex[fofid]
            coadd_mb_obs_lists = []
            me_mb_obs_lists = []
            for mindex in mindexes:
                c,me = self._get_multi_band_observations(mindex)
                coadd_mb_obs_lists.append(c)
                me_mb_obs_lists.append(me)

            if 'obj_flags' in self.extra_data:
                self._flag_objects(coadd_mb_obs_lists,me_mb_obs_lists,mindexes)
                
            if self.conf['model_nbrs']:                
                self._add_nbrs_info(coadd_mb_obs_lists,me_mb_obs_lists,mindexes)
                
            self.fofindex += 1
            return coadd_mb_obs_lists,me_mb_obs_lists

    next = __next__

    def _flag_objects(coadd_mb_obs_lists,me_mb_obs_lists,mindexes):
        qnz, = numpy.where(self.extra_data['obj_flags']['flags'] != 0)
        for mindex,coadd_mb_obs_list,me_mb_obs_list in zip(mindexes,coadd_mb_obs_lists,me_mb_obs_lists):
            q, = np.where(self.extra_data['obj_flags']['id'][qnz] == me_mb_obs_list.meta['id'])
            if len(q) > 0:
                assert len(q) == 1
                assert me_mb_obs_list.meta['id'] ==  self.extra_data['obj_flags']['id'][qnz[q[0]]]
                coadd_mb_obs_list.meta['obj_flags'] = self.extra_data['obj_flags']['flags'][qnz[q[0]]]
                me_mb_obs_list.meta['obj_flags'] = self.extra_data['obj_flags']['flags'][qnz[q[0]]]        
    
    def _add_nbrs_info(self,coadd_mb_obs_lists,me_mb_obs_lists,mindexes):
        """
        adds nbr info to obs lists
        """

        # save orig images and weights
        for mb_obs_list in coadd_mb_obs_lists:
            for obs_list in mb_obs_list:
                for obs in obs_list:
                    obs.image_orig = obs.image.copy()
                    obs.weight_orig = obs.weight.copy()

        for mb_obs_list in me_mb_obs_lists:
            for obs_list in mb_obs_list:
                for obs in obs_list:
                    obs.image_orig = obs.image.copy()
                    obs.weight_orig = obs.weight.copy()        
        
        # do indexes
        if len(mindexes) == 1:
            for cen,mindex in enumerate(mindexes):
                nbrs_inds = []
                nbrs_ids = []
                coadd_mb_obs_lists[cen].update_meta_data({'nbrs_inds':nbrs_inds,'nbrs_ids':nbrs_ids})
                me_mb_obs_lists[cen].update_meta_data({'nbrs_inds':nbrs_inds,'nbrs_ids':nbrs_ids})
        else:
            for cen,mindex in enumerate(mindexes):
                nbrs_inds = []
                nbrs_ids = []
                q, = numpy.where(self.extra_data['nbrs']['number'] == self.meds_list[0]['number'][mindex])
                for ind in q:
                    if self.extra_data['nbrs']['nbr_number'][ind] != -1:
                        qi, = numpy.where(self.meds_list[0]['number'][mindexes] == self.extra_data['nbrs']['nbr_number'][ind])
                        assert len(qi) == 1
                        nbrs_inds.append(qi[0])
                        nbrs_ids.append(self.meds_list[0]['id'][mindexes[qi[0]]])
                        assert coadd_mb_obs_lists[nbrs_inds[-1]].meta['id'] == nbrs_ids[-1]
                        assert me_mb_obs_lists[nbrs_inds[-1]].meta['id'] == nbrs_ids[-1]
                
                coadd_mb_obs_lists[cen].update_meta_data({'nbrs_inds':nbrs_inds,'nbrs_ids':nbrs_ids,'cen_ind':cen})
                me_mb_obs_lists[cen].update_meta_data({'nbrs_inds':nbrs_inds,'nbrs_ids':nbrs_ids,'cen_ind':cen})
                assert cen not in nbrs_inds,'weird error where cen_ind is in nbrs_ind!'
                
        # now do psfs and jacobians
        self._add_nbrs_psfs_and_jacs(coadd_mb_obs_lists,mindexes)
        self._add_nbrs_psfs_and_jacs(me_mb_obs_lists,mindexes)        

    def _add_nbrs_psfs_and_jacs(self,mb_obs_lists,mindexes):
        # for each object
        for cen,mindex in enumerate(mindexes):
            # for each band per object
            for band,obs_list in enumerate(mb_obs_lists[cen]):
                # for each obs per band per object
                for obs in obs_list:
                    if obs.meta['flags'] == 0:
                        # for each nbr per obs per band per object
                        nbrs_psfs = []
                        nbrs_flags = []
                        nbrs_jacs = []
                        for ind in mb_obs_lists[cen].meta['nbrs_inds']:
                            psf_obs,jac = self._get_nbr_psf_obs_and_jac(band,cen,mindex,obs,ind,mindexes[ind],mb_obs_lists[ind])
                            nbrs_psfs.append(psf_obs)
                            nbrs_jacs.append(jac)

                            if psf_obs is None or jac is None or mb_obs_lists[ind].meta['obj_flags'] != 0:
                                nbrs_flags.append(1)
                            else:
                                nbrs_flags.append(0)
                        
                        obs.update_meta_data({'nbrs_psfs':nbrs_psfs,'nbrs_flags':nbrs_flags,'nbrs_jacs':nbrs_jacs})

    def _get_nbr_psf_obs_and_jac(self,band,cen_ind,cen_mindex,cen_obs,nbr_ind,nbr_mindex,nbrs_obs_list):
        assert nbrs_obs_list.meta['id'] ==  self.meds_list[band]['id'][nbr_mindex]
        assert cen_obs.meta['id'] ==  self.meds_list[band]['id'][cen_mindex]

        cen_file_id = cen_obs.meta['meta_data']['file_id'][0]
        nbr_obs = None
        for obs in nbrs_obs_list[band]:
            if obs.meta['meta_data']['file_id'][0] == cen_file_id and self.meds_list[band]['id'][nbr_mindex] == obs.meta['id']:
                nbr_obs = obs

        if nbr_obs is not None:
            assert nbr_obs.meta['flags'] == 0
            nbr_psf_obs = nbr_obs.get_psf()
            nbr_icut = nbr_obs.meta['icut']
            assert self.meds_list[band]['file_id'][nbr_mindex,nbr_icut] == cen_file_id
            
            # construct jacobian
            # get fiducial location of object in postage stamp
            row = self.meds_list[band]['orig_row'][nbr_mindex,nbr_icut] - cen_obs.meta['orig_start_row']
            col = self.meds_list[band]['orig_col'][nbr_mindex,nbr_icut] - cen_obs.meta['orig_start_col']
            nbr_jac = Jacobian(row,col,
                               self.meds_list[band]['dudrow'][nbr_mindex,nbr_icut],
                               self.meds_list[band]['dudcol'][nbr_mindex,nbr_icut],
                               self.meds_list[band]['dvdrow'][nbr_mindex,nbr_icut],
                               self.meds_list[band]['dvdcol'][nbr_mindex,nbr_icut])
            # FIXME - this is wrong...I think - commented out for now
            #pixscale = jacob.get_scale()
            #row += pars_obj[0]/pixscale
            #col += pars_obj[1]/pixscale
            #nbr_jac.set_cen(row,col)            

            return nbr_psf_obs,nbr_jac
        else:
            # FIXME
            log.info('    FIXME: off-chip nbr %d for cen %d' % (nbr_ind+1,cen_ind+1))
            return None,None
    
    def get_num_fofs(self):
        return copy.copy(self.num_fofs - self.fof_start)
    
    def get_num_bands(self):
        """"
        returns number of bands for galaxy images
        """
        return copy.copy(self.conf['nband'])
    
    def get_meta_data_dtype(self):
        row = self._get_meta_row()        
        return copy.copy(row.dtype.descr)

    def get_epoch_meta_data_dtype(self):
        row = self._get_epoch_meta_row()
        return copy.copy(row.dtype.descr)
    
    def _get_meta_row(self,num=1):
        # build the meta data
        dt=[('id','i8'),
            ('number','i4'),
            ('nimage_tot','i4',self.conf['nband'])]
        meta_row = numpy.zeros(num,dtype=dt)
        for tag in meta_row.dtype.names:
            meta_row[tag][:] = DEFVAL
        return meta_row
    
    def _get_multi_band_observations(self, mindex):
        """
        Get an ObsList object for the Coadd observations
        Get a MultiBandObsList object for the SE observations.
        """
        
        coadd_mb_obs_list=MultiBandObsList()
        mb_obs_list=MultiBandObsList()
        
        for band in self.iband:
            cobs_list, obs_list = self._get_band_observations(band, mindex)
            coadd_mb_obs_list.append(cobs_list)
            mb_obs_list.append(obs_list)

        meta_row = self._get_meta_row()
        meta_row['id'][0] = self.meds_list[0]['id'][mindex]
        meta_row['number'][0] = self.meds_list[0]['number'][mindex]
        meta_row['nimage_tot'][0,:] = numpy.array([self.meds_list[b]['ncutout'][mindex]-1 for b in xrange(self.conf['nband'])],dtype='i4')
        meta = {'meta_data':meta_row,'meds_index':mindex,'id':self.meds_list[0]['id'][mindex],'obj_flags':0}

        coadd_mb_obs_list.update_meta_data(meta)
        mb_obs_list.update_meta_data(meta)

        return coadd_mb_obs_list, mb_obs_list
    
    def _reject_outliers(self, obs_list):
        imlist=[]
        wtlist=[]
        for obs in obs_list:
            if obs.meta['flags'] == 0:
                imlist.append(obs.image)
                wtlist.append(obs.weight)
        
        # weight map is modified
        nreject=meds.reject_outliers(imlist,wtlist)
        if nreject > 0:
            log.info('    rejected: %d' % nreject)

    def _get_band_observations(self, band, mindex):
        """
        Get an ObsList for the coadd observations in each band
        If psf fitting fails, the ObsList will be zero length
        note we have already checked that we have a coadd and a single epoch
        without flags
        """

        meds=self.meds_list[band]
        ncutout=meds['ncutout'][mindex]

        image_flags=self._get_image_flags(band, mindex)

        coadd_obs_list = ObsList()
        obs_list       = ObsList()

        for icut in xrange(ncutout):
            iflags = image_flags[icut]
            if iflags != 0:
                flags = IMAGE_FLAGS
                obs = Observation(numpy.zeros((0,0)))
            else:
                obs = self._get_band_observation(band, mindex, icut)
                flags=0

            # fill the meta data
            self._fill_obs_meta_data(obs,band,mindex,icut)
            
            # set flags
            meta = {'flags':flags}
            obs.update_meta_data(meta)
            
            if icut==0:
                coadd_obs_list.append(obs)
            else:
                obs_list.append(obs)

        if self.conf['reject_outliers'] and len(obs_list) > 0:
            self._reject_outliers(obs_list)
            
        return coadd_obs_list, obs_list

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

    def _get_epoch_meta_row(self,num=1):
        # build the meta data
        dt=[('id','i8'),     # could be coadd_objects_id
            ('number','i4'), # 1-n as in sextractor
            ('band_num','i2'),
            ('cutout_index','i4'), # this is the index in meds
            ('orig_row','f8'),
            ('orig_col','f8'),
            ('file_id','i4'),   # id in meds file
            ('image_id','i8'),  # image_id specified in meds creation, e.g. for image table
            ]
        meta_row = numpy.zeros(num,dtype=dt)
        for tag in meta_row.dtype.names:
            meta_row[tag][:] = DEFVAL
        return meta_row
    
    def _get_band_observation(self, band, mindex, icut):
        """
        Get an Observation for a single band.
        """
        meds=self.meds_list[band]

        fname = self._get_meds_orig_filename(meds, mindex, icut)
        im = self._get_meds_image(meds, mindex, icut)
        wt,wt_us,seg = self._get_meds_weight(meds, mindex, icut)
        jacob = self._get_jacobian(meds, mindex, icut)

        # for the psf fitting code
        wt=wt.clip(min=0.0)

        psf_obs = self._get_psf_observation(band, mindex, icut, jacob)
        
        obs=Observation(im,
                        weight=wt,
                        jacobian=jacob,
                        psf=psf_obs)
        obs.weight_us = wt_us
        obs.weight_raw = wt
        obs.seg = seg                         
        obs.filename=fname
        
        return obs

    def _fill_obs_meta_data(self,obs, band, mindex, icut):
        """
        fill meta data to be included in output files
        """

        meds=self.meds_list[band]

        meta_row = self._get_epoch_meta_row()
        meta_row['id'][0] = meds['id'][mindex]
        meta_row['number'][0] = meds['number'][mindex]
        meta_row['band_num'][0] = band
        meta_row['cutout_index'][0] = icut
        meta_row['orig_row'][0] = meds['orig_row'][mindex,icut]
        meta_row['orig_col'][0] = meds['orig_col'][mindex,icut]
        file_id  = meds['file_id'][mindex,icut].astype('i4')
        image_id = meds._image_info[file_id]['image_id']
        meta_row['file_id'][0]  = file_id
        meta_row['image_id'][0]  = image_id
        
        meta={'icut':icut,
              'orig_start_row':meds['orig_start_row'][mindex, icut],
              'orig_start_col':meds['orig_start_col'][mindex, icut],
              'meta_data':meta_row,
              'id':meds['id'][mindex],
              'band_id':icut}
        obs.update_meta_data(meta)
        
    def _get_meds_orig_filename(self, meds, mindex, icut):
        """
        Get the original filename
        """
        file_id=meds['file_id'][mindex, icut]
        ii=meds.get_image_info()
        return ii['image_path'][file_id]
    
    def _get_meds_image(self, meds, mindex, icut):
        """
        Get an image cutout from the input MEDS file
        """
        im = meds.get_cutout(mindex, icut)
        im = im.astype('f8', copy=False)
        return im
    
    def _get_meds_weight(self, meds, mindex, icut):
        """
        Get a weight map from the input MEDS file

        if self.conf['region']=='seg_and_sky':
            wt=meds.get_cweight_cutout(mindex, icut)
        elif self.conf['region']=="cweight-nearest":
            wt=meds.get_cweight_cutout_nearest(mindex, icut)
        elif self.conf['region']=='weight':
            wt=meds.get_cutout(mindex, icut, type='weight')
        else:
            raise ValueError("support other region types")
        
        wt=wt.astype('f8', copy=False)
        
        w=numpy.where(wt < self.conf['min_weight'])
        if w[0].size > 0:
            wt[w] = 0.0
        return wt
        """

        wt_us = meds.get_cweight_cutout_nearest(mindex, icut)
        wt_us = wt_us.astype('f8', copy=False)
        w = numpy.where(wt_us < self.conf['min_weight'])
        if w[0].size > 0:
            wt_us[w] = 0.0

        wt = meds.get_cutout(mindex, icut, type='weight')
        wt = wt.astype('f8', copy=False)
        w = numpy.where(wt < self.conf['min_weight'])
        if w[0].size > 0:
            wt[w] = 0.0

        seg = meds.get_cweight_cutout(mindex, icut)

        if self.conf['region']=="cweight-nearest":
            wt = wt_us
        
        return wt,wt_us,seg

    def _convert_jacobian_dict(self, jdict):
        """
        Get the jacobian for the input meds index and cutout index
        """
        jacob=Jacobian(jdict['row0'],
                       jdict['col0'],
                       jdict['dudrow'],
                       jdict['dudcol'],
                       jdict['dvdrow'],
                       jdict['dvdcol'])
        return jacob
    
    def _get_jacobian(self, meds, mindex, icut):
        """
        Get a Jacobian object for the requested object
        """
        jdict = meds.get_jacobian(mindex, icut)
        jacob = self._convert_jacobian_dict(jdict)
        return jacob
    
    def _get_psf_observation(self, band, mindex, icut, image_jacobian):
        """
        Get an Observation representing the PSF and the "sigma"
        from the psfex object
        """
        im, cen, sigma_pix, fname = self._get_psf_image(band, mindex, icut)

        psf_jacobian = image_jacobian.copy()
        psf_jacobian.set_cen(cen[0], cen[1])

        psf_obs = Observation(im, jacobian=psf_jacobian)
        psf_obs.filename=fname

        # convert to sky coords
        sigma_sky = sigma_pix*psf_jacobian.get_scale()

        psf_obs.update_meta_data({'sigma_sky':sigma_sky})
        psf_obs.update_meta_data({'Tguess':sigma_sky*sigma_sky})
        
        return psf_obs

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
            
    def _get_replacement_flags(self, filenames):
	from .util import CombinedImageFlags

	if not hasattr(self,'_replacement_flags'):
	    fname=os.path.expandvars(self.conf['replacement_flags'])
	    log.info("reading replacement flags: %s" % fname)
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
            log.info('band %d meds: %s' % (i,f))
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
	    if self.conf['replacement_flags'] is not None and image_flags.size > 1:
                log.info("    replacing image flags")
                image_flags[1:] = \
		    self._get_replacement_flags(image_info['image_path'][1:])

	    # now we reduce the flags to zero or IMAGE_FLAGS_SET
	    # copy out and check image flags just for cutouts
	    cimage_flags=image_flags[1:].copy()

	    w,=numpy.where( (cimage_flags & self.conf['image_flags2check']) != 0)

	    log.info("    flags set for: %d/%d" % (w.size,cimage_flags.size))

	    cimage_flags[:] = 0
	    if w.size > 0:
		cimage_flags[w] = IMAGE_FLAGS_SET

	    # copy back in reduced flags
	    image_flags[1:] = cimage_flags
	    self.all_image_flags.append(image_flags)

	self.nobj_tot = self.meds_list[0].size

    def _get_psfex_lists(self):
	"""
	Load psfex objects for each of the SE images
	include the coadd so we get  the index right
	"""
	log.info('loading psfex')
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
		    log.info("warning: missing psfex: %s" % psfpath)
		    self.all_image_flags[band][i] |= self.conf['image_flags2check']
		else:
		    log.debug("loading: %s" % psfpath)
		    try:
			pex=PSFEx(psfpath)
		    except PSFExError as err:
			log.info("problem with psfex file: %s " % str(err))
			pex=None
    
	    psfex_list.append(pex)

	return psfex_list


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
Y1MEDSImageIO = MEDSImageIO
