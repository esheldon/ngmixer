#!/usr/bin/env python
from __future__ import print_function
import logging
import numpy
import time

# local imports
from . import imageio
from . import fitting
from .defaults import DEFVAL,LOGGERNAME,_CHECKPOINTS_DEFAULT_MINUTES
from .defaults import NO_ATTEMPT,NO_CUTOUTS,BOX_SIZE_TOO_BIG,IMAGE_FLAGS
from .util import UtterFailure

# logging
log = logging.getLogger(LOGGERNAME)

class NGMixER(dict):
    def __init__(self,
                 conf,
                 files,
                 fof_data=None,
                 extra_data=None,
                 random_seed=None,
                 checkpoint_file=None,
                 checkpoint_data=None):
        
        # parameters
        self.update(conf)
        
        # read the data
        imageio_class = imageio.get_imageio_class(self['imageio_type'])
        self.imageio = imageio_class(self,files,fof_data=fof_data,extra_data=extra_data)

        # default pars
        self['work_dir'] = self.get('work_dir','.')
        self['make_plots'] = self.get('make_plots',False)
        self['fit_coadd_galaxy'] = self.get('fit_coadd_galaxy',False)
        self['fit_me_galaxy'] = self.get('fit_me_galaxy',True)
        self['max_box_size']=self.get('max_box_size',2048)
        self.curr_fofindex = 0
        self['nband'] = self.imageio.get_num_bands()
        self.iband = range(self['nband'])

        # get the fitter
        fitter_class = fitting.get_fitter_class(self['fitter_type'])
        self.fitter = fitter_class(self)
        self.default_data = self.fitter.get_default_fit_data(self['fit_me_galaxy'],self['fit_coadd_galaxy'])
        self.default_epoch_data = self.fitter.get_default_epoch_fit_data()

        # random numbers
        if random_seed is not None:
            numpy.random.seed(random_seed)

        # checkpointing
        self.start_fofindex = 0
        self.checkpoint_file = checkpoint_file
        self.checkpoint_data = checkpoint_data        
        self._setup_checkpoints()
        
        # build data
        if self.checkpoint_data is None:
            self.data = []
            self.data_dtype = self._get_dtype()
            self.epoch_data = []
            self.epoch_data_dtype = self._get_epoch_dtype()

    def get_data(self):
        return numpy.array(self.data,dtype=self.data_dtype)

    def get_epoch_data(self):
        return numpy.array(self.epoch_data,dtype=self.epoch_data_dtype)

    def get_file_meta_data(self):
        return self.imageio.get_file_meta_data()
    
    def do_fits(self):
        """
        Fit all objects in our list
        """

        log.info('doing fits')
        
        t0=time.time()
        num = 0
        numtot = self.imageio.get_num_fofs()

        log.info('index: %d:%d' % (self.curr_fofindex+1-self.start_fofindex,numtot))
        for coadd_mb_obs_lists,mb_obs_lists in self.imageio:            
            foflen = len(mb_obs_lists)            
            for coadd_mb_obs_list,mb_obs_list in zip(coadd_mb_obs_lists,mb_obs_lists):
                if foflen > 1:
                    log.info('    fof obj: %d:%d' % (num,foflen))

                num += 1
                ti = time.time()
                self.fit_obj(coadd_mb_obs_list,mb_obs_list)
                ti = time.time()-ti
                log.info('    time: %f' % ti)

            self.curr_fofindex += 1
            
            tm=time.time()-t0                
            self._try_checkpoint(tm)

            if self.curr_fofindex < numtot:
                log.info('index: %d:%d' % (self.curr_fofindex+1,numtot))
            
        tm=time.time()-t0
        log.info("time: %f" % tm)
        log.info("time per: %f" % (tm/num))

    def fit_obj(self,coadd_mb_obs_list,mb_obs_list):
        """
        fit a single object
        """

        t0 = time.time()

        # get data to fill
        self.curr_data = self._make_struct()
        for tag in self.default_data.dtype.names:
            self.curr_data[tag] = self.default_data[tag]
        self.curr_data_index = 0

        # get the box size
        box_size = self._get_box_size(mb_obs_list)
        if box_size < 0:
            box_size = self._get_box_size(coadd_mb_obs_list)
        self.curr_data['box_size'][self.curr_data_index] = box_size
        log.info('    box_size: %d' % self.curr_data['box_size'][self.curr_data_index])

        # check flags
        flags = 0
        if box_size < 0:
            flags = UTTER_FAILURE

        if flags == 0:
            flags |= self._obj_check(coadd_mb_obs_list)
        if flags == 0:
            flags |= self._obj_check(mb_obs_list)
        
        if flags == 0:
            fit_flags = self.fit_all_obs_lists(coadd_mb_obs_list,mb_obs_list)
            flags |= fit_flags
        
        # add in data
        self.curr_data['flags'][self.curr_data_index] = flags
        self.curr_data['time'][self.curr_data_index] = time.time()-t0        

        # fill in from mb_obs_meta
        for tag in mb_obs_list.meta['meta_data'].dtype.names:
            self.curr_data[tag][self.curr_data_index] = mb_obs_list.meta['meta_data'][tag][0]

        self.data.extend(list(self.curr_data))

    def _fill_epoch_data(self,mb_obs_list):
        # fill in epoch data
        for band,obs_list in enumerate(mb_obs_list):
            for obs in obs_list:
                if 'fit_data' in obs.meta and obs.meta['fit_data'] is not None \
                   and 'meta_data' in obs.meta:
                    ed = self._make_epoch_struct()
                    for tag in self.default_epoch_data.dtype.names:
                        ed[tag] = self.default_epoch_data[tag]

                    for tag in obs.meta['fit_data'].dtype.names:
                        ed[tag] = obs.meta['fit_data'][tag][0]
                        
                    for tag in obs.meta['meta_data'].dtype.names:
                        ed[tag] = obs.meta['meta_data'][tag][0]
                        
                    self.epoch_data.extend(list(ed))
                            
    def fit_all_obs_lists(self,coadd_mb_obs_list,mb_obs_list):
        """
        fit all obs lists
        """

        fit_flags = None

        if self['fit_me_galaxy']:
            log.info('    fitting me galaxy')
            try:
                me_fit_flags = self.fitter(mb_obs_list,coadd=False)

                # fill in epoch data
                self._fill_epoch_data(mb_obs_list)
                
                # fill in fit data
                for tag in mb_obs_list.meta['fit_data'].dtype.names:
                    self.curr_data[tag][self.curr_data_index] = mb_obs_list.meta['fit_data'][tag][0]
                
            except UtterFailure as err:
                log.info("    me fit got utter failure error: %s" % str(err))
                me_fit_flags = UTTER_FAILURE

            if fit_flags is None:
                fit_flags = 0
            fit_flags |= me_fit_flags

        if self['fit_coadd_galaxy']:
            log.info('    fitting coadd galaxy')
            try:
                coadd_fit_flags = self.fitter(coadd_mb_obs_list,coadd=True)

                # fill in epoch data
                self._fill_epoch_data(coadd_mb_obs_list)
                
                # fill in fit data
                for tag in mb_obs_list.meta['fit_data'].dtype.names:
                    self.curr_data[tag][self.curr_data_index] = coadd_mb_obs_list.meta['fit_data'][tag][0]
                
            except UtterFailure as err:
                log.info("    coadd fit got utter failure error: %s" % str(err))
                coadd_fit_flags = UTTER_FAILURE

            if fit_flags is None:
                fit_flags = 0
                fit_flags |= coadd_fit_flags

        if fit_flags is None:
            fit_flags = NO_ATTEMPT
        
        return fit_flags

    def _get_box_size(self, mb_obs_list):
        box_size = DEFVAL
        for band,obs_list in enumerate(mb_obs_list):
            for obs in obs_list:
                if obs.meta['flags'] == 0:
                    box_size = obs.image.shape[0]
                    break
        return box_size
    
    def _obj_check(self, mb_obs_list):
        """
        Check box sizes, number of cutouts
        Require good in all bands
        """
        for band,obs_list in enumerate(mb_obs_list):
            flags=self._obj_check_one(band,obs_list)
            if flags != 0:
                break
        return flags

    def _obj_check_one(self, band, obs_list):
        """
        Check box sizes, number of cutouts, flags on images
        """
        flags=0
        
        ncutout = len(obs_list)
        
        if ncutout == 0:
            log.info('    no cutouts')
            flags |= NO_CUTOUTS
            return flags
        
        num_use = 0
        for obs in obs_list:
            if obs.meta['flags'] == 0:
                num_use += 1
                
        if num_use < ncutout:
            log.info("    for band %d removed %d/%d images due to flags"
                     % (band, ncutout-num_use, ncutout))
            
        if num_use == 0:
            flags |= IMAGE_FLAGS
            return flags

        box_size = self.curr_data['box_size'][self.curr_data_index]
        if box_size > self['max_box_size']:
            log.info('    box size too big: %d' % box_size)
            flags |= BOX_SIZE_TOO_BIG
            
        return flags
    
    def _get_epoch_dtype(self):
        """
        makes epoch dtype
        """
        dt = self.imageio.get_epoch_meta_data_dtype()
        dt += self.fitter.get_epoch_fit_data_dtype()
        return dt
    
    def _make_epoch_struct(self,ncutout=1):
        """
        returns ncutout epoch structs to be filled
        """
        dt = self._get_epoch_dtype()
        epoch_data = numpy.zeros(ncutout, dtype=dt)
        return epoch_data
    
    def _get_dtype(self):
        dt = self.imageio.get_meta_data_dtype()
        dt += [('flags','i4'),
               ('time','f8'),
               ('box_size','i2')]
        dt += self.fitter.get_fit_data_dtype(self['fit_me_galaxy'],self['fit_coadd_galaxy'])
        return dt
    
    def _make_struct(self,num=1):
        """
        make an output structure
        """
        dt = self._get_dtype()
        data=numpy.zeros(num, dtype=dt)
        data['flags'] = NO_ATTEMPT
        data['time'] = DEFVAL
        data['box_size'] = DEFVAL
        return data
    
    def _setup_checkpoints(self):
        """
        Set up the checkpoint times in minutes and data

        self.checkpoint_data and self.checkpoint_file
        """
        self.checkpoints = self.get('checkpoints',_CHECKPOINTS_DEFAULT_MINUTES)
        self.n_checkpoint    = len(self.checkpoints)
        self.checkpointed    = [0]*self.n_checkpoint

        self._set_checkpoint_data()

        if self.checkpoint_file is not None:
            self.do_checkpoint=True
        else:
            self.do_checkpoint=False

    def _set_checkpoint_data(self):
        """
        See if checkpoint data was sent
        """
        import fitsio
        import cPickle
        
        if self.checkpoint_data is not None:
            # data
            self.data = self.checkpoint_data['data']
            fitsio.fitslib.array_to_native(self.data, inplace=True)
            self.data_dtype = self.data.dtype.descr
            self.data = list(self.data)
            
            # for nband==1 the written array drops the arrayness
            if self['nband']==1:
                raise ValueError("fix for 1 band")

            # epoch data
            if 'epoch_data' in self.checkpoint_data:
                self.epoch_data = self.checkpoint_data['epoch_data']
                fitsio.fitslib.array_to_native(self.epoch_data, inplace=True)
                self.epoch_data_dtype = self.epoch_data.dtype.descr
                self.epoch_data = list(self.epoch_data)
            else:
                self.epoch_data_dtype = self._get_epoch_dtype()
                self.epoch_data = []

            # checkpoint data
            rs = cPickle.loads(self.checkpoint_data['checkpoint_data']['random_state'][0])
            numpy.random.set_state(rs)
            self.curr_fofindex = self.checkpoint_data['checkpoint_data']['curr_fofindex'][0]
            self.imageio.set_fof_start(self.curr_fofindex)
            self.start_fofindex = self.checkpoint_data['checkpoint_data']['curr_fofindex'][0]
            
    def _try_checkpoint(self, tm):
        """
        Checkpoint at certain intervals.  
        Potentially modified self.checkpointed
        """

        should_checkpoint, icheck = self._should_checkpoint(tm)

        if should_checkpoint:
            self._write_checkpoint(tm)
            self.checkpointed[icheck]=1

    def _should_checkpoint(self, tm):
        """
        Should we write a checkpoint file?
        """

        should_checkpoint=False
        icheck=-1

        if self.do_checkpoint:
            tm_minutes=tm/60

            for i in xrange(self.n_checkpoint):

                checkpoint=self.checkpoints[i]
                checkpointed=self.checkpointed[i]

                if tm_minutes > checkpoint and not checkpointed:
                    should_checkpoint=True
                    icheck=i

        return should_checkpoint, icheck

    def _write_checkpoint(self, tm):
        """
        Write out the current data structure to a temporary
        checkpoint file.
        """
        import fitsio
        from .files import StagedOutFile
        import cPickle
        
        log.info('checkpointing at %f minutes' % (tm/60))
        log.info(self.checkpoint_file)

        # make checkpoint data        
        cd = numpy.zeros(1,dtype=[('curr_fofindex','i8'),('random_state','|S16384')])
        cd['curr_fofindex'][0] = self.curr_fofindex
        cd['random_state'][0] = cPickle.dumps(numpy.random.get_state())
        
        with StagedOutFile(self.checkpoint_file, tmpdir=self['work_dir']) as sf:
            with fitsio.FITS(sf.path,'rw',clobber=True) as fobj:                
                fobj.write(numpy.array(self.data,dtype=self.data_dtype), extname="model_fits")
                if len(self.epoch_data) > 0:
                    fobj.write(numpy.array(self.epoch_data,dtype=self.epoch_data_dtype), extname="epoch_data")
                fobj.write(cd, extname="checkpoint_data")
        

