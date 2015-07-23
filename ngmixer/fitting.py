#!/usr/bin/env python
from __future__ import print_function
import logging
import numpy
import time

# local imports
from . import imageio
from .defaults import NO_ATTEMPT,DEFVAL,LOGGERNAME,_CHECKPOINTS_DEFAULT_MINUTES

# logging
log = logging.getLogger(LOGGERNAME)

class NGMixER(dict):
    def __init__(self,
                 conf,
                 files,
                 random_seed=None,
                 checkpoint_file=None,
                 checkpoint_data=None):
        
        # parameters
        self.update(conf)
        self._set_some_defaults()
        
        # read the data
        imageio_class = imageio.get_imageio_class(self['imageio_type'])
        self.imageio = imageio_class(conf,files)
        self.curr_fofindex = 0
        self['nband'] = self.imageio.get_num_bands()
        
        # random numbers
        if random_seed is not None:
            numpy.random.seed(random_seed)

        # checkpointing
        self.checkpoint_file = checkpoint_file
        self.checkpoint_data = checkpoint_data        
        self._setup_checkpoints()
        
        # build data
        if self.checkpoint_data is None:
            self.data = []
            self.data_dtype = self._get_dtype()
            self.epoch_data = []
            self.epoch_data_dtype = self._get_epoch_dtype()

    def _set_some_defaults(self):
        self['work_dir'] = self.get('work_dir','.')
                
    def get_data(self):
        return numpy.array(self.data,dtype=self.data_dtype)

    def get_epoch_data(self):
        return numpy.array(self.epoch_data,dtype=self.epoch_data_dtype)

    def do_fits(self):
        """
        Fit all objects in our list
        """

        t0=time.time()
        num = 0
        numtot = self.imageio.get_num_fofs()
        
        for coadd_mb_obs_lists,mb_obs_lists in self.imageio:
            log.info('index: %d:%d' % (self.curr_fofindex+1,self.numtot))

            foflen = len(mb_obs_lists)            
            for coadd_mb_obs_list,mb_obs_list in zip(coadd_mb_obs_lists,mb_obs_list):
                if foflen > 1:
                    log.info('    fof obj: %d:%d' % (num,foflen))

                num += 1
                ti = time.time()
                self.fit_obj(coadd_mb_obs_list,mb_obs_list)
                ti = time.time()-ti
                log.info('    time:',ti)

            self.curr_fofindex += 1
            
            tm=time.time()-t0                
            self._try_checkpoint(tm)
            
        tm=time.time()-t0
        log.info("time:",tm)
        log.info("time per:",tm/num)

    def fit_obj(self,coadd_mb_obs_list,mb_obs_list):
        """
        fits a single object
        """
        raise NotImplementedError("fit_obj of NGMixER must be defined in subclass.")
        
    def _get_epoch_dtype(self):
        """
        makes epoch dtype
        """
        dt = self.imageio.get_epoch_meta_data_dtype()
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
        return dt
    
    def _make_struct(self,num=1):
        """
        make an output structure
        """
        dt = self._get_dtype()
        data=numpy.zeros(num, dtype=dt)
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
            self.epoch_data = self.checkpoint_data['epoch_data']
            fitsio.fitslib.array_to_native(self.epoch_data, inplace=True)
            self.epoch_data_dtype = self.epoch_data.dtype.descr
            self.epoch_data = list(self.epoch_data)

            # checkpoint data
            rs = cPickle.loads(self.checkpoint_data['checkpoint_data']['random_state'][0])
            numpy.random.set_state(rs)
            self.curr_fofindex = self.checkpoint_data['checkpoint_data']['curr_fofindex'][0]
            self.imageio.set_fof_start(self.curr_fofindex)
            
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
        
        log.info('checkpointing at',tm/60,'minutes')
        log.info(self.checkpoint_file)

        # make checkpoint data        
        cd = numpy.zeros(1,dtype=[('curr_fofindex','i8'),('random_state','|S16384')])
        cd['curr_fofindex'][0] = self.curr_fofindex
        cd['random_state'][0] = cPickle.dumps(numpy.random.get_state())
        
        with StagedOutFile(self.checkpoint_file, tmpdir=self['work_dir']) as sf:
            with fitsio.FITS(sf.path,'rw',clobber=True) as fobj:                
                fobj.write(numpy.array(self.data,dtype=self.data_dtype), extname="model_fits")
                fobj.write(numpy.array(self.epoch_data,dtype=self.epoch_data_dtype), extname="epoch_data")
                fobj.write(cd, extname="checkpoint_data")
        

