#!/usr/bin/env python
from __future__ import print_function
import logging
import numpy

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
            self.data_dtype = self._get_data_dtype()
            self.epoch_data = []
            self.epoch_data_dtype = self.get_epoch_data_dtype()
        
    def _set_some_defaults(self):
        self['fit_models'] = list(self['model_pars'].keys())
        self['max_box_size']=self.get('max_box_size',2048)
        self['make_plots']=self.get('make_plots',False)
        self['work_dir'] = self.get('work_dir','.')
        self['fit_coadd_galaxy'] = self.get('fit_coadd_galaxy',False)
        self['fit_me_galaxy'] = self.get('fit_me_galaxy',True)

    def get_data(self):
        return numpy.array(self.data,dtype=self.data_dtype)

    def get_epoch_data(self):
        return numpy.array(self.epoch_data,dtype=self.epoch_data_dtype)
        
    def _check_models(self):
        """
        make sure all models are supported
        """
        for model in self['fit_models']:
            if model not in ['exp','dev','cm']:
                raise ValueError("model '%s' not supported" % model)

    def _get_all_models(self):
        """
        get all model names, includeing the coadd_ ones
        """
        models=[]
        
        if self['fit_coadd_galaxy']:
            models = models + ['coadd_%s' % model for model in self['fit_models']]
            
        if self['fit_me_galaxy']:
            models = models + se;f['fit_models']

        return models

    def _get_epoch_dtype(self):
        """
        makes epoch dtype
        """
        psf_ngauss=self['psf_em_pars']['ngauss']
        npars=psf_ngauss*6
        dt=[('npix','i4'),
            ('wsum','f8'),
            ('wmax','f8'),
            ('psf_fit_flags','i4'),
            ('psf_counts','f8'),
            ('psf_fit_g','f8',2),
            ('psf_fit_T','f8'),
            ('psf_fit_pars','f8',npars)]
        dt += self.imageio.get_epoch_meta_data_dtype()

        return dt
    
    def _make_epoch_struct(self,ncutout=1):
        """
        returns ncutout epoch structs to be filled
        """
        epoch_data = numpy.zeros(ncutout, dtype=self.epoch_data_dtype)
        epoch_data['npix'] = DEFVAL
        epoch_data['wsum'] = DEFVAL
        epoch_data['wmax'] = DEFVAL
        epoch_data['psf_counts'] = DEFVAL
        epoch_data['psf_fit_g'] = DEFVAL
        epoch_data['psf_fit_T'] = DEFVAL
        epoch_data['psf_fit_pars'] = DEFVAL
        epoch_data['psf_fit_flags'] = NO_ATTEMPT
        
        return epoch_data
    
    def _get_dtype(self):
        self._check_models()

        nband=self['nband']
        bshape=(nband,)
        simple_npars=5+nband

        dt=[('processed','i1'),
            ('flags','i4'),
            ('nimage_tot','i4',bshape),
            ('nimage_use','i4',bshape),
            ('time','f8'),

            ('box_size','i2'),

            ('coadd_npix','i4'),
            ('coadd_mask_frac','f8'),
            ('coadd_psfrec_T','f8'),
            ('coadd_psfrec_g','f8', 2),

            ('mask_frac','f8'),
            ('psfrec_T','f8'),
            ('psfrec_g','f8', 2)
        ]
        dt += self.imageio.get_meta_data_dtype()
        
        # the psf flux fits are done for each band separately
        for name in ['coadd_psf','psf']:
            n=Namer(name)
            dt += [(n('flags'),   'i4',bshape),
                   (n('flux'),    'f8',bshape),
                   (n('flux_err'),'f8',bshape),
                   (n('chi2per'),'f8',bshape),
                   (n('dof'),'f8',bshape)]

        if nband==1:
            fcov_shape=(nband,)
        else:
            fcov_shape=(nband,nband)

        models=self._get_all_models()
        for model in models:
            n=Namer(model)
            np=simple_npars
            
            dt+=[(n('flags'),'i4'),
                 (n('pars'),'f8',np),
                 (n('pars_cov'),'f8',(np,np)),
                 (n('flux'),'f8',bshape),
                 (n('flux_cov'),'f8',fcov_shape),
                 (n('g'),'f8',2),
                 (n('g_cov'),'f8',(2,2)),
                 
                 (n('max_flags'),'i4'),
                 (n('max_pars'),'f8',np),
                 (n('max_pars_cov'),'f8',(np,np)),                 
                 (n('chi2per'),'f8'),
                 (n('dof'),'f8'),
                 (n('s2n_w'),'f8'),
            ]
            
            if self['do_shear']:
                dt += [(n('g_sens'), 'f8', 2),
                       (n('P'), 'f8'),
                       (n('Q'), 'f8', 2),
                       (n('R'), 'f8', (2,2))]
            
        return dt

    def _make_struct(self,num=1):
        """
        make an output structure
        """
        
        data=numpy.zeros(num, dtype=self.data_dtype)

        data['processed'] = 0
        data['flags'] = NO_ATTEMPT        
        data['nimage_tot'] = DEFVAL
        data['nimage_use'] = DEFVAL
        data['time'] = DEFVAL
        data['box_size'] = DEFVAL

        data['coadd_npix'] = DEFVAL
        data['coadd_mask_frac'] = DEFVAL
        data['coadd_psfrec_T'] = DEFVAL
        data['coadd_psfrec_g'] = DEFVAL

        data['mask_frac'] = DEFVAL
        data['psfrec_T'] = DEFVAL
        data['psfrec_g'] = DEFVAL
        
        for name in ['coadd_psf','psf']:
            n=Namer(name)
            data[n('flags')] = NO_ATTEMPT
            data[n('flux')] = DEFVAL
            data[n('flux_err')] = DEFVAL
            data[n('chi2per')] = DEFVAL
            data[n('dof')] = DEFVAL

        models=self._get_all_models()
        for model in models:
            n=Namer(model)

            data[n('flags')] = NO_ATTEMPT
            
            data[n('pars')] = DEFVAL
            data[n('pars_cov')] = DEFVAL
            data[n('flux')] = DEFVAL
            data[n('flux_cov')] = DEFVAL
            data[n('g')] = DEFVAL
            data[n('g_cov')] = DEFVAL

            data[n('s2n_w')] = DEFVAL
            data[n('chi2per')] = DEFVAL

            data[n('max_flags')] = NO_ATTEMPT
            data[n('max_pars')] = DEFVAL
            data[n('max_pars_cov')] = DEFVAL

            if self['do_shear']:
                data[n('g_sens')] = DEFVAL
                data[n('P')] = DEFVAL
                data[n('Q')] = DEFVAL
                data[n('R')] = DEFVAL
        
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
        

