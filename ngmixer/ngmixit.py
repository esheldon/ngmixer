#!/usr/bin/env python
import os
import pprint
import fitsio

# local imports
from . import files

# logging
import logging
from .defaults import LOGGERNAME
log = logging.getLogger(LOGGERNAME)

class NGMixIt(object):
    """
    Command class for running ngmixers
    """

    def __init__(self,
                 conf_file,
                 output_file,
                 data_files,
                 extra_data=None,
                 random_seed=None,
                 fof_file=None,
                 fof_range=None,
                 work_dir='.',
                 profile=False,
                 make_plots=False):

        # set the data
        self.conf_file = conf_file
        self.output_file = output_file
        self.data_files = data_files
        self.fof_file = fof_file
        self.fof_range = fof_range
        self.work_dir = work_dir
        self.profile = profile
        self.make_plots = make_plots
        self.random_seed = random_seed
        self.extra_data = extra_data
        
        #run the code
        self.set_random_seed()
        self.read_config()
        pprint.pprint(self.conf)
        self.set_priors()
        
        self.check_checkpoint()
        self.setup_work_files()
        self.read_fof_data()
        self.read_extra_data()
        if self.profile:
            self.go_profile()
        else:
            self.go()
        self.write_data()
        self.cleanup_checkpoint()

    def set_random_seed(self):
        pass
        
    def read_config(self):
        self.conf = files.read_yaml(self.conf_file)
        self.conf['make_plots'] = self.conf.get('make_plots',self.make_plots)
        self.conf['work_dir'] = self.conf.get('work_dir',self.work_dir)

    def set_priors(self):
        from .priors import set_priors
        set_priors(self.conf)
        
    def check_checkpoint(self):
        """
        See if the code was checkpointed in a previous run
        """
        self.checkpoint_file = self.output_file.replace('.fits','-checkpoint.fits')
        self.checkpoint_data = None
        
        if os.path.exists(self.checkpoint_file):
            self.checkpoint_data={}
            log.info('reading checkpoint data: %s' % self.checkpoint_file)
            with fitsio.FITS(self.checkpoint_file) as fobj:
                self.checkpoint_data['data'] = fobj['model_fits'][:]

                if 'epoch_data' in fobj:
                    self.checkpoint_data['epoch_data']=fobj['epoch_data'][:]

                if 'checkpoint_data' in fobj:
                    self.checkpoint_data['checkpoint_data'] = fobj['checkpoint_data'][:]
        
    def setup_work_files(self):
        pass

    def read_fof_data(self):
        if self.fof_file is not None:
            self.fof_data = fitsio.read(self.fof_file)
        else:
            self.fof_data = None        

    def read_extra_data(self):
        pass

    def get_file_meta_data(self):
        return self.ngmixer.get_file_meta_data()
    
    def go(self):
        from .ngmixing import NGMixER
        self.epoch_data=None

        self.ngmixer = NGMixER(self.conf,
                               self.data_files,
                               fof_data=self.fof_data,
                               extra_data=self.extra_data,
                               random_seed=self.random_seed,
                               checkpoint_file=self.checkpoint_file,
                               checkpoint_data=self.checkpoint_data)
        self.ngmixer.do_fits()

        self.data = self.ngmixer.get_data()
        self.epoch_data = self.ngmixer.get_epoch_data()
        self.meta = self.get_file_meta_data()
        
    def go_profile(config_file, meds_files, out_file, options):
        import cProfile
        import pstats
        
        log.info("doing profile")
        
        cProfile.runctx('self.go()',
                        globals(),locals(),
                        'profile_stats')        
        p = pstats.Stats('profile_stats')
        p.sort_stats('time').print_stats()
        
    def write_data(self):
        """
        write the actual data.  clobber existing
        """
        from .files import StagedOutFile
        work_dir = self.conf['work_dir']
        with StagedOutFile(self.output_file, tmpdir=work_dir) as sf:
            log.info('writing %s' % sf.path)
            with fitsio.FITS(sf.path,'rw',clobber=True) as fobj:
                fobj.write(self.data,extname="model_fits")

                if self.epoch_data is not None:
                    fobj.write(self.epoch_data,extname="epoch_data")

                if self.meta is not None:
                    fobj.write(self.meta,extname="meta_data")

    def cleanup_checkpoint(self):
        """
        if we get this far, we have succeeded in writing the data. We can remove
        the checkpoint file
        """
        if os.path.exists(self.checkpoint_file):
            log.info('removing checkpoint file %s' % self.checkpoint_file)
            os.remove(self.checkpoint_file)

