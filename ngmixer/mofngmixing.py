#!/usr/bin/env python
from __future__ import print_function
import logging
import numpy
import time
import pprint
import os
import fitsio
import copy

# local imports
from . import imageio
from . import fitting
from . import files
from .ngmixing import NGMixer
from .defaults import DEFVAL,LOGGERNAME,_CHECKPOINTS_DEFAULT_MINUTES
from .defaults import NO_ATTEMPT,NO_CUTOUTS,BOX_SIZE_TOO_BIG,IMAGE_FLAGS
from .util import UtterFailure
from ngmix import print_pars

# logging
log = logging.getLogger(LOGGERNAME)

class MOFNGMixer(NGMixer):
    def _check_convergence(self,foflen):
        """
        check convergence of fits
        """
        
        dlevel = 0
        
        npars = 5+self['nband']
        maxabs = numpy.zeros(npars,dtype='f8')
        maxabs[:] = -numpy.inf
        maxfrac = numpy.zeros(npars,dtype='f8')
        maxfrac[:] = -numpy.inf
        maxerr = numpy.zeros(npars,dtype='f8')
        maxerr[:] = -numpy.inf
        
        log.info('convergence:')
        for fofind in xrange(foflen):
            log.info('    fof obj: %ld' % fofind)

            #FIXME - mofngmixer doesn't know what tags to use - prob get from fitter
            for model in ['coadd_cm','cm']:
                if model+'_pars' not in self.curr_data.dtype.names:
                    continue
                
                old = self.prev_data[model+'_pars'][fofind]
                new = self.curr_data[model+'_pars'][fofind]
                absdiff = numpy.abs(new-old)
                absfracdiff = numpy.abs(new/old-1.0)
                abserr = numpy.abs((old-new)/numpy.sqrt(numpy.diag(self.curr_data[model+'_pars_cov'][fofind])))
                
                for i in xrange(npars):
                    if absdiff[i] > maxabs[i]:
                        maxabs[i] = copy.copy(absdiff[i])
                    if absfracdiff[i] > maxfrac[i]:
                        maxfrac[i] = copy.copy(absfracdiff[i])
                    if abserr[i] > maxerr[i]:
                        maxerr[i] = copy.copy(abserr[i])
                
                log.info('        %s:' % model)
                print_pars(old,        front='            old      ')
                print_pars(new,        front='            new      ')
                print_pars(absdiff,    front='            abs diff ')
                print_pars(absfracdiff,front='            frac diff')
                print_pars(abserr,front='            err diff ')
        
                    
        fmt = "%8.3g "*len(maxabs)
        log.info("    max abs diff : "+fmt % tuple(maxabs))
        log.info("    max frac diff: "+fmt % tuple(maxfrac))
        log.info("    max err diff : "+fmt % tuple(maxerr))
        
        self.maxabs = maxabs
        self.maxfrac = maxfrac
        self.maxerr = maxerr
        
        if numpy.all((maxabs <= self['mof_maxabs_conv']) | (maxfrac <= self['mof_maxfrac_conv']) | (maxerr <= self['mof_maxerr_conv'])):
            return True
        else:
            return False

    def do_fits(self):
        """
        Fit all objects in our list
        """

        self.done = False
        
        log.info('doing fits')
        
        t0=time.time()
        num = 0
        numfof = 0
        numtot = self.imageio.get_num_fofs()
        
        log.info('fof index: %d:%d' % (self.curr_fofindex+1-self.start_fofindex,numtot))
        for coadd_mb_obs_lists,mb_obs_lists in self.imageio:            
            numfof += 1

            foflen = len(mb_obs_lists)            
            log.info('    num in fof: %d' % foflen)

            # deal with plots
            if self['make_plots']:
                make_plots_at_end = True
                self['make_plots'] = False
            else:
                make_plots_at_end = False
            
            # get data to fill
            self.curr_data = self._make_struct(num=foflen)
            for tag in self.default_data.dtype.names:
                self.curr_data[tag][:] = self.default_data[tag]

            # fit the fof once with no nbrs
            self.curr_data_index = 0
            for coadd_mb_obs_list,mb_obs_list in zip(coadd_mb_obs_lists,mb_obs_lists):
                if foflen > 1:
                    log.info('fof obj: %d:%d' % (self.curr_data_index+1,foflen))
                log.info('    id: %d' % mb_obs_list.meta['id'])

                num += 1
                ti = time.time()
                self.fit_obj(coadd_mb_obs_list,mb_obs_list,nbrs_fit_data=None)
                ti = time.time()-ti
                log.info('    time: %f' % ti)

                self.curr_data_index += 1

            if make_plots_at_end:
                self['make_plots'] = True
            
            # now fit again with nbrs - simple for now
            for itr in xrange(self['mof_max_itr']):
                log.info('itr %d:' % itr)
                
                # data
                self.prev_data = self.curr_data.copy()

                # fitting
                for i in numpy.random.choice(foflen,size=foflen,replace=False):
                    self.curr_data_index = i                                    
                    coadd_mb_obs_list = coadd_mb_obs_lists[i]
                    mb_obs_list = mb_obs_lists[i]
                    if foflen > 1:
                        log.info('fof obj: %d:%d' % (self.curr_data_index+1,foflen))
                    log.info('    id: %d' % mb_obs_list.meta['id'])

                    num += 1
                    ti = time.time()
                    self.fit_obj(coadd_mb_obs_list,mb_obs_list,nbrs_fit_data=self.curr_data)
                    ti = time.time()-ti
                    log.info('    time: %f' % ti)
                    
                if itr >= 1 and self._check_convergence(foflen):
                    break

            # append data and incr.
            self.data.extend(list(self.curr_data))
            self.curr_fofindex += 1
            
            tm=time.time()-t0                
            self._try_checkpoint(tm)
            
            if self.curr_fofindex < numtot:
                log.info('fof index: %d:%d' % (self.curr_fofindex+1,numtot))
            
        tm=time.time()-t0
        log.info("time: %f" % tm)
        log.info("time per fit: %f" % (tm/num))
        log.info("time per fof: %f" % (tm/numfof))

        self.done = True

    def fit_obj(self,coadd_mb_obs_list,mb_obs_list,nbrs_fit_data=None):
        """
        fit a single object
        """
        
        t0 = time.time()

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
            fit_flags = self.fit_all_obs_lists(coadd_mb_obs_list,mb_obs_list,nbrs_fit_data=nbrs_fit_data)
            flags |= fit_flags
        
        # add in data
        self.curr_data['flags'][self.curr_data_index] = flags
        self.curr_data['time'][self.curr_data_index] = time.time()-t0        

        # fill in from mb_obs_meta
        for tag in mb_obs_list.meta['meta_data'].dtype.names:
            self.curr_data[tag][self.curr_data_index] = mb_obs_list.meta['meta_data'][tag][0]

    def fit_all_obs_lists(self,coadd_mb_obs_list,mb_obs_list,nbrs_fit_data=None):
        """
        fit all obs lists
        """

        fit_flags = None

        if self['fit_me_galaxy']:
            log.info('    fitting me galaxy')
            try:
                me_fit_flags = self.fitter(mb_obs_list,coadd=False,nbrs_fit_data=nbrs_fit_data,make_plots=self['make_plots'])

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
                coadd_fit_flags = self.fitter(coadd_mb_obs_list,coadd=True,nbrs_fit_data=nbrs_fit_data,make_plots=self['make_plots'])

                # fill in epoch data
                self._fill_epoch_data(coadd_mb_obs_list)
                
                # fill in fit data
                for tag in coadd_mb_obs_list.meta['fit_data'].dtype.names:
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

