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
        
        models_to_check,cov_models_to_check = self.fitter.get_models_for_checking()
        if self['fit_coadd_galaxy']:
            coadd_models_to_check = ['coadd_'+modl for modl in models_to_check]
            models_to_check.extend(coadd_models_to_check)

            coadd_cov_models_to_check = ['coadd_'+modl for modl in cov_models_to_check]
            cov_models_to_check.extend(coadd_cov_models_to_check)
        
        npars = 5+self['nband']
        maxabs = numpy.zeros(npars,dtype='f8')
        maxabs[:] = -numpy.inf
        maxfrac = numpy.zeros(npars,dtype='f8')
        maxfrac[:] = -numpy.inf
        maxerr = numpy.zeros(npars,dtype='f8')
        maxerr[:] = -numpy.inf
        
        for fofind in xrange(foflen):
            log.debug('    fof obj: %ld' % fofind)

            for model,model_cov in zip(models_to_check,cov_models_to_check):
                if model not in self.curr_data.dtype.names:
                    continue

                if self.curr_data['flags'][fofind] or self.prev_data['flags'][fofind]:
                    log.info('    skipping fof obj %s in convergence check' % (fofind+1))
                    continue
                
                old = self.prev_data[model][fofind]
                new = self.curr_data[model][fofind]
                absdiff = numpy.abs(new-old)
                absfracdiff = numpy.abs(new/old-1.0)
                abserr = numpy.abs((old-new)/numpy.sqrt(numpy.diag(self.curr_data[model_cov][fofind])))
                
                for i in xrange(npars):
                    if absdiff[i] > maxabs[i]:
                        maxabs[i] = copy.copy(absdiff[i])
                    if absfracdiff[i] > maxfrac[i]:
                        maxfrac[i] = copy.copy(absfracdiff[i])
                    if abserr[i] > maxerr[i]:
                        maxerr[i] = copy.copy(abserr[i])
                
                log.debug('        %s:' % model)
                if log.getEffectiveLevel() <= logging.DEBUG:
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

        if numpy.all((maxabs <= self['mof']['maxabs_conv']) | (maxfrac <= self['mof']['maxfrac_conv']) | (maxerr <= self['mof']['maxerr_conv'])):
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

            # get data to fill
            self.curr_data = self._make_struct(num=foflen)
            for tag in self.default_data.dtype.names:
                self.curr_data[tag][:] = self.default_data[tag]

            # fit the fof once with no nbrs
            # sort by stamp size
            bs = []
            for coadd_mb_obs_list,mb_obs_list in zip(coadd_mb_obs_lists,mb_obs_lists):
                box_size = self._get_box_size(mb_obs_list)
                if box_size < 0:
                    box_size = self._get_box_size(coadd_mb_obs_list)
                bs.append(box_size)
            bs = numpy.array(bs)
            q = numpy.argsort(bs)            
            for i in q:
                self.curr_data_index = i                                    
                coadd_mb_obs_list = coadd_mb_obs_lists[i]
                mb_obs_list = mb_obs_lists[i]
                if foflen > 1:
                    log.info('fof obj: %d:%d' % (self.curr_data_index+1,foflen))
                log.info('    id: %d' % mb_obs_list.meta['id'])

                num += 1
                ti = time.time()
                self.fit_obj(coadd_mb_obs_list,mb_obs_list,nbrs_fit_data=None)
                ti = time.time()-ti
                log.info('    time: %f' % ti)

            if foflen > 1:
                # now fit again with nbrs
                converged = False
                for itr in xrange(self['mof']['max_itr']):
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
                    
                    if itr >= self['mof']['min_itr']:
                        log.info('convergence itr %d:' % (itr+1))
                        if self._check_convergence(foflen):
                            converged = True
                            break

                log.info('fof index: %d' % (self.curr_fofindex+1-self.start_fofindex))
                log.info('    converged: %s' % str(converged))
                log.info('    num itr: %d' % (itr+1))
                fmt = "%8.3g "*len(self.maxabs)
                log.info("    max abs diff : "+fmt % tuple(self.maxabs))
                log.info("    max frac diff: "+fmt % tuple(self.maxfrac))
                log.info("    max err diff : "+fmt % tuple(self.maxerr))

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

