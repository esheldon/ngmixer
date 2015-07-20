from __future__ import print_function
import os
import numpy
from numpy import log
import fitsio

import ngmix
from ngmix import srandu, GMixRangeError, print_pars
from ngmix.priors import LOWVAL

def clip_element_wise(arr, minvals, maxvals):
    """
    Clip each element of an array separately
    """
    for i in xrange(arr.size):
        arr[i] = arr[i].clip(min=minvals[i],max=maxvals[i])

class UtterFailure(Exception):
    """
    could not make a good guess
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)


class Namer(object):
    """
    create strings with a specified front prefix
    """
    def __init__(self, front=None):
        self.front=front
    def __call__(self, name):
        if self.front is None or self.front=='':
            return name
        else:
            return '%s_%s' % (self.front, name)

def print_pars_and_logl(pars, logl, fmt='%8.3g', front=None):
    """
    print the parameters with a uniform width
    """
    from sys import stdout
    if front is not None:
        stdout.write(front)
        stdout.write(' ')

    allfmt = ' '.join( [fmt+' ']*len(pars) )
    stdout.write(allfmt % tuple(pars))
    stdout.write(" logl: %g\n" % logl)


def plot_autocorr(trials, window=100, show=False, **kw):
    import biggles
    import emcee

    arr=biggles.FramedArray(trials.shape[1], 1)
    arr.uniform_limits=True

    func=emcee.autocorr.function(trials)
    tau2 = emcee.autocorr.integrated_time(trials, window=window)

    xvals=numpy.arange(func.shape[0])
    zc=biggles.Curve( [0,func.shape[0]-1],[0,0] )

    for i in xrange(trials.shape[1]):
        pts=biggles.Curve(xvals,func[:,i],color='blue')
        
        lab=biggles.PlotLabel(0.9,0.9,
                              r'$%s tau\times 2: %s$' % (i,tau2[i]),
                              halign='right')
        arr[i,0].add(pts,zc,lab)

    if show:
        arr.show(**kw)

    return arr

class CombinedImageFlags(object):
    """
    replacement astrometry flags
    """
    def __init__(self, filename):
        self.filename=filename

        self._load()

    def _load(self):
        import json
        with open(self.filename) as fobj:
            self.data=json.load(fobj)

    def get_key(self, filename):
        bname=os.path.basename(filename)
        bs=bname.split('_')

        expid = int(bs[1])
        ccdid = int( bs[2].split('.')[0] )
        
        key='%s-%02d' % (expid, ccdid)
        return key

    def get_flags_multi(self, image_names, default=0):

        flaglist=numpy.zeros(len(image_names))
        for i,image_name in enumerate(image_names):
            flags=self.get_flags(image_name,default=default)

            flaglist[i] = flags
        
        return flaglist

    def get_flags(self, image_name, default=0):
        """
        match based on image id
        """

        key=self.get_key(image_name)

        flags = self.data.get(key,default)
        return flags


class AstromFlags(object):
    """
    replacement astrometry flags
    """
    def __init__(self, filename):
        self.data=fitsio.read(filename,lower=True)

    def get_flags(self, image_ids):
        """
        match based on image id
        """
        import esutil as eu

        image_ids=numpy.array(image_ids, ndmin=1, dtype='i8',copy=False)

        # default is flagged, to indicated not found
        flags=numpy.ones(image_ids.size,dtype='i8')
        minput, mastro = eu.numpy_util.match(image_ids, self.data['imageid'])

        nmiss=image_ids.size - minput.size 
        if nmiss > 0:
            print("        %d/%d did not "
                  "match astrom flags" % (nmiss,image_ids.size))
        else:
            print("    all matched")



        if minput.size > 0:
            flags[minput] = self.data['astrom_flag'][mastro]

        return flags
