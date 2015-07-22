#!/usr/bin/env python
from .__init__ import IMAGEIO

def get_imageio_class(ftype):
    """
    returns the imageio class for a given a ftype
    """
    
    cftype = ftype.upper()
    assert cftype in IMAGEIO,'could not find image i/o class %s' % cftype        

    return IMAGEIO[cftype]
    
