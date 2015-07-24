#!/usr/bin/env python

def get_imageio_class(ftype):
    """
    returns the imageio class for a given a ftype
    """
    from .__init__ import IMAGEIO
    
    cftype = ftype.upper()
    assert cftype in IMAGEIO,'could not find image i/o class %s' % cftype        

    return IMAGEIO[cftype]
    
class ImageIO(object):
    """
    abstract base class for reading images
    
    To implement an image reader, you need to fill out the abstract methods below
    and make a __next__ and next method like this.
    
    class MyImageIO(ImageIO):
        def __init__(self,*args,**kwargs):
            super(MEDSImageIO, self).__init__(*args,**kwargs)
            self.num_fofs = # of mb obs lists to return
            
        def __next__(self):
            if self.fofindex >= self.num_fofs:
                raise StopIteration
            else:
                coadd_mb_obs_lists,se_mb_obs_lists = # get mb obs lists for fofindex
                self.fofindex += 1
                return coadd_mb_obs_lists,se_mb_obs_lists

        next = __next__
    
        ...
    
    The coadd_mb_obs_lists and se_mb_obs_lists need to have their meta data set with the field
    
        'meta_row': numpy array with meta data (same dtype as returned by get_meta_data_dtype)
    
    Each individual observation in the lists needs to have the following meta data fields set
    
        'flags': non-zero if the observation should be ignored
        'meta_row': numpy array with epoch meta data (same dtype as returned by get_epoch_meta_data_dtype)
    
    You can use the update_meta_data method of these objects to set the data.
    
    """
    
    def __init__(self,*args,**kwargs):
        self.set_fof_start(0)
        self.num_fofs = 0

        if 'fof_data' in kwargs:
            self.fof_data = kwargs['fof_data']
        else:
            self.fof_data = None

        if 'extra_data' in kwargs:
            self.extra_data = kwargs['extra_data']
        else:
            self.extra_data = None
        
    def get_num_bands(self):
        """
        returns number of bands for galaxy images        
        """
        raise NotImplementedError("get_num_bands method of ImageIO must be defined in subclass.")
    
    def get_meta_data_dtype(self):
        """
        returns a numpy dtype for the galaxy meta data as a list
        
        For example
        
            return [('id','i8'),('number','i4')]
        """
        raise NotImplementedError("get_meta_data_dtype method of ImageIO must be defined in subclass.")
    
    def get_epoch_meta_data_dtype(self):
        """
        returns a numpy dtype for the galaxy per epoch meta data as a list
        
        For example
        
            return [('id','i8'),('number','i4'),('icut','i4')]
        """
        raise NotImplementedError("get_epoch_meta_data_dtype method of ImageIO must be defined in subclass.")
    
    def set_fof_start(self,start):
        self.fof_start = start

    def get_num_fofs(self):
        """
        returns # of fofs code will yeild
        """
        raise NotImplementedError("get_num_fofs method of ImageIO must be defined in subclass.")
        
    def __iter__(self):
        self.fofindex = self.fof_start
        return self
