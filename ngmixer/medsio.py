#!/usr/bin/env python

class MEDSImageIO(object):
    """
    Class for MEDS image I/O.
    """
    
    def __init__(self,conf,meds_files):
	self.conf = conf

	if not isinstance(meds_files, list):
	    self.meds_files = [meds_files]
	else:
	    self.meds_files =  meds_files
			    
	# load meds files and image flags array
	self._load_meds_files()
	
	self.conf['use_mof_fofs'] = self.conf.get('use_mof_fofs',False)
	if self.conf['use_mof_fofs']:
	    assert False,"Need to setup FoF reading code!"
	else:
	    self.fof_data = get_dummy_fofs(self.meds_list[0]['number'])
	
	self._set_and_check_index_lookups()
	
	self.psfex_lists = self._get_psfex_lists()
	#self.wcs_transforms = self._get_wcs_transforms()	    

	# some defaults
	self.iband = range(len(self.meds_list))
	self.conf['nband'] = len(self.meds_list)
	
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

	# first, we error check
	for band,meds in enumerate(self.meds_list):
	    assert numpy.array_equal(meds['number'],self.fof_data['number']),"FoF number is not the same as MEDS number for band %d!" % band

	#set some useful stuff here
	self.fofids = numpy.unique(self.fof_data['fofid'])

	#build the fof hash
	self.fofid2mindex = {}
	self.number2mindex = {}
	for fofid in self.fofids:
	    q, = numpy.where(self.fof_data['fofid'] == fofid)
	    assert len(q) > 0, 'Found zero length FoF! fofid = %ld' % fofid
	    self.fofid2mindex[fofid] = q.copy()
	    
	#build the number to mindex hash
	self.number2mindex = {}
	for mindex in xrange(self.nobj_tot):
	    self.number2mindex[self.fof_data['number'][mindex]] = mindex
	

    def __next__(self):
	assert False,"Need to setup iteration!"

    def _get_wcs_transforms(self):
	"""
	Load the WCS transforms for each meds file	  
	"""
	import json
	from esutil.wcsutil import WCS
	
	print('loading WCS')
	wcs_transforms = {}
	for band in self.iband:
	    mname = self.conf.['meds_files_full'][band]
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
    
    def _get_replacement_flags(self, filenames):
	from .util import CombinedImageFlags

	if not hasattr(self,'_replacement_flags'):
	    fname=os.path.expandvars(self.conf['replacement_flags'])
	    print("Reading replacement flags:",fname)
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

	for i,f in enumerate(self.meds_files):
	    print(f)
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
		image_flags[1:] = \
		    self._get_replacement_flags(image_info['image_path'][1:])

	    # now we reduce the flags to zero or IMAGE_FLAGS_SET
	    # copy out and check image flags just for cutouts
	    cimage_flags=image_flags[1:].copy()

	    w,=numpy.where( (cimage_flags & self.conf['image_flags2check']) != 0)

	    print("    flags set for: %d/%d" % (w.size,cimage_flags.size))

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
	print('loading psfex')
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
		    print("warning: missing psfex: %s" % psfpath)
		    self.all_image_flags[band][i] |= self.conf['image_flags2check']
		else:
		    print("loading:",psfpath)
		    try:
			pex=PSFEx(psfpath)
		    except PSFExError as err:
			print("	   problem with psfex file:",str(err))
			pex=None
    
	    psfex_list.append(pex)

	return psfex_list
