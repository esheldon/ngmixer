from __future__ import print_function
import os
import sys
import numpy
import fitsio
from .. import files
from ..util import Namer

class ConcatError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Concat(object):
    """
    Concatenate split files
    """
    def __init__(self,
                 run,
                 config_file,
                 chunk_list,
                 output_dir,
                 output_file,
                 clobber=False,
                 skip_errors=False,
                 **kwargs):

        self.run=run
        self.config_file=config_file

        self.output_dir = output_dir
        self.output_file = output_file
        self.clobber = clobber
        self.skip_errors = skip_errors
        self.chunk_file_list = chunk_list

        self.config = files.read_yaml(config_file)

        self.make_collated_dir()
        self.set_collated_file()

    def pick_fields(self,data0):
        """
        modify and/or add fields to the data array
        """
        return data0

    def pick_epoch_fields(self,epoch_data0):
        """
        modify and/or add fields to the epoch data array
        """
        return epoch_data0

    def make_collated_dir(self):
        """
        set collated file output dir
        """
        files.try_makedir(self.output_dir)

    def set_collated_file(self):
        """
        set the output file and the temporary directory
        """
        if self.blind:
            extra='-blind'
        else:
            extra=''

        self.collated_file = os.path.join(self.output_dir, "%s-%s%s.fits" % (self.run,self.output_file,extra))
        self.tmpdir = files.get_temp_dir()

    def read_chunk(self, fname):
        """
        Read the chunk data
        """
        if not os.path.exists(fname):
            raise ConcatError("file not found: %s" % fname)

        try:
            with fitsio.FITS(fname) as fobj:
                data0 = fobj['model_fits'][:]
                epoch_data0 = fobj['epoch_data'][:]
                meta = fobj['meta_data'][:]
        except IOError as err:
            raise ConcatError(str(err))

        data = self.pick_fields(data0,meta)
        epoch_data = self.pick_epoch_fields(epoch_data0)

        return data, epoch_data, meta

    def verify(self):
        """
        just run through and read the data, verifying we can read it
        """

        dlist = []

        nchunk=len(self.chunk_file_list)
        for i,fname in enumerate(self.chunk_file_list):

            print('\t%d/%d %s' %(i+1,nchunk,fname))
            sys.stdout.flush()
            try:
                data, epoch_data, meta = self.read_chunk(fname)
                dlist.extend(list(data))
            except ConcatError as err:
                print("error found: %s" % str(err))

        data = numpy.array(data,dtype=data.dtype.descr)
        if not numpy.array_equal(numpy.sort(numpy.unique(data['number'])),numpy.sort(data['number'])):
            print("object 'number' field is not unique!")

    def concat(self):
        """
        actually concatenate the data, and add any new fields
        """
        print('writing:',self.collated_file)

        if os.path.exists(self.collated_file) and not self.clobber:
            print('file already exists, skipping')
            return

        dlist=[]
        elist=[]

        nchunk=len(self.chunk_file_list)
        for i,fname in enumerate(self.chunk_file_list):

            print('\t%d/%d %s' %(i+1,nchunk,fname))
            sys.stdout.flush()
            try:
                data, epoch_data, meta = self.read_chunk(fname)
                dlist.extend(list(data))
                elist.extend(list(epoch_data))
            except ConcatError as err:
                if not self.skip_errors:
                    raise err
                print("\tskipping problematic chunk")

        data = numpy.array(dlist,dtype=data.dtype.descr)
        epoch_data = numpy.array(elist,dtype=epoch_data.dtype.descr)

        if not numpy.array_equal(numpy.sort(numpy.unique(data['number'])),numpy.sort(data['number'])):
            print("object 'number' field is not unique!")

        # note using meta from last file
        self._write_data(data, epoch_data, meta)

    def _write_data(self, data, epoch_data, meta):
        """
        write the data, first to a local file then staging out
        the the final location
        """
        with files.StagedOutFile(self.collated_file, tmpdir=self.tmpdir) as sf:
            with fitsio.FITS(sf.path,'rw',clobber=True) as fits:
                fits.write(data,extname="model_fits")
                fits.write(epoch_data,extname='epoch_data')
                fits.write(meta,extname="meta_data")

        print('output is in:',self.collated_file)
