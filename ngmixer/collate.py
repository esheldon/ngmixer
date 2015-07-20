from __future__ import print_function
import os
import sys
import numpy
import fitsio
import json

from . import files
from .files import DEFAULT_NPER

from .util import Namer

class ConcatError(Exception):
    """
    EM algorithm hit max iter
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)


# need to fix up the images instead of this
from .constants import PIXSCALE2, SHAPENOISE2

class Concat(object):
    """
    Concatenate the split files

    This is the more generic interface
    """
    def __init__(self,
                 run,
                 config_file,
                 meds_files,
                 bands=None, # band names for each meds file
                 root_dir=None,
                 nper=DEFAULT_NPER,
                 sub_dir=None,
                 blind=True,
                 clobber=False,
                 skip_errors=False):

        from . import files

        self.run=run
        self.config_file=config_file
        self.meds_file_list=meds_files
        self.nbands=len(meds_files)
        if bands is None:
            bands = [str(i) for i in xrange(self.nbands)]
        else:
            emess=("wrong number of bands: %d "
                   "instead of %d" % (len(bands),self.nbands))
            assert len(bands)==self.nbands,emess
        self.bands=bands

        self.sub_dir=sub_dir
        self.nper=nper
        self.blind=blind
        self.clobber=clobber
        self.skip_errors=skip_errors

        self.config = files.read_yaml(config_file)
        self.config['fit_models']=list(self.config['model_pars'].keys())

        self.set_files(run, root_dir)
        
        self.make_collated_dir()
        self.set_collated_file()

        self.load_meds()
        self.set_chunks()

        if self.blind:
            self.blind_factor = get_blind_factor()


    def verify(self):
        """
        just run through and read the data, verifying we can read it
        """

        nchunk=len(self.chunk_list)
        for i,split in enumerate(self.chunk_list):

            print('\t%d/%d ' %(i+1,nchunk), end='')
            try:
                data, epoch_data, meta = self.read_chunk(split)
            except ConcatError as err:
                print("error found: %s" % str(err))

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

        nchunk=len(self.chunk_list)
        for i,split in enumerate(self.chunk_list):

            print('\t%d/%d ' %(i+1,nchunk), end='')
            sys.stdout.flush()
            try:
                data, epoch_data, meta = self.read_chunk(split)


                if self.blind:
                    self.blind_data(data)

                dlist.append(data)

                if (epoch_data is not None 
                        and epoch_data.dtype.names is not None):
                    elist.append(epoch_data)
            except ConcatError as err:
                if not self.skip_errors:
                    raise err
                print("skipping problematic chunk")

        # note using meta from last file
        self._write_data(dlist, elist, meta)

    def _write_data(self, dlist, elist, meta):
        """
        write the data, first to a local file then staging out
        the the final location
        """
        import esutil as eu
        from .files import StagedOutFile

        data=eu.numpy_util.combine_arrlist(dlist)
        if len(elist) > 0:
            do_epochs=True
            epoch_data=eu.numpy_util.combine_arrlist(elist)
        else:
            do_epochs=False

        with StagedOutFile(self.collated_file, tmpdir=self.tmpdir) as sf:
            with fitsio.FITS(sf.path,'rw',clobber=True) as fits:
                fits.write(data,extname="model_fits")

                if do_epochs:
                    fits.write(epoch_data,extname='epoch_data')

                fits.write(meta,extname="meta_data")

        print('output is in:',self.collated_file)

    def blind_data(self,data):
        """
        multiply all shear type values by the blinding factor

        This must be run after copying g values out of pars into
        the model_g fields

        This also includes the Q values from B&A
        """
        models=self.get_models(data)

        names=data.dtype.names
        for model in models:
            
            n=Namer(model)

            w,=numpy.where(data[n('flags')] == 0)
            if w.size > 0:
                data[n('pars')][:,2] *= self.blind_factor
                data[n('pars')][:,3] *= self.blind_factor
                if n('pars_best') in names:
                    data[n('pars_best')][:,2] *= self.blind_factor
                    data[n('pars_best')][:,3] *= self.blind_factor

                if n('g') in names:
                    data[n('g')][w,:] *= self.blind_factor

                if n('Q') in names:
                    data[n('Q')][w,:] *= self.blind_factor

    def pick_epoch_fields(self, epoch_data0):
        """
        pick out some fields, add some fields, rename some fields
        """
        import esutil as eu

        wkeep,=numpy.where(epoch_data0['cutout_index'] >= 0)
        if wkeep.size==0:
            print("None found with cutout_index >= 0")
            print(epoch_data0['cutout_index'])
            return numpy.zeros(1)

        epoch_data0=epoch_data0[wkeep]

        dt=epoch_data0.dtype.descr

        names=epoch_data0.dtype.names
        ind=names.index('band_num')
        dt.insert( ind, ('band','S1') )

        epoch_data = numpy.zeros(epoch_data0.size, dtype=dt)
        eu.numpy_util.copy_fields(epoch_data0, epoch_data)

        for band_num in xrange(self.nbands):

            w,=numpy.where(epoch_data['band_num'] == band_num)
            if w.size > 0:
                epoch_data['band'][w] = self.bands[band_num]

        return epoch_data

    def get_models(self, data):
        models=[]

        model_names = self.config['model_pars'].keys()
        model_names = ['coadd_%s' % mod for mod in model_names] + model_names

        model_names = ['coadd_gauss'] + model_names
        
        names=list( data.dtype.names )
        for model in model_names:
            n=Namer(model)

            if n('flux') in names:
                models.append(model)

        return models

    def pick_fields(self, data0, meta):
        """
        pick out some fields, add some fields, rename some fields
        """
        import esutil as eu

        nbands=self.nbands
        
        models=self.get_models(data0)

        names=list( data0.dtype.names )
        dt=[tdt for tdt in data0.dtype.descr]

        if 'coadd_psf_flux_err' in names:
            flux_ind = names.index('coadd_psf_flux_err')
            dt.insert(flux_ind+1, ('coadd_psf_flux_s2n','f8',nbands) )
            names.insert(flux_ind+1,'coadd_psf_flux_s2n')

            dt.insert(flux_ind+2, ('coadd_psf_mag','f8',nbands) )
            names.insert(flux_ind+2,'coadd_psf_mag')


        flux_ind = names.index('psf_flux_err')
        dt.insert(flux_ind+1, ('psf_flux_s2n','f8',nbands) )
        names.insert(flux_ind+1,'psf_flux_s2n')

        dt.insert(flux_ind+2, ('psf_mag','f8',nbands) )
        names.insert(flux_ind+2,'psf_mag')


        do_T=False
        for ft in models:

            n=Namer(ft)

            # turn on when we fix sign
            gcind = names.index('%s_g_cov' % ft)
            wtf = (n('weight'), 'f8')
            dt.insert(gcind+1, wtf)
            names.insert(gcind+1, n('weight'))

            flux_ind = names.index(n('flux'))

            offset=1
            dt.insert(flux_ind+offset, (n('flux_s2n'), 'f8', nbands) )
            names.insert(flux_ind+offset,n('flux_s2n'))

            offset += 1
            magf = (n('mag'), 'f8', nbands)
            dt.insert(flux_ind+offset, magf)
            names.insert(flux_ind+offset, n('mag'))

            offset += 1
            logsbf = (n('logsb'), 'f8', nbands)
            dt.insert(flux_ind+offset, logsbf)
            names.insert(flux_ind+offset, n('logsb'))
           
            pars_best_name='%s_pars_best' % ft
            if pars_best_name in names:
                offset += 1
                magf = (n('mag_best'), 'f8', nbands)
                dt.insert(flux_ind+offset, magf)
                names.insert(flux_ind+offset, n('mag_best'))

            if n('T') not in data0.dtype.names:
                fadd=[(n('T'),'f8'),
                      (n('T_err'),'f8'),
                      (n('T_s2n'),'f8')]
                ind = names.index('%s_flux_cov' % ft)
                for f in fadd:
                    dt.insert(ind+1, f)
                    names.insert(ind+1, f[0])
                    ind += 1

                do_T=True


        data=numpy.zeros(data0.size, dtype=dt)
        eu.numpy_util.copy_fields(data0, data)


        all_models=models
        if 'coadd_psf_flux' in names:
            all_models=all_models + ['coadd_psf']
        if 'psf_flux' in names:
            all_models=all_models + ['psf']

        if do_T:
            self.add_T_info(data, models)

        for ft in all_models:
            for band in xrange(nbands):
                self.calc_mag_and_flux_stuff(data, meta, ft, band)

        # turn this on when we fix sign
        self.add_weight(data, models)

        return data

    def add_T_info(self, data, models):
        """
        Add T S/N etc.
        """
        for ft in models:
            n=Namer(ft)

            data[n('T')][:]   = -9999.0
            data[n('T_err')][:]  =  9999.0
            data[n('T_s2n')][:] = -9999.0

            Tcov=data[n('pars_cov')][:,4,4]
            w,=numpy.where( (data[n('flags')] == 0) & (Tcov > 0.0) )
            if w.size > 0:
                data[n('T')][w]   = data[n('pars')][w, 4]
                data[n('T_err')][w]  =  numpy.sqrt(Tcov[w])
                data[n('T_s2n')][w] = data[n('T')][w]/data[n('T_err')][w]


    def add_weight(self, data, models):
        """
        Add weight for each model
        """
        for model in models:
            n=Namer(model)

            w,=numpy.where(data[n('flags')]==0)
            if w.size > 0:
                c=data[n('g_cov')]
                weight=1.0/(2.*SHAPENOISE2 + c[w,0,0] + 2*c[w,0,1] + c[w,1,1])

                data[n('weight')][w] = weight

                data[n('T_s2n')][w] = data[n('T')][w]/numpy.sqrt(data[n('pars_cov')][w,4,4])

    def calc_mag_and_flux_stuff(self, data, meta, model, band):
        """
        Get magnitudes
        """

        names = data.dtype.names

        n=Namer(model)

        nband=self.nbands

        if nband == 1:
            data[n('mag')] = -9999.
            data[n('flux_s2n')] = 0.0
        else:
            data[n('mag')][:,band] = -9999.
            data[n('flux_s2n')][:,band] = 0.0

        if 'psf' not in model:
            if nband == 1:
                data[n('logsb')] = -9999.0
            else:
                data[n('logsb')][:,band] = -9999.0

        if model in ['coadd_psf','psf']:
            if nband == 1:
                w,=numpy.where(data[n('flags')] == 0)
            else:
                w,=numpy.where(data[n('flags')][:,band] == 0)
        else:
            w,=numpy.where(data[n('flags')] == 0)

        if w.size > 0:
            if nband == 1:
                flux = ( data[n('flux')][w]/PIXSCALE2 ).clip(min=0.001)
            else:
                flux = ( data[n('flux')][w,band]/PIXSCALE2 ).clip(min=0.001)
            magzero=meta['magzp_ref'][band]

            if nband==1:
                data[n('mag')][w] = magzero - 2.5*numpy.log10( flux )
            else:
                data[n('mag')][w,band] = magzero - 2.5*numpy.log10( flux )

            if 'psf' not in model:
                if nband==1:
                    sb = data[n('flux')][w]/data[n('T')][w]
                    data[n('logsb')][w] = numpy.log10(numpy.abs(sb))
                else:
                    sb = data[n('flux')][w,band]/data[n('T')][w]
                    data[n('logsb')][w,band] = numpy.log10(numpy.abs(sb))

            if n('pars_best') in names:
                if nband==1:
                    flux_best = data[n('pars_best')][w,5]
                    flux_best = (flux_best/PIXSCALE2 ).clip(min=0.001)
                    data[n('mag_best')][w] = magzero - 2.5*numpy.log10( flux_best )
                else:
                    flux_best = data[n('pars_best')][w,5+band]
                    flux_best = (flux_best/PIXSCALE2 ).clip(min=0.001)
                    data[n('mag_best')][w,band] = magzero - 2.5*numpy.log10( flux_best )

            if model in ['coadd_psf','psf']:
                if nband==1:
                    flux=data[n('flux')][w]
                    flux_err=data[n('flux_err')][w]
                    w2,=numpy.where(flux_err > 0)
                    if w2.size > 0:
                        flux=flux[w2]
                        flux_err=flux_err[w2]
                        data[n('flux_s2n')][w[w2]] = flux/flux_err
                else:
                    flux=data[n('flux')][w,band]
                    flux_err=data[n('flux_err')][w,band]
                    w2,=numpy.where(flux_err > 0)
                    if w2.size > 0:
                        flux=flux[w2]
                        flux_err=flux_err[w2]
                        data[n('flux_s2n')][w[w2],band] = flux/flux_err
            else:
                if nband==1:
                    flux=data[n('flux_cov')][w]
                    flux_var=data[n('flux_cov')][w]

                    w2,=numpy.where(flux_var > 0)
                    if w.size > 0:
                        flux=flux[w2]
                        flux_err=numpy.sqrt(flux_var[w2])
                        data[n('flux_s2n')][w[w2]] = flux/flux_err
                else:
                    flux=data[n('flux_cov')][w]
                    flux_var=data[n('flux_cov')][w,band,band]

                    w2,=numpy.where(flux_var > 0)
                    if w.size > 0:
                        flux=flux[w2]
                        flux_err=numpy.sqrt(flux_var[w2])
                        data[n('flux_s2n')][w[w2], band] = flux/flux_err


    def read_data(self, fname, split):
        """
        Read the chunk data
        """

        if not os.path.exists(fname):
            raise ConcatError("file not found: %s" % fname)

        try:
            with fitsio.FITS(fname) as fobj:
                data0       = fobj['model_fits'][:]
                if 'epoch_data' in fobj:
                    epoch_data0 = fobj['epoch_data'][:]
                else:
                    print("    file has no epochs")
                    epoch_data0 = None
                meta        = fobj['meta_data'][:]
        except IOError as err:
            raise ConcatError(str(err))

        # watching for an old byte order bug
        expected_index = numpy.arange(split[0],split[1]+1)+1
        w,=numpy.where(data0['number'] != expected_index)
        if w.size > 0:
            raise ConcatError("number field is corrupted in file: %s" % fname)

        data = self.pick_fields(data0,meta)

        if epoch_data0 is not None:
            if epoch_data0.dtype.names is not None:
                epoch_data = self.pick_epoch_fields(epoch_data0)
            else:
                epoch_data = epoch_data0
        else:
            epoch_data = epoch_data0

            
        return data, epoch_data, meta

    def set_chunks(self):
        """
        set the chunks in which the meds file was processed
        """
        self.chunk_list=files.get_chunks(self.nrows, self.nper)

    def load_meds(self):
        """
        Load the associated meds files
        """
        import meds
        print('loading meds')

        self.meds_list=[]
        for fname in self.meds_file_list:
            m=meds.MEDS(fname)
            self.meds_list.append(m)

        self.nrows=self.meds_list[0]['id'].size

    def make_collated_dir(self):
        collated_dir = self._files.get_collated_dir()
        files.try_makedir(collated_dir)

    def set_collated_file(self):
        """
        set the output file and the temporary directory
        """
        if self.blind:
            extra='blind'
        else:
            extra=None
            
        self.collated_file = self._files.get_collated_file(sub_dir=self.sub_dir,
                                                           extra=extra)
        self.tmpdir=files.get_temp_dir()


    def read_chunk(self, split):
        """
        read data and epoch data from a given split
        """
        chunk_file=self._files.get_output_file(split,sub_dir=self.sub_dir)

        # continuing line from above
        print(chunk_file)
        data, epoch_data, meta=self.read_data(chunk_file, split)
        return data, epoch_data, meta

    def set_files(self, run, root_dir):
        self._files=files.Files(run, root_dir=root_dir)
    

def get_tile_key(tilename,band):
    key='%s-%s' % (tilename,band)
    return key

def key_by_tile_band(data0, ftype):
    """
    Group files from a goodlist by tilename-band and sort the lists
    """
    print('grouping by tile/band')
    data={}

    for d in data0:
        fname=d['output_files'][ftype]
        key=get_tile_key(d['tilename'],d['band'])

        if key not in data:
            data[key] = [fname]
        else:
            data[key].append(fname)

    for key in data:
        data[key].sort()

    print('found %d tile/band combinations' % len(data))

    return data

def get_blind_factor():
    """
    by joe zuntz
    """
    import sys
    import hashlib

    code_phrase = "DES is blinded"

    #hex number derived from code phrase
    m = hashlib.md5(code_phrase).hexdigest()
    #convert to decimal
    s = int(m, 16)
    # last 8 digits
    f = s%100000000
    # turn 8 digit number into value between 0 and 1
    g = f*1e-8
    #get value between 0.9 and 1
    return 0.9 + 0.1*g


