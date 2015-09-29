from __future__ import print_function
import os
import sys
import numpy
import fitsio
from .. import files
from ..util import Namer

from .concat import Concat,ConcatError

SHAPENOISE=0.16
SHAPENOISE2=SHAPENOISE**2

class DESConcat(Concat):
    """
    Concat for DES, database prep and blinding
    """
    def __init__(self,*args,**kwargs):        
        assert 'bands' in kwargs,"band names must be supplied to DESConcat"
        self.bands = kwargs.pop('bands')
        self.nbands = len(self.bands)
        self.blind = kwargs.pop('blind',True)
        if self.blind:
            self.blind_factor = self.get_blind_factor()
        super(DESConcat,self).__init__(*args,**kwargs)
        self.config['fit_models'] = list(self.config['model_pars'].keys())

    def get_blind_factor(self):
        """
        by joe zuntz
        """
        import sys
        import hashlib

        try:
            with open(os.environ['DESBLINDPHRASE'],'r') as fp:
                code_phrase = fp.read()
            code_phrase = code_phrase.strip()
        except:
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

    def read_chunk(self, fname):
        """
        Read the chunk data
        """
        d,ed,m = super(DESConcat,self).read_chunk(fname)

        if self.blind:
            self.blind_data(d)
            
        return d,ed,m

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
        for nm in epoch_data0.dtype.names:
            if nm in epoch_data.dtype.names:
                epoch_data0[nm] = epoch_data[nm]

        for band_num in xrange(self.nbands):
            w,=numpy.where(epoch_data['band_num'] == band_num)
            if w.size > 0:
                epoch_data['band'][w] = self.bands[band_num]

        return epoch_data

    def get_models(self, data):
        models=[]

        model_names = self.config['model_pars'].keys()
        model_names = ['%s_max' % mod for mod in model_names] + model_names
        model_names = ['coadd_%s' % mod for mod in model_names] + model_names

        names=list( data.dtype.names )
        for model in model_names:
            n=Namer(model)

            if n('flux') in names or n('pars') in names:
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
        do_flux=False
        for ft in models:

            n=Namer(ft)

            if '%s_g_cov' % ft in names:
                gcind = names.index('%s_g_cov' % ft)
                wtf = (n('weight'), 'f8')
                dt.insert(gcind+1, wtf)
                names.insert(gcind+1, n('weight'))

            if n('flux') in names:
                flux_ind = names.index(n('flux'))
            else:
                do_flux = True
                pars_cov_ind = names.index(n('pars_cov'))

                offset = 1
                dt.insert(pars_cov_ind+offset, (n('flux'), 'f8', nbands) )
                names.insert(pars_cov_ind+offset,n('flux'))

                offset += 1
                dt.insert(pars_cov_ind+offset, (n('flux_cov'), 'f8', (nbands,nbands)) )
                names.insert(pars_cov_ind+offset,n('flux_cov'))

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

            if n('T') not in data0.dtype.names:
                fadd=[(n('T'),'f8'),
                      (n('T_err'),'f8'),
                      (n('T_s2n'),'f8')]
                ind = names.index('%s_pars_cov' % ft)
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
                self.calc_mag_and_flux_stuff(data, meta, ft, band, do_flux=do_flux)

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
                if n('g_cov') in data.dtype.names:
                    c=data[n('g_cov')]
                    weight=1.0/(2.*SHAPENOISE2 + c[w,0,0] + 2*c[w,0,1] + c[w,1,1])
                    data[n('weight')][w] = weight

    def calc_mag_and_flux_stuff(self, data, meta, model, band, do_flux=False):
        """
        Get magnitudes
        """

        names = data.dtype.names

        n=Namer(model)

        nband=self.nbands

        if nband == 1:
            data[n('mag')] = -9999.
            data[n('flux_s2n')] = 0.0
            if do_flux:
                data[n('flux')] = -9999.
                data[n('flux_cov')] = -9999.
        else:
            data[n('mag')][:,band] = -9999.
            data[n('flux_s2n')][:,band] = 0.0
            if do_flux and 'psf' not in model:
                data[n('flux')][:,band] = -9999.
                data[n('flux_cov')][:,:,band] = -9999.

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
            if do_flux and 'psf' not in model:
                if nband == 1:
                    data[n('flux')][w] = data[n('pars')][w,5]
                    data[n('flux_cov')][w,1,1] = data[n('pars_cov')][w,5,5]
                else:
                    data[n('flux')][w,band] = data[n('pars')][w,5+band]
                    data[n('flux_cov')][w,:,:] = data[n('pars_cov')][w,5:5+nband,5:5+nband]

            if nband == 1:
                flux = (data[n('flux')][w]).clip(min=0.001)
            else:
                flux = (data[n('flux')][w,band]).clip(min=0.001)
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
                    flux=data[n('flux')][w]
                    flux_var=data[n('flux_cov')][w]

                    w2,=numpy.where(flux_var > 0)
                    if w.size > 0:
                        flux=flux[w2]
                        flux_err=numpy.sqrt(flux_var[w2])
                        data[n('flux_s2n')][w[w2]] = flux/flux_err
                else:
                    flux=data[n('flux')][w,band]
                    flux_var=data[n('flux_cov')][w,band,band]

                    w2,=numpy.where(flux_var > 0)
                    if w.size > 0:
                        flux=flux[w2]
                        flux_err=numpy.sqrt(flux_var[w2])
                        data[n('flux_s2n')][w[w2], band] = flux/flux_err
