#!/usr/bin/env python
import os
import sys
import meds
import fitsio
import numpy as np
import glob
from ..files import read_yaml
from .concat_io import get_concat_class

class BaseNGMegaMixer(dict):
    def __init__(self,conf,extra_cmds='',seed=None):
        self.update(conf)
        self.ngmix_conf = read_yaml(conf['ngmix_config'])
        self['model_nbrs'] = self.ngmix_conf.get('model_nbrs',False)
        self['extra_cmds'] = extra_cmds        
        self.rng = np.random.RandomState(seed=seed)

    def get_files(self,full_coadd_tile):
        """
        get all paths to files
        """
        files = {}
        files['full_coadd_tile'] = full_coadd_tile
        coadd_tile = full_coadd_tile.split('_')[-1]
        files['coadd_tile'] = coadd_tile
        
        # desdata
        DESDATA = os.environ.get('DESDATA')
        files['DESDATA'] = DESDATA

        # meds files
        files['meds_files'] = []
        for band in self['bands']:
            medsf = os.path.join(DESDATA,
                                 'meds',
                                 self['meds_version'],
                                 files['full_coadd_tile'].split('/')[-1],
                                 '%s-%s-meds-%s.fits.fz' % (coadd_tile,band,self['meds_version']))
            assert os.path.exists(medsf),"MEDS file %s for band %s does not exist!" % (medsf,band)
            files['meds_files'].append(medsf)

        # now look for nbrs
        nbrsf = os.path.join(DESDATA,
                             'EXTRA',
                             'meds',
                             self['meds_version'],
                             'nbrs-data',
                             self['nbrs_version'],
                             files['full_coadd_tile'].split('/')[-1],
                             '%s-meds-%s-nbrslist-%s.fits' % (coadd_tile,self['meds_version'],self['nbrs_version']))
        files['nbrs_file'] = nbrsf

        # do the fofs
        foff = os.path.join(DESDATA,
                            'EXTRA',
                            'meds',
                            self['meds_version'],
                            'nbrs-data',
                            self['nbrs_version'],
                            files['full_coadd_tile'].split('/')[-1],
                            '%s-meds-%s-nbrsfofs-%s.fits' % (coadd_tile,self['meds_version'],self['nbrs_version']))
        files['fof_file'] = foff
        
        # finally look for flags
        flagsf = os.path.join(DESDATA,
                              'EXTRA',
                              'meds',
                              self['meds_version'],
                              'obj-flags-data',
                              self['obj_flags_version'],
                              files['full_coadd_tile'].split('/')[-1],
                              '%s-meds-%s-flags-%s.fits' % (coadd_tile,self['meds_version'],self['obj_flags_version']))
        files['obj_flags'] = flagsf
        
        return files

    def get_fof_ranges(self,files):
        if self['model_nbrs']:
            fofs = fitsio.read(files['fof_file'])
            num_fofs = len(np.unique(fofs['fofid']))
        else:
            m = meds.MEDS(files['meds_files'][0])
            num_fofs = m.size
            m.close()

        nchunks = num_fofs/self['num_fofs_per_chunk']
        if nchunks*self['num_fofs_per_chunk'] < num_fofs:
            nchunks += 1

        fof_ranges = []
        for chunk in xrange(nchunks):
            sr = chunk*self['num_fofs_per_chunk']
            sp = sr + self['num_fofs_per_chunk'] - 1
            if sp >= num_fofs:
                sp = num_fofs-1
            fof_ranges.append([sr,sp])

        return fof_ranges

    def get_main_output_dir(self,coadd_tile):
        return os.path.join(self['output_dir'],coadd_tile)

    def get_work_output_dir(self,coadd_tile):
        return os.path.join(self.get_main_output_dir(coadd_tile),'work')

    def get_chunk_output_dir(self,coadd_tile,chunk,rng):
        return os.path.join(self.get_work_output_dir(coadd_tile),'chunk%05d_%d_%d' % (chunk,rng[0],rng[1]))

    def get_chunk_output_basename(self,coadd_tile,chunk,rng):
        return '%s-%s-%d-%d' % (coadd_tile,self['run'],rng[0],rng[1])

    def make_output_dirs(self,files,fof_ranges):
        """
        make output dirs
        """
        odir = self.get_main_output_dir(files['coadd_tile'])
        wdir = self.get_work_output_dir(files['coadd_tile'])
        os.system('rm -rf %s/*' % odir)
        for dr in [odir,wdir]:
            if not os.path.exists(dr):
                os.mkdir(dr)

        os.system('cp %s %s' % (self['run_config'],os.path.join(odir,'.')))

        for chunk,rng in enumerate(fof_ranges):
            dr = self.get_chunk_output_dir(files['coadd_tile'],chunk,rng)
            if not os.path.exists(dr):
                os.mkdir(dr)

        if len(self['extra_cmds']) > 0:
            os.system('cp %s %s' % (self['extra_cmds'],os.path.join(odir,'.')))

    def make_scripts(self,files,fof_ranges):
        os.system('cp %s %s' % (self['ngmix_config'],os.path.join(self.get_main_output_dir(files['coadd_tile']),'.')))

        for i,rng in enumerate(fof_ranges):
            self.write_script(files,i,rng)
            self.write_job_script(files,i,rng)

    def write_script(self,files,i,rng):
        fmt = r"""#!/bin/bash
chunk={chunk}
config={ngmix_config}
obase={base_name}
ofile=$obase".fits"
lfile=$obase".log"
meds="{meds_files}"
tmpdir={tmpcmd}

# call with -u to avoid buffering
cmd="`which {cmd}` \
    --fof-range={start},{stop} \
    --work-dir=$tmpdir \
    {fof_opt} \
    {nbrs_opt} \
    {flags_opt} \
    {seed_opt} \
    $config $ofile $meds"

echo $cmd
python -u $cmd &> $lfile

"""
        args = {}
        args['chunk'] = i
        args['start'] = rng[0]
        args['stop'] = rng[1]
        args['ngmix_config'] = os.path.join('..','..',self['ngmix_config'])
        args['meds_files'] = ' '.join([medsf.replace(files['DESDATA'],'${DESDATA}') for medsf in files['meds_files']])
        args['base_name'] = self.get_chunk_output_basename(files['coadd_tile'],self['run'],rng)
        args['tmpcmd'] = self.get_tmp_dir()
        args['cmd'] = 'ngmixit'

        if os.path.exists(files['fof_file']):
            args['fof_opt'] = '--fof-file=%s'% files['fof_file'].replace(files['DESDATA'],'${DESDATA}')
        else:
            args['fof_opt'] = ''

        if os.path.exists(files['nbrs_file']):
            args['nbrs_opt'] = '--nbrs-file=%s'% files['nbrs_file'].replace(files['DESDATA'],'${DESDATA}')
        else:
            args['nbrs_opt'] = ''

        if os.path.exists(files['obj_flags']):
            args['flags_opt'] = '--obj-flags=%s'% files['obj_flags'].replace(files['DESDATA'],'${DESDATA}')
        else:
            args['flags_opt'] = ''

        if 'seed' not in self:
            seed = self.rng.randint(low=1,high=1000000000)
            args['seed_opt'] = '--seed=%d' % seed
        else:
            args['seed_opt'] = ''

        scr = fmt.format(**args)

        scr_name = os.path.join(self.get_chunk_output_dir(files['coadd_tile'],i,rng),'runchunk.sh')
        with open(scr_name,'w') as fp:
            fp.write(scr)

        os.system('chmod 755 %s' % scr_name)

    def get_files_fof_ranges(self,coadd_tile):
        files = self.get_files(coadd_tile)

        # error check
        if self['model_nbrs']:
            assert os.path.exists(files['fof_file']),"fof file %s must be made to model nbrs!" % files['fof_file']
            assert os.path.exists(files['nbrs_file']),"nbrs file %s must be made to model nbrs!" % files['nbrs_file']

        fof_ranges = self.get_fof_ranges(files)

        return files,fof_ranges

    def setup_coadd_tile(self,coadd_tile):
        files,fof_ranges = self.get_files_fof_ranges(coadd_tile)

        self.make_output_dirs(files,fof_ranges)
        self.make_scripts(files,fof_ranges)

    def run_coadd_tile(self,coadd_tile):
        files,fof_ranges = self.get_files_fof_ranges(coadd_tile)

        for chunk,rng in enumerate(fof_ranges):
            dr = self.get_chunk_output_dir(files['coadd_tile'],chunk,rng)
            base = self.get_chunk_output_basename(files['coadd_tile'],self['run'],rng)
            fname = os.path.join(dr,base+'.fits')
            if not os.path.exists(fname):
                self.run_chunk(files,chunk,rng)

    def rerun_coadd_tile(self,coadd_tile):
        files,fof_ranges = self.get_files_fof_ranges(coadd_tile)

        for chunk,rng in enumerate(fof_ranges):
            dr = self.get_chunk_output_dir(files['coadd_tile'],chunk,rng)
            base = self.get_chunk_output_basename(files['coadd_tile'],self['run'],rng)

            fname = os.path.join(dr,base+'-checkpoint.fits')
            if os.path.exists(fname):
                os.remove(fname)

            fname = os.path.join(dr,base+'.fits')
            if os.path.exists(fname):
                os.remove(fname)

            self.run_chunk(files,chunk,rng)

    def collate_coadd_tile(self,coadd_tile,verify=False,blind=True,clobber=True,skip_errors=False):
        files,fof_ranges = self.get_files_fof_ranges(coadd_tile)

        clist = []
        for chunk,rng in enumerate(fof_ranges):
            dr = self.get_chunk_output_dir(files['coadd_tile'],chunk,rng)
            base = self.get_chunk_output_basename(files['coadd_tile'],self['run'],rng)
            fname = os.path.join(dr,base+'.fits')
            clist.append(fname)

        tc = get_concat_class(self['concat_type'])
        tc = tc(self['run'],
                self['ngmix_config'],
                clist,
                self.get_main_output_dir(files['coadd_tile']),
                files['coadd_tile'],
                bands=self['bands'],
                blind=blind,
                clobber=clobber,
                skip_errors=skip_errors)

        if verify:
            tc.verify()
        else:
            tc.concat()

    def get_tmp_dir(self):
        return '`mktemp -d /tmp/XXXXXXXXXX`'

    def write_job_script(self,files,i,rng):
        """
        method that writes a script to run the runchunk.sh script

        The script must run the extra cmds in the file in self['extra_cmds'],
        if this file exists, and then run 'runchunk.sh'.

        The script should assume it is in the same working dir as runchunk.sh.

        See the example below.

        """
        raise NotImplementedError("write_job_script method of BaseNGMegaMixer must be defined in subclass.")

    def run_chunk(self,files,chunk,rng):
        """
        This method must make some sort of system call to actually submit a single chunk to a queue
        or to run it on the local node.

        See the example below.
        """
        raise NotImplementedError("run_chunk method of BaseNGMegaMixer must be defined in subclass.")

class NGMegaMixer(BaseNGMegaMixer):
    def write_job_script(self,files,i,rng):
        fname = os.path.join(self.get_chunk_output_dir(files['coadd_tile'],i,rng),'job.sh')

        if len(self['extra_cmds']) > 0:
            with open(self['extra_cmds'],'r') as f:
                ec = f.readlines()
            ec = '\n'.join([e.strip() for e in ec])
        else:
            ec = ''

        with open(fname,'w') as fp:
            fp.write("""#!/bin/bash
{extracmds}
./runchunk.sh

""".format(extracmds=ec))

        os.system('chmod 755 %s' % fname)

    def run_chunk(self,files,chunk,rng):
        dr = self.get_chunk_output_dir(files['coadd_tile'],chunk,rng)
        os.system('cd %s && ./job.sh && cd -' % dr)
