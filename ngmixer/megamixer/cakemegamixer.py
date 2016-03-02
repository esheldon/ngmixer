#!/usr/bin/env python
import os
import sys
import meds
import fitsio
import numpy as np
import glob
from ..files import read_yaml

from .megamixer import NGMegaMixer
import cake

class CakeNGMegaMixer(NGMegaMixer):
    def write_job_script(self,files,i,rng):
        fname = os.path.join(self.get_chunk_output_dir(files,i,rng),'job.sh')
        jobname = self.get_chunk_output_basename(files,self['run'],rng)

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
        
        # add to cake
        cake_file = os.path.join(files['work_output_dir'],'cake_chunks.db')
        dr = self.get_chunk_output_dir(files,chunk,rng)
        with cake.taskdb.SQLiteTaskDB(cake_file) as taskdb:
            taskdb.add('cd %s && ./job.sh && cd -' % dr,id=jobname)

        # add to global cake
        cake_file = os.path.join(files['run_output_dir'],'cake_chunks.db')
        dr = self.get_chunk_output_dir(files,chunk,rng)
        with cake.taskdb.SQLiteTaskDB(cake_file) as taskdb:
            taskdb.add('cd %s && ./job.sh && cd -' % dr,id=jobname)

    def run_chunk(self,files,chunk,rng):
        print "To run the chunks, build a job using cake (i.e., cake run /path/to/cake_chunks.db)"

    def write_nbrs_job_script(self,files):
        fname = os.path.join(files['work_output_dir'],'jobnbrs.sh')
        jobname = '%s-nbrs' % files['coadd_tile']

        if len(self['extra_cmds']) > 0:
            with open(self['extra_cmds'],'r') as f:
                ec = f.readlines()
            ec = '\n'.join([e.strip() for e in ec])
        else:
            ec = ''

        with open(fname,'w') as fp:
            fp.write("""#!/bin/bash

{extracmds}
./runnbrs.sh

""".format(extracmds=ec))

        os.system('chmod 755 %s' % fname)

        # add to cake
        cake_file = os.path.join(files['work_output_dir'],'cake_nbrs.db')
        dr = files['work_output_dir']
        with cake.taskdb.SQLiteTaskDB(cake_file) as taskdb:
            taskdb.add('cd %s && ./jobnbrs.sh && cd -' % dr,id=jobname)

        # add to global cake
        cake_file = os.path.join(files['run_output_dir'],'cake_nbrs.db')
        dr = files['work_output_dir']
        with cake.taskdb.SQLiteTaskDB(cake_file) as taskdb:
            taskdb.add('cd %s && ./jobnbrs.sh && cd -' % dr,id=jobname)

    def run_nbrs(self,files):
        print "To run the chunks, build a job using cake (i.e., cake run /path/to/cake_nbrs.db)"
