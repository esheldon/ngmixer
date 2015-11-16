#!/usr/bin/env python
import os
import sys
import meds
import fitsio
import numpy as np
import glob
from ..files import read_yaml

from .megamixer import NGMegaMixer

class SLACNGMegaMixer(NGMegaMixer):
    def __init__(self,*args,**kwargs):
        super(SLACNGMegaMixer,self).__init__(*args,**kwargs)
        self['queue'] = self.get('queue','medium')

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
#BSUB -q {queue}
#BSUB -J {jobname}
#BSUB -oo ./{jobname}.oe
#BSUB -R "linux64 && rhel60 && scratch > 2"
#BSUB -n 1
#BSUB -We 24:00
#BSUB -W 48:00

{extracmds}
./runchunk.sh

""".format(extracmds=ec,queue=self['queue'],jobname=jobname))

        os.system('chmod 755 %s' % fname)

    def run_chunk(self,files,chunk,rng):
        dr = self.get_chunk_output_dir(files,chunk,rng)
        os.system('cd %s && bsub < job.sh && cd -' % dr)
