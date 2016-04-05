#!/usr/bin/env python
import os
import sys
import meds
import fitsio
import numpy as np
import glob
from ..files import read_yaml

from .megamixer import NGMegaMixer

class BNLCondorMegaMixer(NGMegaMixer):
    def __init__(self,*args,**kwargs):
        super(BNLCondorMegaMixer,self).__init__(*args,**kwargs)
        self['queue'] = self.get('queue','medium')


    def setup_coadd_tile(self,coadd_tile):
        """
        Clean up the condor script
        """

        # this makes the individual scripts
        files,fof_ranges=super(BNLCondorMegaMixer,self).setup_coadd_tile(coadd_tile)

        # above sets self['work_output_dir']
        condor_script=self.get_condor_file(files)
        condor_script_all=self.get_condor_file(files,doall=True)
        try:
            os.remove(condor_script)
            os.remove(condor_script_all)
        except:
            pass

        self.write_master_script(files)

        # this holds everything and is not automatically used.
        # when using the megamixer run command, another file
        # is written just for those outputs that don't exist
        self.write_condor(files,fof_ranges,doall=True)

        # this one just holds the jobs for which the output file
        # was not found.
        # note when using megamixit run this will be over-written
        # just in case some more files were completed or removed
        # after running setup
        self.write_condor(files,fof_ranges,doall=False)



    def run_coadd_tile(self,coadd_tile):
        """
        Write the condor file and submit it, only doing those
        for which the output file doesn't exist
        """
        files,fof_ranges = self.get_files_fof_ranges(coadd_tile)

        fname,nwrite=self.write_condor(files,fof_ranges)

        if  nwrite == 0:
            print "    no unfinished jobs left to run"
            return
        fname=os.path.basename(fname)

        dr = files['work_output_dir']
        cmd='cd %s && condor_submit %s && cd -' % (dr,fname)
        print "condor command: '%s'" % cmd
        os.system(cmd)


    def get_tmp_dir(self):
        return '${TMPDIR}'

    #def get_chunk_output_dir(self,files,chunk,rng):
    #    return os.path.join(files['work_output_dir'],
    #                        'chunk%06d' % (chunk+1))

    def get_chunk_jobname(self, files, rng):
        return '%s-%05d-%05d' % (files['coadd_tile'], rng[0], rng[1])

    def get_chunk_output_dir(self,files,chunk,rng):
        return os.path.join(files['work_output_dir'], \
                            'chunk%03d-%05d-%05d' % (chunk,rng[0],rng[1]))

    def get_chunk_output_basename(self,files,chunk,rng):
        return '%s-%s-%05d-%05d' % (files['coadd_tile'],self['run'],rng[0],rng[1])

    def get_script_template(self):
        """
        for condor we need to use the directory made by the system
        So we ignore tmpdir
        """
        template = r"""#!/bin/bash

output_dir={output_dir}
chunk={chunk}
config={ngmix_config}
obase={base_name}
output_file="$output_dir/$obase.fits"
meds="{meds_files}"

# ignoring tmpcmd
tmpdir=$TMPDIR

pushd $tmpdir

{cmd} \
    --fof-range={start},{stop} \
    --work-dir=$tmpdir \
    {fof_opt} \
    {nbrs_opt} \
    {flags_opt} \
    {seed_opt} \
    $config $output_file $meds

popd
        """
        return template


    def write_master_script(self,files):
        """
        this is a master script to run the individual scripts,
        move to the temporary directory made by condor, and
        move the log file to the output
        """
        path=self.get_master_script_file(files)
        text=self.get_master_script_text()

        print "writing:",path
        with open(path,'w') as fobj:
            fobj.write(text)
        os.system('chmod 755 %s' % path)

    def get_master_script_file(self, files):
        """
        location of the master script
        """
        path=os.path.join(files['work_output_dir'], 'master.sh')
        return path

    def get_master_script_text(self):
        """
        for condor we need to use the directory made by the system
        We cd there, write the log file and move to the final dir
        """
        template = r"""#!/bin/bash

if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
    # the condor system creates a scratch directory for us,
    # and cleans up afterward
    tmpdir=$_CONDOR_SCRATCH_DIR
    export TMPDIR=$tmpdir
else
    # otherwise use the TMPDIR
    tmpdir=$TMPDIR
    mkdir -p $tmpdir
fi


# this will be a path to a script
cmd=$1
logfile=$2

# move the log file into that directory
odir=$(dirname $logfile)
tmplog=$(basename $logfile)

mkdir -p $odir

# any temp files written to cwd, plus log, go into the
# temporary directory
pushd $tmpdir

echo $cmd
$cmd &> $tmplog

mv -v "$tmplog" "$odir/"

popd
        \n"""
        return template



    def get_condor_file(self, files, doall=False):
        """
        need to have run get_files* to set the dir below

        will hold a line for every job
        """
        if doall:
            name='submit_all.condor'
        else:
            name='submit.condor'

        path=os.path.join(files['work_output_dir'], name)
        return path


    def get_condor_head_template(self,files):
        master=self.get_master_script_file(files)
        text="""
Universe        = vanilla

Notification    = Never

# Run this exe with these args
Executable      = {master_script}

Image_Size      = 1000000

GetEnv = True

kill_sig        = SIGINT

#requirements = (cpu_experiment == "star") || (cpu_experiment == "phenix")
#requirements = (cpu_experiment == "star")

+Experiment     = "astro"\n\n"""
        return text.format(master_script=master)

    def get_condor_job_line(self, jobname, script_file, log_file):
        text="""
+job_name = "%(jobname)s"
Arguments = %(script_file)s %(log_file)s
Queue\n"""
        text=text % dict(
            jobname=jobname,
            script_file=script_file,
            log_file=log_file,
        )
        return text

    def write_condor(self,files,fof_ranges, doall=False):
        """
        write the condor submit file
        """

        fname=self.get_condor_file(files,doall=doall)

        print "writing:",fname
        nwrite=0
        with open(fname,'w') as fobj:
            head=self.get_condor_head_template(files)
            fobj.write(head)

            for chunk,rng in enumerate(fof_ranges):

                output_file=self.get_chunk_output_file(files,chunk, rng)

                if doall:
                    dowrite=True
                elif not os.path.exists(output_file):
                    dowrite=True
                else:
                    dowrite=False

                if dowrite:
                    nwrite+=1
                    jobname=self.get_chunk_jobname(files, rng)
                    script_file=self.get_chunk_script_file(files, chunk, rng)
                    log_file=self.get_chunk_log_file(files, chunk, rng)

                    line=self.get_condor_job_line(jobname, script_file, log_file)

                    fobj.write(line)

        print "    wrote",nwrite,"jobs"
        return fname, nwrite

    def write_job_script(self,files,i,rng):
        """
        for condor we don't have a separate job script
        """
        return


    def write_nbrs_job_script(self,files):
        return

    def run_nbrs(self,files):
        raise NotImplementedError("set up run nbrs")


