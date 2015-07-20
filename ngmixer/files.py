from __future__ import print_function
import os
import copy
import numpy

DEFAULT_NPER=10

class Files(dict):
    """
    files for gmix meds fitting

    parameters
    ----------
    run: string
        The run identifier
    root_dir: optional, string
        Send a root directory for output.  If not sent, the default is gotten
        from the GMIX_MEDS_DATADIR environment variable
    """
    def __init__(self, run, root_dir=None):

        self._run=run
        self._set_root_dir(root_dir=root_dir)
        self._set_run_dir()

    def get_root_dir(self):
        """
        get the root directory
        """
        return self._root_dir

    def get_run_dir(self):
        """
        run directory

        parameters
        ----------
        run: string
            The run identifier
        """
        return self._run_dir

    def get_file_dir(self, *path_elements):
        """
        get a directory name starting at run_dir with the
        additional path elements

        parameters
        ----------
        *path_elements:
            directory path elements, e.g.
            path_element1, path_element2, ...: each a string

        returns
        -------
        $run_dir/$element1/$element2/...
        """

        return os.path.join(self._run_dir, *path_elements)

    def get_file_name(self, *path_elements, **kw):
        """
        get file name, with sub_dir and split possibility

        parameters
        ----------
        *path_elements:
            directory path elements, e.g.
            path_element1, path_element2, ...: each a string

        split: 2-element sequence or None 
            The split numbers
        extra: string
            extra string for file name
        ext: string
            file extension, default 'fits'

        see get_file_dir for how the directory will be created

        the name is
        {run}-{path_element1}-{path_element2}-....{ext}
        {run}-{path_element1}-{path_element2}-...-{extra}.{ext}
        {run}-{path_element1}-{path_element2}-...-{split[0]}-{split[1]}.{ext}
        {run}-{path_element1}-{path_element2}-...-{split[0]}-{split[1]}_{extra}.{ext}
        """

        split=kw.get('split',None)
        extra=kw.get('extra',None)
        ext=kw.get('ext','fits')

        dir=self.get_file_dir(*path_elements)

        name=[copy.copy(self._run)] + list(path_elements)

        if split is not None:
            if len(split) != 2:
                raise ValueError("split should be [beg,end]")

            name.append('%06d' % split[0])
            name.append('%06d' % split[1])

        if extra is not None:
            name.append(extra)

        name='-'.join(name)
        name='%s.%s' % (name, ext)

        return os.path.join(dir, name)

    def get_script_dir(self):
        """
        get the directory for the script
        """
        return self.get_file_dir('script')

    def get_script_file(self):
        """
        get the path to the master script
        """
        path=self.get_file_name('script', ext='sh')
        return path

    def get_output_dir(self, sub_dir=None):
        """
        output directory

        parameters
        ----------
        sub_dir: string, optional
            An optional sub-directory name to use, e.g.
            a tile name for DES data

        %(run)s-%(beg)06d-%(end)%06d.fits
        %(run)s-%(sub_dir)s-%(beg)06d-%(end)%06d.fits
        """

        path_elements=['output']
        if sub_dir is not None:
            path_elements.append(sub_dir)
        return self.get_file_dir(*path_elements)

    def get_output_file(self, split, sub_dir=None):
        """
        output file name

        parameters
        ----------
        split: 2-element sequence
            [beg,end] for split.
        sub_dir: string, optional
            An optional sub-directory name to use, e.g.
            a tile name for DES data
        """

        path_elements=['output']
        if sub_dir is not None:
            path_elements.append(sub_dir)
        path=self.get_file_name(*path_elements, split=split, ext='fits')
        return path

    def get_collated_dir(self):
        """
        collated directory
        """
        return self.get_file_dir('collated')

    def get_oracle_dir(self):
        """
        collated directory
        """
        return self.get_file_dir('collated','oracle')

 
    def get_collated_file(self, sub_dir=None, extra=None):
        """
        output directory file name

        parameters
        ----------
        sub_dir: string, optional
            An optional sub-directory name to use, e.g.
            a tile name for DES data.  this does *not* become
            part of the directory, only the name
        extra: string
            Extra string, e.g. 'blind'
        """

        if sub_dir is not None:
            if extra is None:
                extra=[sub_dir]
            else:
                extra = [sub_dir, extra]
            extra = '-'.join(extra)

        path=self.get_file_name('collated', extra=extra, ext='fits')
        return path

    def get_wq_dir(self, sub_dir=None):
        """
        wq directory

        parameters
        ----------
        sub_dir: string, optional
            An optional sub-directory name to use, e.g.
            a tile name for DES data
        """

        path_elements=['wq']
        if sub_dir is not None:
            path_elements.append(sub_dir)
        return self.get_file_dir(*path_elements)

    def get_wq_file(self, sub_dir=None, split=None, extra=None):
        """
        wq file name

        parameters
        ----------
        sub_dir: string, optional
            An optional sub-directory name to use, e.g.
            a tile name for DES data
        split: optional, 2-element sequence
            [beg,end] for split.
        extra: string, optional
            e.g. 'missing'
        """

        path_elements=['wq']
        if sub_dir is not None:
            path_elements.append(sub_dir)

        path=self.get_file_name(*path_elements, split=split, ext='yaml', extra=extra)
        return path

    def get_condor_dir(self):
        """
        condor directory
        """

        return self.get_file_dir('condor')

    def get_condor_file(self, sub_dir=None, extra=None):
        """
        condor file name

        parameters
        ----------
        sub_dir: string, optional
            This is the subdir for the output files; condor files are all
            in the same directory but this goes into the name
        extra: string, optional
            e.g. 'missing'
        """

        ex=None
        if sub_dir is not None:
            ex=[sub_dir]
        if extra is not None:
            if ex is None:
                ex=[extra]
            else:
                ex.append(extra)

        if ex is not None:
            ex='-'.join(ex)

        path=self.get_file_name('condor', extra=ex, ext='condor')
        return path

    def _set_root_dir(self, root_dir=None):
        """
        set root dir, perhaps taking the default
        """
        if root_dir is None:
            root_dir=get_default_root_dir()

        root_dir=os.path.expandvars(root_dir)
        root_dir=os.path.expanduser(root_dir)
        self._root_dir=root_dir

    def _set_run_dir(self):
        """
        set the run directory
        """
        self._run_dir=os.path.join(self._root_dir, self._run)

def get_default_root_dir():
    """
    The root directory, under which the output run
    directories will be placed
    """
    root_dir=os.environ['GMIX_MEDS_DATADIR']
    return root_dir

def get_chunks(ntot, nper):
    """
    get a chunklist 

    [ [beg1,end1], [beg2,end2], ...]

    these are not like python slices where it really means [beg1,end1)

    """
    indices=numpy.arange(ntot)
    nchunk=ntot//nper

    nleft=ntot % nper
    if nleft > 1:
        nchunk += 1

    chunk_list=[]

    for i in xrange(nchunk):

        beg=i*nper
        end=(i+1)*nper-1
        
        if end > (ntot-1):
            end=ntot-1

        chunk_list.append( [beg,end] )

    return chunk_list

def get_condor_head_template():
    text="""
Universe        = vanilla

Notification    = Never 

# Run this exe with these args
Executable      = %(master_script)s

#Image_Size      = 1000000
Image_Size      = 1500000

GetEnv = True

kill_sig        = SIGINT

#requirements = (cpu_experiment == "star") || (cpu_experiment == "phenix")
#requirements = (cpu_experiment == "star")

+Experiment     = "astro"\n\n"""
    return text

def get_condor_job_template():
    text="""
+job_name = "%(job_name)s"
Arguments = %(config_file)s %(beg)d %(end)d %(out_file)s %(log_file)s %(meds_files_spaced)s
Queue\n"""
    return text

def get_wq_template():
    text="""
command: |
    source ~/.bashrc
    source ~/shell_scripts/gmix-meds-prepare.sh

    config_file="%(config_file)s"
    meds_files="%(meds_files_spaced)s"
    beg="%(beg)s"
    end="%(end)s"
    out_file="%(out_file)s"
    log_file="%(log_file)s"

    master_script=%(master_script)s

    $master_script $config_file $beg $end $out_file $log_file "$meds_files"

job_name: "%(job_name)s"\n"""

    return text

def get_concat_wq_template(noblind=False, verify=False):

    if noblind:
        noblind_line="            --noblind              \\"
    else:
        noblind_line="                                   \\"
    if verify:
        verify_line ="            --verify               \\"
    else:
        verify_line ="                                   \\"



    text="""
command: |
    source ~/.bashrc
    source ~/shell_scripts/gmix-meds-prepare.sh

    run="%(run_name)s"
    config_file="%(config_file)s"
    meds_files="%(meds_files_spaced)s"

    bands="%(bands_csv)s"
    nper="%(nper)s"
    sub_dir="%(sub_dir)s"

    python -u $GMIX_MEDS_DIR/bin/gmix-meds-collate     \\
{noblind_line}
{verify_line}
            --nper    "$nper"      \\
            --sub-dir "$sub_dir"   \\
            --bands   "$bands"     \\
            "$run"                 \\
            "$config_file"         \\
            $meds_files

mode: bynode
job_name: "%(job_name)s"\n""".format(noblind_line=noblind_line,
                                     verify_line=verify_line)

    return text

def get_oracle_wq_template(create=False, noblind=False):
    """
    des specific

    sub_dir is assumed to correspond to tilename
    """

    if noblind:
        noblind_line="            --noblind              \\"
    else:
        noblind_line="                                   \\"
    if create:
        create_line ="            --create              \\"
    else:
        create_line ="                                   \\"


    text="""
command: |
    source ~/.bashrc
    source ~/shell_scripts/gmix-meds-prepare.sh

    run="%(run_name)s"
    table_name="%(run_name)s"
    tilename="%(sub_dir)s"

    python -u $GMIX_MEDS_DIR/bin/gmix-meds-make-oracle \\
{noblind_line}
{create_line}
            "$run"                \\
            "$table_name"         \\
            "$tilename"

mode: bynode
job_name: "%(job_name)s"\n""".format(noblind_line=noblind_line,
                                     create_line=create_line)

    return text



def get_master_script_text():
    text="""#!/bin/bash
function go {
    hostname
    nproc=$(cat /proc/cpuinfo  | grep processor | wc -l )
    ut=$(uptime)
    echo "nproc: $nproc uptime: $ut"

    python -u $GMIX_MEDS_DIR/bin/gmix-fit-meds     \\
            --obj-range $beg,$end                  \\
            --work-dir $tmpdir                     \\
            $config_file $out_file "${meds_files[@]}"
    
    exit_status=$?

}

if [ $# -lt 6 ]; then
    echo "error: config_file beg end out_file log_file meds_file1 [meds_file2 ...]"
    exit 1
fi

args=("$@")

config_file=${args[0]}
beg=${args[1]}
end=${args[2]}
out_file=${args[3]}
log_file=${args[4]}
meds_files=( "${args[@]:5}" )

if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
    tmpdir=$_CONDOR_SCRATCH_DIR
else
    tmpdir=$TMPDIR
    mkdir -p $tmpdir
fi

outdir=$(dirname $out_file)
mkdir -p $outdir

lbase=$(basename $log_file)
tmplog="$tmpdir/$lbase"

go &> "$tmplog"
mv -v "$tmplog" "$log_file"

exit $exit_status

    \n"""

    return text

class StagedOutFile(object):
    """
    A class to represent a staged file

    If tmpdir=None no staging is performed and the original file
    path is used

    parameters
    ----------
    fname: string
        Final destination path for file
    tmpdir: string, optional
        If not sent, or None, the final path is used and no staging
        is performed
    must_exist: bool, optional
        If True, the file to be staged must exist at the time of staging
        or an IOError is thrown. If False, this is silently ignored.
        Default False.

    examples
    --------

    # using a context for the staged file
    fname="/home/jill/output.dat"
    tmpdir="/tmp"
    with StagedOutFile(fname,tmpdir=tmpdir) as sf:
        with open(sf.path,'w') as fobj:
            fobj.write("some data")

    # without using a context for the staged file
    sf=StagedOutFile(fname,tmpdir=tmpdir)
    with open(sf.path,'w') as fobj:
        fobj.write("some data")
    sf.stage_out()

    """
    def __init__(self, fname, tmpdir=None, must_exist=False):
        self.final_path=fname
        self.tmpdir=tmpdir
        self.must_exist=must_exist

        self.was_staged_out=False

        if tmpdir is None:
            self.is_temp=False
            self.path=self.final_path
        else:
            self.is_temp=True

            if not os.path.exists(tmpdir):
                os.makedirs(tmpdir)

            bname=os.path.basename(fname)
            self.path=os.path.join(tmpdir, bname)

            if self.path == self.final_path:
                # the user sent tmpdir as the final output dir, no
                # staging is performed
                self.is_temp=False

    def stage_out(self):
        """
        if a tempdir was used, move the file to its final destination

        note you normally would not call this yourself, but rather use a
        context, in which case this method is called for you

        with StagedOutFile(fname,tmpdir=tmpdir) as sf:
            #do something
        """
        import shutil

        if self.is_temp and not self.was_staged_out:
            if not os.path.exists(self.path):
                if self.must_exist:
                    raise IOError("temporary file not found:",self.path)
            else:
                if os.path.exists(self.final_path):
                    print("removing existing file:",self.final_path)
                    os.remove(self.final_path)

                makedir_fromfile(self.final_path)
                print("staging out",self.path,"->",self.final_path)
                shutil.move(self.path,self.final_path)

        self.was_staged_out=True

    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        self.stage_out()

class StagedInFile(object):
    """
    A class to represent a staged file

    If tmpdir=None no staging is performed and the original file
    path is used

    parameters
    ----------
    fname: string
        original file location
    tmpdir: string, optional
        If not sent, the TMPDIR environment variable is used.

    examples
    --------

    # using a context for the staged file
    fname="/home/jill/output.dat"
    tmpdir="/tmp"
    with StagedInFile(fname,tmpdir=tmpdir) as sf:
        with open(sf.path) as fobj:
            # read some data

    # without using a context for the staged file
    sf=StagedInFile(fname,tmpdir=tmpdir)
    with open(sf.path) as fobj:
        # read some data
    sf.cleanup()
    """
    def __init__(self, fname, tmpdir=None, funpack=False):

        self.funpack=funpack
        self.original_path=fname
        self._set_tmpdir(tmpdir=tmpdir)
        self._set_local_path()

        self._stage_in()

    def _set_tmpdir(self, tmpdir=None):
        if tmpdir is None:
            if not 'TMPDIR' in os.environ:
                raise RuntimeError("TMPDIR must be set, or send tmpdir=")
            tmpdir=os.environ['TMPDIR']
        self.tmpdir=tmpdir

    def _set_local_path(self):
        bname=os.path.basename(self.original_path)
        self.path=os.path.join(self.tmpdir, bname)

        if self.path == self.original_path:
            raise ValueError("tmp path is same as original path")

        if self.funpack:
            self.path = self.path.replace('.fits.fz','.fits')

    def _stage_in(self):
        """
        make a local copy of the file
        """
        import shutil

        if not os.path.exists(self.original_path):
            raise IOError("file not found:",self.original_path)

        if os.path.exists(self.path):
            print("removing existing file:",self.path)
            os.remove(self.path)
        else:
            makedir_fromfile(self.path)

        if self.funpack:
            command='funpack -O %s %s' % (self.path, self.original_path)
            print(command)
            res=os.system(command)
            if res != 0:
                raise RuntimeError("error funpacking")
        else:
            print("staging in",self.original_path,"->",self.path)
            shutil.copy(self.original_path,self.path)

    def cleanup(self):
        if os.path.exists(self.path):
            print("removing temporary file:",self.path)
            os.remove(self.path)

    def __del__(self):
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.cleanup()

def makedir_fromfile(fname):
    dname=os.path.dirname(fname)
    try_makedir(dname)

def try_makedir(dir):
    if not os.path.exists(dir):
        try:
            print("making directory:",dir)
            os.makedirs(dir)
        except:
            # probably a race condition
            pass

class MakerBase(dict):
    """
    create=True means write the oracle create table sql file
    """
    def __init__(self, run_name, config_file, meds_files,
                 root_dir=None, sub_dir=None, missing=False,
                 bands=None,
                 noblind=False,
                 create=False,
                 nper=DEFAULT_NPER):
        self['run_name']=run_name
        self['config_file']=config_file

        self['meds_file_list'] = meds_files
        self['meds_files_spaced']=' '.join(meds_files)

        self['root_dir']=root_dir
        self['sub_dir']=sub_dir
        self['missing']=missing
        self['nper']=nper

        self['nbands'] = len(meds_files)

        self['noblind']=noblind
        self['create']=create

        if bands is None:
            bands = [str(i) for i in xrange(self['nbands'])]
        else:
            assert len(bands)==self['nbands'],"wrong number of bands: %d" % len(bands)
        self['bands']=bands
        self['bands_csv'] = ','.join(bands)


        self._files=Files(self['run_name'],
                          root_dir=self['root_dir'])
        self['master_script'] = self._files.get_script_file()

        self._load_config()
        self._count_objects()

    def write(self):
        """
        write master script and wq yaml scripts

        over-ride this method, calling this one with
        super to write the master script and  then write the
        specific files for the child object
        """
        self._make_dirs()
        self._write_master_script()
        self._write_collate_wq()
        self._write_collate_wq(verify=True)
        self._write_oracle_wq()
        self._write_oracle_loaders()

    def _write_master_script(self):
        """
        write the master script
        """
       
        sfile=self['master_script']
        print("writing master script:",sfile)
        with open(sfile,'w') as fobj:
            master_text=get_master_script_text()
            fobj.write(master_text)

        cmd='chmod 755 %s' % sfile
        print(cmd)
        os.system(cmd)

    def _write_collate_wq(self, verify=False):
        """
        write a script to do the collation
        """


        job_name=[self['run_name']]
        if verify:
            extra=['verify']
        else:
            extra=['collate']

        if self['sub_dir'] is not None:
            extra += [self['sub_dir']]
            job_name += [self['sub_dir']]

        extra='-'.join(extra)
        job_name = '-'.join(job_name)

        self['job_name']=job_name

        text=get_concat_wq_template(noblind=self['noblind'], verify=verify)
        text = text % self

        wq_file=self._files.get_file_name('wq', extra=extra, ext='yaml')
        print("writing:",wq_file)
        with open(wq_file,'w') as fobj:
            fobj.write(text)

    def _write_oracle_wq(self, verify=False):
        """
        write a script to make the oracle input files
        """


        job_name=[self['run_name']]
        extra=['oracle']

        if self['sub_dir'] is not None:
            extra += [self['sub_dir']]
            job_name += [self['sub_dir']]

        extra='-'.join(extra)
        job_name = '-'.join(job_name)

        self['job_name']=job_name

        text=get_oracle_wq_template(noblind=self['noblind'],
                                    create=self['create'])
        text = text % self

        wq_file=self._files.get_file_name('wq', extra=extra, ext='yaml')
        print("writing:",wq_file)
        with open(wq_file,'w') as fobj:
            fobj.write(text)


    def _write_oracle_loaders(self):
        """
        Write scripts to run all the oracle loading
        """
        odir=self._files.get_oracle_dir()
        if not os.path.exists(odir):
            os.makedirs(odir)

        load_file=os.path.join(odir, 'load-all.sh')
        load_epochs_file=os.path.join(odir, 'load-all-epochs.sh')

        print("writing:",load_file)
        with open(load_file,'w') as fobj:

            fobj.write("""
echo -n "password: "
read -s password
echo
for f in $(ls *.ctl | grep -v epochs | sort); do
    echo "loading $f"
    time run-sqlldr $f esheldon $password 10000
done &> load-all.log\n""")

        print("writing:",load_epochs_file)
        with open(load_epochs_file,'w') as fobj:
            fobj.write("""
echo -n "password: "
read -s password
echo
for f in $(ls *.ctl | grep epochs | sort); do
    echo "loading $f"
    time run-sqlldr $f esheldon $pass 10000
done &> load-all-epochs.log\n""")

    def _load_config(self):
        conf=read_yaml(self['config_file'])
        self.update(conf)

    def _count_objects(self):
        import meds
        fname=self['meds_file_list'][0]
        with meds.MEDS(fname) as meds_obj:
            nobj=meds_obj.size

        self['nobj']=nobj

    def _make_dirs(self):
        """
        make all the output directories

        to add new directories, over-ride this method, calling this one with
        super and then working with your directories

        """
        # no sub dir here on wq dir because this is for the collate scripts
        dirs=[self._files.get_wq_dir(),
              self._files.get_output_dir(sub_dir=self['sub_dir']),
              self._files.get_script_dir()]

        for dir in dirs:
            try_makedir(dir)


class CondorMaker(MakerBase):
    def write(self):
        """
        write master script and condor script
        """
        super(CondorMaker,self).write()
        self._write_condor_script()

    def _write_condor_script(self):
        """
        write the one big condor submit script
        """

        # we dump all the condor files into the same directory
        if self['missing']:
            extra='missing'
        else:
            extra=None

        condor_file=self._files.get_condor_file(sub_dir=self['sub_dir'],
                                                extra=extra)
        print(condor_file)

        head=get_condor_head_template()
        head=head % self


        nper=self['nper']
        nobj=self['nobj']

        chunklist = get_chunks(self['nobj'], self['nper'])

        nchunk=len(chunklist)

        ltemplate=get_condor_job_template()
        nwrite=0
        with open(condor_file,'w') as fobj:
            fobj.write(head)

            for split in chunklist:
                self['beg']=split[0]
                self['end']=split[1]
                self['out_file']=self._files.get_output_file(split,
                                                             sub_dir=self['sub_dir'])
                self['log_file']=self['out_file'].replace('.fits','.log')

                if self['missing'] and os.path.exists(self['out_file']):
                    continue

                job_name=[self['run_name']]
                if self['sub_dir'] is not None:
                    job_name.append(self['sub_dir'])
                job_name.append('%06d' % split[0])
                job_name.append('%06d' % split[1])

                self['job_name'] = '-'.join(job_name)

                job=ltemplate % self

                fobj.write(job)
                nwrite += 1

        if self['missing'] and nwrite==0:
            print("none were written, removing condor file")
            os.remove(condor_file)
        else:
            print("wrote",nwrite,"jobs")

    def _make_dirs(self):
        """
        make all the output directories
        """
        super(CondorMaker,self)._make_dirs()
        dir=self._files.get_condor_dir()
        if not os.path.exists(dir):
            print("making condor dir:",dir)
            os.makedirs(dir)


class WQMaker(MakerBase):
    def write(self):
        """
        write master script
        """
        super(WQMaker,self).write()
        self._write_wq_scripts()

    def _write_wq_scripts(self):
        """
        write the wq scripts
        """

        nper=self['nper']
        nobj=self['nobj']

        chunklist = get_chunks(self['nobj'], self['nper'])

        nchunk=len(chunklist)

        for split in chunklist:
            self._write_wq_file(split)

    def _write_wq_file(self, split):

        self['beg']=split[0]
        self['end']=split[1]
        self['out_file']=self._files.get_output_file(split, sub_dir=self['sub_dir'])
        self['log_file']=self['out_file'].replace('.fits','.log')

        if self['missing']:
            if os.path.exists(self['out_file']):
                return

            extra='missing'
        else:
            extra=None

        job_name=[]
        if self['sub_dir'] is not None:
            job_name.append(self['sub_dir'])
        job_name.append('%06d' % split[0])
        job_name.append('%06d' % split[1])

        self['job_name'] = '-'.join(job_name)

        print(wq_file)
        with open(wq_file,'w') as fobj:
            text=get_wq_template()
            text = text % self
            fobj.write(text)

def get_temp_dir():
    tmpdir=os.environ.get('_CONDOR_SCRATCH_DIR',None)
    if tmpdir is None:
        tmpdir=os.environ.get('TMPDIR',None)
        if tmpdir is None:
            import tempfile
            return tempfile.mkdtemp()

def read_yaml(config_path):
    """
    read from the file assuming it is yaml
    """
    import yaml
    with open(config_path) as fobj:
        conf=yaml.load(fobj)
    return conf

def get_default_config_file(name):
    """
    assume the config is located at $GMIX_MEDS_DIR/config/{name}.yaml

    for most tasks, it is not necessary to store the config files at that location
    """
    base=os.environ['GMIX_MEDS_DIR']
    path=os.path.join(base, 'share', 'gmix-meds-config', '%s.yaml' % name)
    return path


def read_default_config(name):
    """
    assume the config is located at $GMIX_MEDS_DIR/config/{name}.yaml

    for most tasks, it is not necessary to store the config files at that location
    """
    path=get_default_config_file(name)
    return read_yaml(path)

def get_default_runconfig_file(name):
    """
    assume the config is located at $GMIX_MEDS_DIR/runconfig/{name}.yaml

    for most tasks, it is not necessary to store the config files at that location
    """
    base=os.environ['GMIX_MEDS_DIR']
    path=os.path.join(base, 'share', 'gmix-meds-runconfig', '%s.yaml' % name)
    return path


def read_default_runconfig(name):
    """
    assume the config is located at $GMIX_MEDS_DIR/runconfig/{name}.yaml

    for most tasks, it is not necessary to store the config files at that location
    """
    path=get_default_runconfig_file(name)
    return read_yaml(path)



