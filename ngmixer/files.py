"""
matt generally sprinkled path determination throughout
the different classes.

I'm (ESS) beginning to move things into this central area
"""
from __future__ import print_function
import os
import copy
import numpy
import esutil as eu

def read_config(fname):
    """
    read a config file, making sure the run name
    and file name match

    expect name 'run-{runstring}.yaml'
    """
    data=eu.io.read(fname)

    bname=os.path.basename(fname)
    bname=bname.replace('.yaml','')
    fs=bname.split('-')[1:]

    rs = '-'.join(fs)

    if rs != data['run']:
        raise ValueError("run in config '%s' does not match "
                         "filename: '%s'" % (data['run'], bname))

    return data


def extract_meds_info(meds_file):
    """
    extract information from a Y3+ meds file path
    """

    bname=os.path.basename(meds_file)

    fs = bname.split('_')

    tilename=fs[0]
    reqatt = fs[1]
    band = fs[2]

    # use underscore for consistency
    meds_id='%s_%s' % (tilename,reqatt)

    return {
        'meds_id':meds_id,       # tilename_reqatt
        'tilename':tilename,     # coadd tile
        'reqatt':reqatt,         # combination of reqnum and attnum, e.g r2577p01
        'band':band,             # the filter band
    }

def get_meds_id(meds_file):
    """
    e.g. DES0205-3706_r2577p01
    """
    info = extract_meds_info(meds_file)
    return info['meds_id']

def get_temporary_meds_dir(meds_file):
    """
    this is the directory where we will sync the meds files
    and psf files.  This is not the "usual" place where DESDM
    puts things
    """

    if 'MEDS_DIR' not in os.environ:
        raise RuntimeError("MEDS_DIR environment variable not set")

    info = extract_meds_info(meds_file)

    return os.path.join(
        os.environ['MEDS_DIR'],
        info['meds_id'],
    )

def get_temporary_psf_dir(meds_file):
    """
    this is the directory where we will sync the meds files
    and psf files.  This is not the "usual" place where DESDM
    puts things
    """

    dir = get_temporary_meds_dir(meds_file)
    return os.path.join(dir, 'psfs')


def get_ngmixer_output_dir():
    #return '$NGMIXER_OUTPUT_DIR'

    if 'NGMIXER_OUTPUT_DIR' not in os.environ:
        raise ValueError("environment variable NGMIXER_OUTPUT_DIR not set")
    return os.environ['NGMIXER_OUTPUT_DIR']

def get_nbrs_dir(meds_file, run):
    """
    get the directory holding the nbrs info for the
    indicated meds file
    """
    meds_vers=get_meds_id(meds_file)
    return os.path.join(
        get_ngmixer_output_dir(),
        run,
        meds_vers,
    )


def get_nbrs_file(meds_file, run):
    """
    get the path to a nbrs file given a MEDS file
    """
    dir=get_nbrs_dir(meds_file, run)
    info=extract_meds_info(meds_file)

    info['run'] = run
    fname = '%(meds_id)s-%(run)s-nbrslist.fits'
    fname = fname % info

    return os.path.join(
        dir,
        fname,
    )

def get_fof_file(meds_file, run):
    """
    get the path to a nbrs FOF file given a MEDS file
    """
    dir=get_nbrs_dir(meds_file, run)
    info=extract_meds_info(meds_file)

    info['run'] = run
    fname = '%(meds_id)s-%(run)s-nbrsfofs.fits'
    fname = fname % info

    return os.path.join(
        dir,
        fname,
    )


def get_chunk_dir(meds_file, run, rng):
    """
    get the directory holding the output for a chunk
    """
    meds_vers=get_meds_id(meds_file)
    return os.path.join(
        get_ngmixer_output_dir(),
        run,
        meds_vers,
        'chunk%06d-%06d' % tuple(rng),
    )


def get_chunk_file(meds_file, run, rng, ext='fits'):
    """
    get the path to a nbrs file given a MEDS file
    """
    dir=get_chunk_dir(meds_file, run, rng)

    info=extract_meds_info(meds_file)

    info['run'] = run
    info['start']=rng[0]
    info['end']=rng[1]
    info['ext']=ext

    fname = '%(meds_id)s-%(run)s-%(start)06d-%(end)06d.%(ext)s'
    fname = fname % info

    return os.path.join(
        dir,
        fname,
    )




def makedirs_fromfile(fname, ntry=2):
    import time

    fname=os.path.expandvars(fname)
    ok=False

    for i in xrange(ntry):
        try:
            eu.ostools.makedirs_fromfile(
                fname,
                verbose=True,
                allow_fail=False,
            )
            ok=True
        except err:
            print("failed to make directory:",os.path.dirname(fname))
            print("trying again after 1 second")
            time.sleep(1)
            pass

    if not ok:
        raise err

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
        if tmpdir is not None:
            self.tmpdir=os.path.expandvars(tmpdir)
        else:
            self.tmpdir = tmpdir
        self.must_exist=must_exist

        self.was_staged_out=False

        fpath = os.path.split(fname)[0]
        if fpath == '':
            fpath = '.'

        if self.tmpdir is None or os.path.samefile(self.tmpdir,fpath):
            self.is_temp=False
            self.path=self.final_path
        else:
            self.is_temp = True

            if not os.path.exists(self.tmpdir):
                os.makedirs(self.tmpdir)

            bname=os.path.basename(fname)
            self.path=os.path.join(self.tmpdir, bname)

            # just be really sure...
            assert os.path.samefile(os.path.split(self.path)[0],fpath) == False

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

def read_yaml(config_path):
    """
    read from the file assuming it is yaml
    """
    import yaml

    fname=os.path.expandvars(config_path)
    fname=os.path.expanduser(fname)
    with open(fname) as fobj:
        conf=yaml.load(fobj)
    return conf

def get_temp_dir():
    tmpdir=os.environ.get('_CONDOR_SCRATCH_DIR',None)
    if tmpdir is None:
        tmpdir=os.environ.get('TMPDIR',None)
        if tmpdir is None:
            import tempfile
            return tempfile.mkdtemp()
