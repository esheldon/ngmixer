from __future__ import print_function
import os
import copy
import numpy

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
    with open(config_path) as fobj:
        conf=yaml.load(fobj)
    return conf

def get_temp_dir():
    tmpdir=os.environ.get('_CONDOR_SCRATCH_DIR',None)
    if tmpdir is None:
        tmpdir=os.environ.get('TMPDIR',None)
        if tmpdir is None:
            import tempfile
            return tempfile.mkdtemp()
