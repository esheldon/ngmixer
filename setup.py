import os
import glob
from distutils.core import setup
import subprocess

scripts = glob.glob('./bin/*')
scripts = [os.path.basename(f) for f in scripts if f[-1] != '~']
scripts = [os.path.join('bin',s) for s in scripts]

# get git hash, if this is a git clone
try:
    githash = str(subprocess.check_output(["git","log","-n 1"]).split()[1])
    dirty = str(subprocess.check_output(['git','status','--porcelain']))
    if len(dirty) > 0:
        githash += '-dirty'
except subprocess.CalledProcessError:
    githash=None

# add to package
os.system('echo "#!/usr/bin/env python\nhash = \\"%s\\"\n\n" > ngmixer/githash.py' % githash)

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py

setup(
    name="ngmixer",
    version="v0.9.9",
    description="Run ngmix on data",
    license = "GPL",
    author="Matthew R. Becker, Erin Scott Sheldon",
    author_email="becker.mr@gmail.com, erin.sheldon@gmail.com",
    scripts=scripts,
    packages=['ngmixer','ngmixer.imageio','ngmixer.megamixer'],
    cmdclass={'build_py': build_py},
)

# return package to original state
os.system('echo "#!/usr/bin/env python\nhash = None\n\n" > ngmixer/githash.py')
