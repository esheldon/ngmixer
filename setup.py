import os
import glob
from setuptools import setup
import subprocess

scripts=glob.glob('./bin/*')
scripts = [os.path.basename(f) for f in scripts if f[-1] != '~']
scripts=[os.path.join('bin',s) for s in scripts]

# get git hash
githash = subprocess.check_output(["git","log","-n 1"]).split()[1]
print githash

setup(name="ngmixer", 
      version="0.1.0",
      description="Run ngmix on data",
      license = "GPL",
      author="Matthew R. Becker, Erin Scott Sheldon",
      author_email="becker.mr@gmail.com, erin.sheldon@gmail.com",
      scripts=scripts,
      packages=['ngmixer','ngmixer.imageio','ngmixer.megamixer'])
