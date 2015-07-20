import os
import glob
from distutils.core import setup

scripts=[]


scripts=[os.path.join('bin',s) for s in scripts]

conf_files=glob.glob('config/*.yaml')
runconf_files=glob.glob('runconfig/*.yaml')

data_files=[]
for f in conf_files:
    d=os.path.dirname(f)
    n=os.path.basename(f)
    d=os.path.join('share','ngmixer-config')

    data_files.append( (d,[f]) )

for f in runconf_files:
    d=os.path.dirname(f)
    n=os.path.basename(f)
    d=os.path.join('share','ngmixer-runconfig')

    data_files.append( (d,[f]) )



setup(name="ngmixer", 
      version="0.1.0",
      description="Run ngmix on data",
      license = "GPL",
      author="Matthew R. Becker, Erin Scott Sheldon",
      author_email="becker.mr@gmail.com, erin.sheldon@gmail.com",
      scripts=scripts,
      data_files=data_files,
      packages=['ngmixer'])
