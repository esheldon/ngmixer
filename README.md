# ngmixer
code to run ngmix on data

Structure of Repo
-----------------

* imageio - sub-package to read images/MEDS
* megamixer - sub-package to setup runs
* fitting.py - base class for implementing fitters
* bootfit.py - child fitter class for ngmix bootstrappers
* ngmixing.py - base class for running fitter over data (uses fitters and imageio classes)
* mofngmixing.py - MOF for running fitter over data
* nbrsfofs.py - support utils for constructing FoFs for MOF
* priors.py - code to deal with ngmix priors (doesn't really belong here, but it had to go somewhere)
* util.py - utils of various kinds
* defaults.py - default flags. etc.
* files.py - I/O utils

Getting Started
---------------

#### Setup the I/O

You will need to make sure you have an appropriate MEDS I/O class. The ones for DES are in 
imageio/desmedsio.py. The Y1 class should work out-of-the box (called Y1DESMEDSImageIO). When 
you implement a new one, you have to register it in the `__init__.py` in the imageio sub-package.

WARNING: We have not updated the image flags for Y1. See code here https://github.com/esheldon/ngmixer/blob/master/ngmixer/imageio/desmedsio.py#L334. This needs to be done.

#### Setup the Fitter

Similar to above, you need to have an appropriate fitter. The ngmix ones we have so far are in bootfit.py. 
There are some max-like and metacal fitters there already. Any new fitter has to be registered in the main package `__init__.py`.

#### Running

Once you have a fitter and an imageio class, you can run the code. You will need any configs etc. 
setup properly. Start with the MOF Y1 configs (`ngmix-y1-014.yaml` in `esheldon/ngmix-y1-configs`) 
and make sure to set the following:

```yaml
read_wcs: False
read_me_wcs: False

model_nbrs: False
model_nbrs_method: 'subtract'
mof:
    maxabs_conv:  [1.0e-3,1.0e-3,1.0e-6,1.0e-6,1.0e-6,1.0e-6,1.0e-6,1.0e-6,1.0e-6]
    maxfrac_conv: [1.0e-6,1.0e-6,1.0e-6,1.0e-6,1.0e-6,1.0e-3,1.0e-3,1.0e-3,1.0e-3]
    maxerr_conv: 0.5
    max_itr: 15
    min_itr: 10
    min_useg_itr: 1
    write_convergence_data: False
    convergence_model: 'cm'

prop_sat_starpix: False
flag_y1_stellarhalo_masked: False
intr_prof_var_fac: 0.0

region: "cweight-nearest"
```

This turns off all of the MOF magic.

Then you can directly call `ngmixit` (which gets installed when the repo is installed.) This bit 
of code is just like the old script from gmix_meds. (See here https://github.com/esheldon/ngmixer/blob/master/bin/ngmixit).

#### Big Runs

For large runs, it makes sense to use the megamixer, but is not needed. The megamixer 
has both the old concat code that postprocessed runs and it sets up the jobs. To call it, 
use the command line util `megamixit`. For this you will need a run config (see the Y1 one here 
https://github.com/esheldon/ngmix-y1-config/blob/master/run_config/run-y1-014.yaml). You can ignore the 
`nbrs_version` tag. By default, if you set the `run` field in the run config to `y1-014` 
the code expects an ngmix config called `ngmix-y1-014.yaml`. You can set this by hand by 
setting the `ngmix_config` field in the run config.

You can call the megamixer like this (at SLAC)

```bash
#!/bin/bash

seed=3461743891
megamixit \
    --clobber \
    --system=slac \
    --queue=medium \
    --extra-cmds=extra_cmds.txt \
    --seed=${seed} \
    ${1} \
    run-y1-014.yaml \
    tiles.txt
```

The extra-cmds options puts any commands in the file it points to into the job scripts. 
Always give a seed. You then specify a run config and finally a list of tiles. The ${1} 
above is the command. To do a run you feed it three commands (assuming you have put the 
above code into a script called megamix.sh)

```
./megamix.sh setup   # builds all jobs and configs on disk
./megamix.sh run     # submits jobs to medium queue at SLAC
./megamix.sh collate # when all jobs are done run this to recombine the files to single outputs per tile
./megamix.sh link    # links all outputs to ${run}/output
```
Implementing a new megamixer is not that hard at all. See the base class here

https://github.com/esheldon/ngmixer/blob/master/ngmixer/megamixer/megamixer.py

You need to supply implementations of write_job_script and run_chunk. A default 
implementation is in the file above. Note that the jobs get run in the directory 
given by `dr = self.get_chunk_output_dir(files,chunk,rng)`. Finally register the new megamixer here

https://github.com/esheldon/ngmixer/blob/master/bin/megamixit#L66

The SLAC one is here

https://github.com/esheldon/ngmixer/blob/master/ngmixer/megamixer/slacmegamixer.py

Be sure to process the extra_commands option. I use this to supply a path to the 
activate function on virtualenvs I build for each version. So my extra_cmds.txt looks like

```bash
source /nfs/slac/des/fs1/g/sims/beckermr/DES/y1_ngmix/tb_y1a1_v01c/work011/bin/activate
```
