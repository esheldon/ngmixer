# ngmixer
code to run ngmix on data

documentation
-----------------

Please see the [wiki](https://github.com/esheldon/ngmixer/wiki) for detailed documentation

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
* render_ngmix_nbrs.py - code to render nbrs for ngmix MOF

Getting Started
---------------

#### Setup the I/O

You will need to make sure you have an appropriate MEDS I/O class. 

The doc string for the base class `ImageIO` indicates how to write a child class that implements the 
proper interface. In general, the `ImageIO` classes are iterators that yield a list of things to fit.

For DES MEDS files, interfaces have already been written. These are in `imageio/desmedsio.py`. The Y1 class 
should work out-of-the box (called Y1DESMEDSImageIO). 

WARNING: We have not updated the image flags for Y1. See code here 
https://github.com/esheldon/ngmixer/blob/master/ngmixer/imageio/desmedsio.py#L334. This needs to be done.

Whenever you implement a new interface, you have to register it in the `__init__.py` in the imageio sub-package.

#### Setup the Fitter

Similar to the `ImageIO` classes above, you need to have an appropriate fitter. 

The doc strings on all of the methods of `BaseFitter` in `fitting.py` describe what needs to be implemented. 

The ngmix fitters are in `bootfit.py`. There are some max-like and metacal fitters there 
already. Any new fitter has to be registered in the main package `__init__.py`.

#### Running

Once you have a fitter and an imageio class, you can run the code. You will need any configs etc. 
setup properly. 

For ngmix, see the metacal and MOF examples on the main repo wiki. Both of these examples eventally use the 
`ngmixit` command to run the code.

#### Big Runs

For large runs, it makes sense to use the megamixer, but is not needed. The megamixer parallelizes the run by dividing 
each input file to `ngmixit` into a bunch of smaller chunks (typically ~300 per coadd tile in DES). Job scripts and 
configs are written for each of the ~300 chunks. The megamixer can then submit these jobs to the queue and recombine the 
output when the jobs are done. It can also do things like tar up the log files to archive runs and collect the final 
outputs into a series of flat files. 

In order to parallelize the run, the megamixer uses the `--fof-range` input option to `ngmixit` to specify ranges of 
objects in each file to fit. If you are not doing a MOF run, this parameter defaults to one object per FoF 
(i.e., it makes dummy FoFs). Unfortunately, this whole scheme currently requires that MEDS files be input into the code. 
Fixing this dependence def needs to happen, but has not yet.

To call the megamixer, use the command line util `megamixit`. Again see the wiki for details.

Implementing a new megamixer is not that hard at all. See the base class here

https://github.com/esheldon/ngmixer/blob/master/ngmixer/megamixer/megamixer.py

You need to supply implementations of `write_job_script` and `run_chunk`. A default 
implementation is in the file above. Note that the jobs must be run in the directory 
given by `dr = self.get_chunk_output_dir(files,chunk,rng)`. The megamixer uses relative paths so that the whole set 
of chunks can be relocated. Finally register the new megamixer here (https://github.com/esheldon/ngmixer/blob/master/bin/megamixit#L66).

The SLAC megamixer, which is a good example, is here

https://github.com/esheldon/ngmixer/blob/master/ngmixer/megamixer/slacmegamixer.py

Be sure to process the extra_commands option. One can use this to supply a path to the 
activate function on, e.g., virtualenvs. For example, an `extra_cmds.txt` might look like

```bash
source /nfs/slac/des/fs1/g/sims/beckermr/DES/y1_ngmix/tb_y1a1_v01c/work011/bin/activate
```

#### Multi-object Fitting

The multi-object fitting (MOF) built into ngmixer is contained in the file `mofngmixing.py` and is iterative. For each 
object on each iteration, its nbrs are subtracted and the object is refit. The `ImageIO` class is responsible for 
yeilding all objects which need to be fit together as an entire group. The fitting class is responsible for actually 
rendering the nrbs, removing them and then fitting the central. 

Thus for the multi-object fitting (MOF), an extra, pre-processing step has to be taken where lists of nbrs for 
each object are built. Plus other things need to be changed in the `ImageIO` and fitting classes to support the 
propagation of the nbrs information and fit parameters. 

##### Build the Nbrs Lists and FoFs

You need to make the nbrs files for each coadd tile. This file is built off of the MEDS files made for each coadd tile 
and is used to decide which objects have nbrs to be fit and which ones do not. 

The code which builds the nbrs files is located in `nbrsfofs.py`. The code works by first constructing for each a object 
(i.e., the central object) a list of nbrs which 'overlap' with the central. For the MEDS files, it is assumed that the 
stamp sizes themsselves indicate the extent of each object and so the overlaps are built by intersecting the stamps with 
some buffer. Once each object has a list of nbr objects, a FoF group of objects is built by linking objects if either is 
in the nbrs list of the other. Finally, the fits of the objects are done by fitting all things in a FoF together and 
parallelizing the entire run over the set of FoFs. (For the DES, including isolated objects as FoFs of size one, 
typically only 40% of the FoFs have two or more objects.)

The command line utility for doing these steps is `ngmixer-meds-make-nbrs-data`. It needs a config file similar to 
the one here https://github.com/esheldon/ngmix-y1-config/blob/master/nbrs_config/nbrs001.yaml. See the help menu of 
`ngmixer-meds-make-nbrs-data` for how to run it. Two nbrs files get written for each coadd tile in the `$DESDATA` 
area and are for example 

```
${DESDATA}/EXTRA/meds/${meds_version}/nbrs-data/${nbrs_version}/${coadd_runn}/\
    ${tile}-meds-${meds_version}-nbrslist-${nbrs_version}.fits

${DESDATA}/EXTRA/meds/${meds_version}/nbrs-data/${nbrs_version}/${coadd_runn}/\
    ${tile}-meds-${meds_version}-nbrsfofs-${nbrs_version}.fits
```
These are for the Y1 pipeline. The Y2+ pipeline may be different. The first file above is the list of nbrs and the second file above is the list of FoF membership constructed from the nbrs. 

##### Add the Nbrs Metadata in the ImageIO class

The `ImageIO` interface used for reading the image data has to be aware of the nbrs files made above and process them so 
that they include the needed information in the `meta_data` attached to the images in order to run the MOF fits. All of 
this is already done for MEDS files. See the doc string on the base `ImageIO` class (mentioned above) in order to see 
what needs to be added if you implement a new interface.

##### Make the Fitter Aware of the Nbrs

Any fitting class with the MOF is responsible for rendering the nbrs for each central and subtracting them out of the 
centrals stamps. For ngmix-based MOF, this has already been done. If you are using another fitter, you will have to 
code this up yourself. Again see the base fitter docs for how to do this. 

##### Run the Code

Once the two files above have been built, the code can be run. For the MOF the two files above have to be specified to 
`ngmixit` with the  `--nbrs-file` (for the `nbrslist`) and the `--fof-file` (for the `nbrsfofs`). If you are using the 
`megamixer`, then this all gets done for you. You will also need to set the following in the fitting config fed to 
`ngmixit` (assuming you are using ngmix)

```yaml
read_wcs: True
read_me_wcs: False

model_nbrs: True
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

prop_sat_starpix: True
flag_y1_stellarhalo_masked: True
intr_prof_var_fac: 0.0

region: "mof"
```

This config setting is for Y1. The `prop_sat_starpix` and `flag_y1_stellarhalo_masked` options may 
not apply to future data releases. 

#### Using the MOF Results
See the repo wiki.
