# flagging
IMAGE_FLAGS=2**0
PSF_FIT_FAILURE=2**1
GAL_FIT_FAILURE=2**2
BOX_SIZE_TOO_BIG=2**3
LOW_PSF_FLUX=2**4
UTTER_FAILURE=2**5
PSF_FLUX_FIT_FAILURE=2**6
NO_CUTOUTS=2**7
NO_ATTEMPT=2**30

# defaults
DEFVAL = -9999
PDEFVAL = 9999
BIG_DEFVAL = -9.999e9
BIG_PDEFVAL = 9.999e9

# logging
LOGGERNAME = __name__.split('.')[0]

# running code
_CHECKPOINTS_DEFAULT_MINUTES=[0,30,60,110]
