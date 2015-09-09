# flagging

# this is here for testing, it will be removed one day
USE_OLD_FLAGS = False

if not USE_OLD_FLAGS:
    # flags used by NGMixer
    BAD_OBJ              = 2**25
    IMAGE_FLAGS          = 2**26
    NO_CUTOUTS           = 2**27
    BOX_SIZE_TOO_BIG     = 2**28
    UTTER_FAILURE        = 2**29
    NO_ATTEMPT           = 2**30

    # flags for fitting codes
    PSF_FIT_FAILURE      = 2**0
    GAL_FIT_FAILURE      = 2**1
    PSF_FLUX_FIT_FAILURE = 2**2
    LOW_PSF_FLUX         = 2**3
else:
    IMAGE_FLAGS=2**8
    NO_CUTOUTS=2**0
    BOX_SIZE_TOO_BIG=2**4
    UTTER_FAILURE=2**7
    NO_ATTEMPT=2**30

    PSF_FIT_FAILURE=2**1
    GAL_FIT_FAILURE=2**3
    PSF_FLUX_FIT_FAILURE=2**9
    LOW_PSF_FLUX=2**6

# defaults
DEFVAL = -9999
PDEFVAL = 9999
BIG_DEFVAL = -9.999e9
BIG_PDEFVAL = 9.999e9

# logging
LOGGERNAME = __name__.split('.')[0]

# running code
_CHECKPOINTS_DEFAULT_MINUTES=[0,30,60,110]

PSF_IND_FIELD='ind_psf'
PSF_IM_FIELD='psf_im'
