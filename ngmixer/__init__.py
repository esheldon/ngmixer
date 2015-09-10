#######################################################
# imports
from . import files
from . import ngmixing
from . import medsio
from . import priors
from . import util
from . import defaults
from . import bootfit
from . import mofngmixing
from . import megamixer
from . import slacmegamixer

#######################################################
# setup logger
import logging
logging.basicConfig(format='%(message)s')
log = logging.getLogger(defaults.LOGGERNAME)
log.setLevel(logging.INFO)

#######################################################
# setup image i/o dict
# each time you create a new image i/o class, add it to this dict
IMAGEIO = {}

# MEDS formats
from . import medsio
IMAGEIO['meds'] = medsio.MEDSImageIO
IMAGEIO['meds-sv'] = medsio.SVMEDSImageIO
IMAGEIO['meds-sv-mof'] = medsio.SVMOFMEDSImageIO
IMAGEIO['meds-y1'] = medsio.Y1MEDSImageIO

# MEDS sim formats
from . import simpsimmedsio
IMAGEIO['meds-simp-sim'] = simpsimmedsio.SimpSimMEDSImageIO

#######################################################
# setup fitter dict
# each time you create a new fitter class, add it to this dict
FITTERS = {}

# boot fitters
from . import bootfit
FITTERS['max-ngmix-boot'] = bootfit.MaxNGMixBootFitter
FITTERS['metacal-ngmix-boot'] = bootfit.MetacalNGMixBootFitter
FITTERS['isamp-ngmix-boot'] = bootfit.ISampNGMixBootFitter
