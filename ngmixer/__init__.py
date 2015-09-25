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

#######################################################
# setup logger
import logging
logging.basicConfig(format='%(message)s')
log = logging.getLogger(defaults.LOGGERNAME)
log.setLevel(logging.INFO)

#######################################################
# setup fitter dict
# each time you create a new fitter class, add it to this dict
FITTERS = {}

# boot fitters
from . import bootfit
FITTERS['max-ngmix-boot'] = bootfit.MaxNGMixBootFitter
FITTERS['metacal-ngmix-boot'] = bootfit.MetacalNGMixBootFitter
FITTERS['isamp-ngmix-boot'] = bootfit.ISampNGMixBootFitter
