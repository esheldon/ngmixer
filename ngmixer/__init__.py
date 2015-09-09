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
IMAGEIO['MEDS'] = medsio.MEDSImageIO
IMAGEIO['SVMEDS'] = medsio.SVMEDSImageIO
IMAGEIO['SVMOFMEDS'] = medsio.SVMOFMEDSImageIO
IMAGEIO['Y1MEDS'] = medsio.Y1MEDSImageIO

# MEDS sim formats
from . import simpsimmedsio
IMAGEIO['SIMPSIMMEDS'] = simpsimmedsio.SimpSimMEDSImageIO

#######################################################
# setup fitter dict
# each time you create a new fitter class, add it to this dict
FITTERS = {}

# boot fitters
from . import bootfit
FITTERS['MAXNGMIXBOOT'] = bootfit.MaxNGMixBootFitter
FITTERS['ISAMPNGMIXBOOT'] = bootfit.ISampNGMixBootFitter
