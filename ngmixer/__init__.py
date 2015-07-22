from . import files
from . import fitting
from . import medsio
from . import priors
from . import util
from . import defaults

# setup logger
import logging
logging.basicConfig(format='%(message)s')
log = logging.getLogger(defaults.LOGGERNAME)
log.setLevel(logging.INFO)

# setup image i/o dict
# each time you create a new image io class, add it to this dict
IMAGEIO = {}
IMAGEIO['MEDS'] = medsio.MEDSImageIO
IMAGEIO['SVMEDS'] = medsio.SVMEDSImageIO
IMAGEIO['SVMOFMEDS'] = medsio.SVMOFMEDSImageIO
IMAGEIO['Y1MEDS'] = medsio.Y1MEDSImageIO

