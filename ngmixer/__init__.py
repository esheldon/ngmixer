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


