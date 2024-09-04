## # Import modules/files to make them available at the package level
from Ingestion.folder import *
from Ingestion.ingestion import *

## also used for setting varibale in the package namespace


# Making it a package I can set up logging configuration specific to it
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Ingestion Folder initialized")
logger.info("Functions for the screen recording and folder creation are available")
