### REMARK: here the order of importing is importand to avoid the circular importing

from .logger import setup_logging
logger = setup_logging()


from Models.LSTM import *
