#For creating package
from pkg_resources import get_distribution, DistributionNotFound

try:
    dist = get_distribution("CandyLLM")
except DistributionNotFound:
    __version__ = "Please install CandyLLM with setup.py"
else:
    __version__ = dist.version

#Package Modules
from .hub import*
