from main import main
from types import SimpleNamespace
from glob import glob

import logging

args = SimpleNamespace()
args.FILES = glob('data/MUTAG/*.gml')
args.num_iterations = 2
args.labels = 'data/MUTAG/Labels.txt'
args.filtration = 'sublevel'
args.grid_search = True

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('P-WL')

main(args, logger)
