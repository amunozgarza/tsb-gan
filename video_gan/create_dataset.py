import sys

import utils

from parameters import *
from inference import Tester


if __name__ == '__main__':
    config = get_parameters()
    config.command = 'python ' + ' '.join(sys.argv)
    print(config)
    trainer = Tester(config)
    trainer.create_dataset()
