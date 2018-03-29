import os

import numpy as np


def generate_range(self, output_dir, simil_func="",
                   start=0, end=0, step=0):
    """
    Generate a range of graphs with thresholds between
    start and end.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    step_precision = len(str(step).split(".")[1])

    for x in np.arange(start=start, stop=end, step=step):
        self.generate(simil_func=simil_func, epsilon=x)
        output_name = "{}_{:.{}}".format(simil_func, x, step_precision)
        output_path = os.path.join(output_dir, output_name)
        self.to_pickle(output_path)
