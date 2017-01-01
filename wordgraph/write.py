import os

import numpy as np

def generate_range(self, output_dir, simil_func="",
                   start=0, end=0, step=0, unit=False):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for x in np.arange(start=start, stop=end, step=step):
        print(x)
        self.generate(simil_func=simil_func, epsilon=x, unit=unit)
        output_name = "{}_{}".format(simil_func, x)
        output_path = os.path.join(output_dir, output_name)
        self.to_pickle(output_path)