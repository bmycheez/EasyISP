# File: lsc.py
# Description: Lens Shading Correction
# Created: 2023/11/14 15:50
# Author: sangmin-enerzai (sangmin.kim@enerzai.com)


import numpy as np
import math

from .basic_module import BasicModule


class LSC(BasicModule):
    def execute(self, data):
        bayer = data['bayer'].astype(np.uint32)
        h, w = bayer.shape

        minR = 0
        maxR = math.sqrt((h/2)**2+(w/2)**2)

        for y in range(h):
            for x in range(w):
                r = math.sqrt((y - h/2)**2+(x - w/2)**2)
                factor = (r - minR) / (maxR - minR)
                bayer[y, x] = bayer[y, x] * \
                    (1 + self.params.intensity * (factor + 0.5))

        data['bayer'] = bayer.astype(np.uint16)
