# Global import
import numpy as np

# Local import
from deyep.core.tools import linear_algebra  as la
from deyep.utils.names import KVName


def fnt(sax_D, sax_I, sax_sn, sax_si):
    sax_sn_ = la.matrix_product(sax_si, sax_I)

    sax_sn_ += la.matrix_product(sax_sn, sax_D)

    return sax_sn_

def fnt_better_parallel()
    # zip srn with each columns of D, then srn.dot(diagonal(D)).data gives you a list of coef that you can store as keys
    # in processing step


