

def fnt(sax_D, sax_I, sax_sn, sax_si):
    res = sax_sn.dot(sax_D) + sax_si.dot(sax_I)
    return res


def fot(sax_O, sax_sn):
    res = sax_sn.dot(sax_O)
    return res


def bnt(sax_D, sax_O, sax_snb, sax_sob, sax_activation):
    sax_snb_ = sax_sob.dot(sax_O.transpose().multiply(sax_activation))
    sax_snb_ += sax_snb.dot(sax_D.transpose())
    return sax_snb_


def bit(sax_I, sax_snb):
    sax_sib = sax_snb.dot(sax_I.transpose())
    return sax_sib


def bcv(sax_got, sax_sob, sax_Cb):
    for i in range(sax_got.shape[1]):
        sax_Cb[:, i] = sax_Cb[:, i] * (sax_got[0, i] - sax_sob[0, i]) > 0

    return sax_Cb


def bcu(sax_Cb, dn, w0=1):
    dn.graph['Cm'] += sax_Cb
    t_nz_c = set(zip(*sax_Cb.nonzero()))
    t_nz_o = set(zip(*dn.O.nonzero()))
    dn.graph['Ow'] += sax_Cb * w0