# Goal: Transport to Inj dump screen.  Do dispersion scan at 130MeV
# Goal: Transport to BC2 dump screen.  Do dispersion scan at 130MeV
#                                      Do energy scan from 2.4GeV to 700MeV


from .sections import sections
from .sections.i1 import make_twiss0

def i1d_optics(dscan_conf):
    i1dlat = sections.make_to_i1d_dump_screen_lattice()



def run_i1d_dispersion_scan(dscan_conf):
    i1dlat = sections.make_to_i1d_dump_screen_lattice()
    from IPython import embed; embed()


def run_i1d_tds_scan():
    pass
