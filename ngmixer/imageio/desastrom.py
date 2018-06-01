'''
new astrometry measurements from Gary Bernstein
'''
from __future__ import print_function

import os
import numpy as np
import fitsio
import ngmix

CCD_NAMES = [
    None,'S29','S30','S31','S25','S26','S27','S28','S20','S21','S22',
    'S23','S24','S14','S15','S16','S17','S18','S19','S8','S9',
    'S10','S11','S12','S13','S1','S2','S3','S4','S5','S6','S7',
    'N1','N2','N3','N4','N5','N6','N7','N8','N9','N10','N11',
    'N12','N13','N14','N15','N16','N17','N18','N19','N20','N21',
    'N22','N23','N24','N25','N26','N27','N28','N29','N30','N31',
]

class AstromReader(dict):
    """
    class to read astrometry wcs from the new astrom files
    """
    def __init__(self, conf):
        self.update(conf)
        self._load_zone_map()

    def get_wcs(self, expnum, ccdnum):
        """
        get the astrometry object for the given identifiers
        """
        import pixmappy
        import galsim
        ccdname = CCD_NAMES[ccdnum]

        zm = self._zone_map

        w,=np.where(
            (zm['expnum']==expnum)
            &
            (zm['detpos']==ccdname)
        )
        if w[0].size != 1:
            mess="%s %s not found in astrometry zone map, returning None"
            mess =mess % (expnum, ccdnum)
            print(mess)
            wcs = None
        else:
            zone = zm['zone'][w[0]]
            fname = self._get_astrom_file(zone)

            print("loading %d,%d from %s" % (expnum,ccdnum,fname))

            wcs = pixmappy.GalSimWCS(fname, exp=expnum, ccdnum=ccdnum)
            wcs._color = 0.  # For now.  Revisit when doing color-dependent PSF.

            # there seems to be some coordinate convention problem here
            wcs = wcs.withOrigin(galsim.PositionD(0.5,0.5))

        return GalsimWCSWrapper(wcs)

    def _get_astrom_dir(self):
        dir = os.environ['ASTROM_DIR']
        version=self['version']
        if version is not None:
            dir = os.path.join(dir, version)

        return dir

    def _get_astrom_file(self, zone):
        dir=self._get_astrom_dir()
        return os.path.join(dir, 'zone%03d.astro' % zone)

    def _get_zone_map_file(self):
        dir=self._get_astrom_dir()
        return os.path.join(dir, 'which_zone.fits')

    def _load_zone_map(self):
        fname = self._get_zone_map_file()
        print('    loading exposure-> zone mapping:',fname)
        self._zone_map = fitsio.read(fname)

class GalsimWCSWrapper(object):
    """
    wrapper for the galsim wcs to extract an ngmix
    jacobian
    """
    def __init__(self, wcs):
        self.wcs=wcs

    def get_jacobian(self,
                     ra,
                     dec,
                     cutout_row,
                     cutout_col,
                    ):
        """
        get a jacobian with new elements as well as any
        required shift
        """
        gs_jac, _, _ = self._get_gs_jacobian_and_rowcol(ra, dec)

        return ngmix.Jacobian(
            row=cutout_row,
            col=cutout_col,

            dudrow = gs_jac.dudy,
            dudcol = gs_jac.dudx,

            dvdrow = gs_jac.dvdy,
            dvdcol = gs_jac.dvdx,
        )

    def get_shifted_jacobian(self,
                             ra,
                             dec,
                             orig_row,
                             orig_col,
                             orig_cutout_row,
                             orig_cutout_col,
                            ):
        """
        get a jacobian with new elements as well as any
        required shift

        This doesn't work because of the absolute shift between
        the original coadd RA,DEC and the truth
        """
        gs_jac, row, col = self._get_gs_jacobian_and_rowcol(ra, dec)

        row_shift = row - orig_row
        col_shift = col - orig_col

        print("    position shift:",row_shift, col_shift)

        cutout_row = orig_cutout_row + row_shift
        cutout_col = orig_cutout_col + col_shift

        return ngmix.Jacobian(
            row=cutout_row,
            col=cutout_col,

            dudrow = gs_jac.dudy,
            dudcol = gs_jac.dudx,

            dvdrow = gs_jac.dvdy,
            dvdcol = gs_jac.dvdx,
        )

    def _get_gs_jacobian_and_rowcol(self, ra, dec):
        """
        get the jacobian and image position (not the postage
        cutout position, which is what we will want). Center of
        jacobian is set to zero, can be updated later
        """
        import galsim
        
        gs_ra = ra*galsim.degrees
        gs_dec = dec*galsim.degrees
        wpos = galsim.CelestialCoord(gs_ra, gs_dec)

        pos = self.wcs.toImage(wpos)
        row, col = pos.y-1, pos.x-1

        # galsim jacobian
        gs_jac = self.wcs.jacobian(image_pos = pos)

        return gs_jac, row, col



