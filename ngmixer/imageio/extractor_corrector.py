from __future__ import print_function
import numpy
import os
import time
import fitsio
import meds

from ..render_ngmix_nbrs import RenderNGmixNbrs

#
# should test against the corrector script
#

class MEDSExtractorCorrector(meds.MEDSExtractor):
    def __init__(self,
                 mof_file,
                 meds_file,
                 start,
                 end,
                 sub_file,
                 replace_bad=True,
                 min_weight=-9999.0,
                 # these are the bands in the mof
                 band_names = ['g','r','i','z'],
                 model='cm',
                 cleanup=False,
                 verbose=False):

        self.mof_file=mof_file

        self.band_names=band_names
        self.model=model

        self.replace_bad=replace_bad
        self.min_weight=min_weight
        self.verbose=verbose

        self._set_band(meds_file)

        # this will run extraction
        super(MEDSExtractorCorrector,self).__init__(
            meds_file,
            start,
            end,
            sub_file,
            cleanup=cleanup,
        )

    def _extract(self):
        # first do the ordinary extraction
        super(MEDSExtractorCorrector,self)._extract()

        # self.sub_file should now have been closed
        time.sleep(2)
        self._set_renderer()

        with fitsio.FITS(self.sub_file,mode='rw') as fits:
            mfile = RefMeds(fits)
            cat = mfile.get_cat()

            nobj=cat.size

            # loop through objects and correct each
            for mindex in xrange(cat['id'].size):

                coadd_object_id=cat['id'][mindex]
                ncutout=cat['ncutout'][mindex]
                box_size=cat['box_size'][mindex]
                start_row=cat['start_row'][mindex]

                # print even if not verbose
                print("%d/%d  %d" % (mindex+1, nobj, coadd_object_id))
                if ncutout > 1 and box_size > 0:
                    for cutout_index in xrange(1,ncutout):

                        # get seg map
                        try:
                            seg = mfile.interpolate_coadd_seg(mindex, cutout_index)
                        except:
                            seg = mfile.get_cutout(mindex, cutout_index, type='seg')

                        if self.verbose:
                            print('    doing nbrs for object at '
                                  'index %d, cutout %d' % (mindex,cutout_index))

                        res = self.renderer.render_nbrs(
                            coadd_object_id,
                            cutout_index,
                            seg,
                            self.model,
                            self.band,
                            total=True,
                            verbose=self.verbose,
                        )
                        if res is None:
                            if self.verbose:
                                print('        no nbrs')
                            continue

                        if self.verbose:
                            print('        found nbrs - correcting images and weight maps')

                        cen_img, nbrs_img, nbrs_mask, nbrs_ids, pixel_scale = res

                        img   = mfile.get_cutout(mindex, cutout_index, type='image')
                        wgt   = mfile.get_cutout(mindex, cutout_index, type='weight')
                        bmask = mfile.get_cutout(mindex, cutout_index, type='bmask')

                        # subtract neighbors
                        img -= nbrs_img*pixel_scale*pixel_scale
                        # possibly zero the weight
                        wgt *= nbrs_mask

                        # set masked or zero weight pixels to the value of the central.
                        # For codes that use the weight, such as max like, this makes
                        # no difference, but it may be important for codes that 
                        # take moments or use FFTs
                        if self.replace_bad:
                            wbad=numpy.where( (bmask != 0) | (wgt < self.min_weight) )
                            if wbad[0].size > 0:
                                print("            setting",wbad[0].size,
                                      "bad bmask/wt pixels to central model")
                                img[wbad] = cen_img[wbad]

                        # now overwrite pixels on disk
                        fits['image_cutouts'].write(img.ravel(), start=[start_row[cutout_index]])
                        fits['weight_cutouts'].write(wgt.ravel(), start=[start_row[cutout_index]])

                        # note in the bad pixel mask where we have masked a neighbor
                        # that failed to converge
                        w=numpy.where(nbrs_mask != 1)
                        if w[0].size > 0:
                            print("            modifying",w[0].size,"bmask pixels")
                            bmask[w] = 2**31-1
                            fits['bmask_cutouts'].write(bmask.ravel(), start=[start_row[cutout_index]])
                else:
                    # we always want to see this
                    print("    not writing ncutout:",ncutout,"box_size:",box_size)

    def _load_ngmix_data(self):
        self.fit_data = fitsio.read(self.mof_file)
        self.nbrs_data = fitsio.read(self.mof_file,ext='nbrs_data')

    def _set_renderer(self):
        self._load_ngmix_data()

        # build the renderer, set options
        conf = {'unmodeled_nbrs_masking_type':'nbrs-seg'}
        self.renderer = RenderNGmixNbrs(
            self.fit_data,
            self.nbrs_data,
            **conf
        )

    def _set_band(self, meds_file):
        # get the band for the file
        band = -1
        for band_name in self.band_names:
            btest = '-%s-' % band_name
            if btest in meds_file:
                band = self.band_names.index(band_name)
                break
        if band == -1:
            raise ValueError("Could not find band for file '%s'!" % corr_file)

        self.band = band


class RefMeds(meds.MEDS):
    """
    meds file object that accepts a ref to an open fitsio FITS object

    does not close the file if used in python context
    """
    def __init__(self,fits,filename=None):
        self._filename = filename
        self._fits = fits
        self._cat = self._fits["object_data"][:]
        self._image_info = self._fits["image_info"][:]
        self._meta = self._fits["metadata"][:]

    def close(self):
        pass


