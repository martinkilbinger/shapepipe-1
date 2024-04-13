import re
import galsim
import numpy as np
from astropy.io import fits
from modopt.math.stats import sigma_mad
from sqlitedict import SqliteDict
from shapepipe.pipeline import file_io

class Tile_cat():
    """Tile cat.
    SExtractor detection catalog from the tile. 

    Parameters
    ----------
    cat_path: str
        path to detection catalog

    """
 
def __init__(self, cat_path):

    self.cat_path = cat_path
    
@classmethod
def get_data(self):
    cat = file_io.FITSCatalogue(
            self.cat_path,
            SEx_catalogue=True,
        )
    cat.open()
     
    self.vign = np.copy(cat.get_data()['VIGNET'])

    self.obj_id = np.copy(cat.get_data()['NUMBER'])
    self.ra = np.copy(cat.get_data()['XWIN_WORLD'])
    self.dec = np.copy(cat.get_data()['YWIN_WORLD'])
    self.flux = np.copy(cat.get_data()['FLUX_AUTO'])
    self.size = np.copy(cat.get_data()['FWHM_WORLD'])
    #self.e = np.copy(cat.get_data()['ELLIPTICITY'])
    #self.theta = np.copy(cat.get_data()['THETA_WIN_WORLD'])

    cat.close()

class Vignet():
    """Vignet.

    Class to hold SqliteDicts of vignettes
    These will be compiled into postage stamps.

    Parameters
    ----------
    gal_vignet_path: str
    bkg_vignet_path: str
    psf_vignet_path: str
    weight_vignet_path: str
    flag_vignet_path: str
    f_wcs_path: str
    """
    def __init__(
        self,
        gal_vignet_path,
        bkg_vignet_path,
        psf_vignet_path,
        weight_vignet_path,
        flag_vignet_path,
        f_wcs_path
    ):

        self.f_wcs_file = SqliteDict(f_wcs_path)
        self.gal_cat = SqliteDict(gal_vignet_path)
        self.bkg_cat = SqliteDict(bkg_vignet_path)
        self.psf_cat = SqliteDict(psf_vignet_path)
        self.weight_cat = SqliteDict(weight_vignet_path)
        self.flag_cat = SqliteDict(flag_vignet_path)

    @classmethod
    def close(self):
        self.f_wcs_file.close()
        self.gal_cat.close()
        self.bkg_cat.close()
        self.flag_cat.close()
        self.weight_cat.close()
        self.psf_cat.close()

class Postage_stamp():
    """Galaxy Postage Stamp.

    Class to hold postage stamps cutouts for a single galaxy

    Parameters
    ----------
    bkg_sub: bool, optional
        ``True`` for background subtraction
    megacam_flip: bool, optional
        ``True`` to flip megaprime coordinates with megapipe processing
    mask_frac: float
        maximum fraction of allowed masked pixels
    rescale_weights: bool
        ``True`` rescales weights into variance maps
    symmetrize_mask: bool
        ``True`` will symmetrize mask     
    """
    def __init__(
        self,
        rng,
        bkg_sub=True,
        megacam_flip=True,
        mask_frac=1/3.0,
        rescale_weights=True,
        symmetrize_mask=False

    ):
        
        self.rng=rng
        self.bkg_sub=bkg_sub
        self.megacam_flip=megacam_flip
        self.mask_frac=mask_frac
        self.rescale_weights=rescale_weights
        self.symmetrize_mask=symmetrize_mask

        self.gals = []
        self.psfs = []
        self.weights = []
        self.noise_ims = []
        self.flags = []
        self.jacobs = []

    def preprocess_postage_stamp(self, vignet, tile_cat, obj_id):
        """preprocess stamps.

        runs per object over all epochs.
        requested preprocessing steps can include:
            background subtraction
            rescaling flux of individual epochs
            flipping megacam vignets
            remove objects that are masked more than mask_frac
            converting a weight map into a variance map

        Parameters
        ----------
        vignet : Vignet
            Array containing the postage stamp to flip
        tile_cat : Tile_cat
            coadded tile detection catalog
        obj_id : int
            ID of object in tile detection catalog
        Returns
        -------
        postage_stamp
            processed postage stamp object
        """

        obj_id_det_cat = obj_id - 1
        obj_id = str(obj_id)

        # raise error if psf or galaxy is missing
        if (
            (vignet.psf_cat[obj_id] == 'empty')
            or (vignet.gal_cat[obj_id] == 'empty')
        ):
            raise AttributeError

        # define per-object lists of individual exposures to go into ngmix
        stamp = Postage_stamp()
    
        #identify exposure and ccd number from psf catalog 
        epoch_list = list(vignet.psf_cat[obj_id].keys())
    
        # process multi-epoch data
        for exp_ccd in epoch_list:
            exp_name, ccd_n_string = re.split('-', exp_ccd)
            ccd_n = int(ccd_n_string)

            gal = (
                vignet.gal_cat[obj_id][exp_ccd]['VIGNET']
            )
            # skip galaxies with zero image size (think about this)
            if len(np.where(gal.ravel() == 0)[0]) != 0:
                continue

            # background subtract
            if stamp.bkg_sub:
                bkg = (
                    vignet.bkg_cat[obj_id][exp_ccd]['VIGNET']
                )
                gal_sub_bkg = _background_subtract(
                    gal,
                    bkg
                )
            else:
                gal_sub_bkg = gal

            # flip tile megacam images to mactch ccd if necessary
            if stamp.megacam_flip:
                tile_vign = (
                    _MegaCamFlip(np.copy(tile_vign[obj_id_det_cat]), ccd_n)
                )

            flag = (
                vignet.flag_vign_cat[obj_id][exp_ccd]['VIGNET']
            )
            flag[np.where(tile_vign == -1e30)] = 2**10

            # remove objects masked greater than mask_frac
            flag_tmp = flag.ravel()
            masked = len(np.where(flag_tmp != 0)[0])
            if  masked / (gal.shape[0] * gal.shape[1]) > self.mask_frac:
                continue

            # prepare weight map
            weight_raw = (
                vignet.weight_cat[obj_id][exp_ccd]['VIGNET']
            )

            gal_masked, weight, noise_img = self._prepare_ngmix_weights(
                gal_sub_bkg,
                weight_raw,
                flag
            )
   
            jacob = _get_galsim_jacobian(
                vignet.f_wcs_file[exp_name][ccd_n]['WCS'],
                tile_cat.ra[obj_id_det_cat],
                tile_cat.dec[obj_id_det_cat]
            )

            # rescale by relative zero-points
            header = fits.Header.fromstring(
                vignet.f_wcs_file[exp_name][ccd_n]['header']
            )

            gal_scaled, weight_scaled = _rescale_epoch_fluxes(
                gal_masked,
                weight,
                header
                )

            # gather postage stamps in all of the epochs
            self.gals.append(gal_scaled)
            self.weights.append(weight_scaled)
            self.flags.append(flag)
            self.jacobs.append(jacob)
            self.noise_ims.append(noise_img)
            self.psfs.append(
                vignet.psf_cat[obj_id][exp_ccd]['VIGNET']
            )

    def _prepare_ngmix_weights(self, 
        gal,    
        weight,
        flag,
        bkground=None,
        exp_ccd=None
    ):
        """Prepares weights for shape measurement.
        bookkeeping for ngmix weights. runs on a single galaxy and epoch

        Parameters
        ----------
        gal : numpy.ndarray
            galaxy images. 
        weight : numpy.ndarray
            weight image 
        flag : numpy.ndarray
            flag image.    
        bkground : numpy.array, optional
            background image.
        exp_ccd: str, optional
            exposure-ccd combination, not currently implemented
        Returns
        -------
        numpy.ndarray
            galaxy image where noise replaces masked regions
        numpy.ndarray
            variance map for NGMIX
        numpy.ndarray
            noise image    

        """ 

        # integrate flag info into weights
        weight_map = np.copy(weight)
        if self.symmetrize_mask:
            flag = _symmetrize_mask(flag)
        
        weight_map[np.where(flag != 0)] = 0.

        if self.bkg_sub == True: 
            sig_noise = _get_noise(bkground)    
        else:
            sig_noise = _sextractor_sky_background_dev(exp_ccd) 
        
        # create gaussian noise image for correlated noise correction
        noise_img = self.rng.standard_normal(weight.shape) * sig_noise
    
        # fill in galaxy image masked regions with noise
        noise_img_gal = self.rng.standard_normal(weight.shape) * sig_noise
        gal_masked = np.copy(gal)
        if (len(np.where(weight_map == 0)[0]) != 0):
            gal_masked[weight_map == 0] = noise_img_gal[weight_map == 0]

        # convert weight map to variance map
        if self.rescale_weights == True:
            weight_map *= 1 / sig_noise ** 2
    
        return gal_masked, weight_map, noise_img

def _background_subtract(gal,bkg):
    """background subtraction.
        
    Parameters
    ----------
    gal : numpy.ndarray
        galaxy image
    bkg : numpy.ndarray
        background
        
    Returns
    -------
    numpy.ndarray
        background subtracted galaxy
    """

    # background subtraction
    gal_vign_sub_bkg = gal - bkg

    return gal_vign_sub_bkg

def _MegaCamFlip(vign, ccd_nb):
    """Flip for MegaCam.

    MegaPipe has CCDs that are upside down. This function flips the
    postage stamps in these CCDs. TO DO: This will give incorrect results
    when used with THELI ccds.  Fix this.

    Parameters
    ----------
    vign : numpy.ndarray
        Array containing the postage stamp to flip
    ccd_nb : int
        ID of the CCD containing the postage stamp

    Returns
    -------
    numpy.ndarray
        The flipped postage stamp

    """   
    if ccd_nb < 18 or ccd_nb in [36, 37]:
        # swap x axis so origin is on top-right
        return np.rot90(vign, k=2)
        print('rotating megapipe image')
    else:
        # swap y axis so origin is on bottom-left
        return vign

def _rescale_epoch_fluxes(gal,weight,header):
    """rescale epoch fluxes.
    rescale epochs by relative zeropoints to be on the same flux scale
        
    Parameters
    ----------
    gal : numpy.ndarray
        background subtracted galaxy image
    weight : numpy.ndarray
        weight image
    header : 
        image header
        
    Returns
    -------
    numpy.ndarray
        rescaled galaxy image
    numpy.ndarray
        rescaled weight image
    """
    Fscale = header['FSCALE']

    gal_scaled = gal * Fscale
    weight_scaled = weight * 1 / Fscale ** 2

    return gal_scaled, weight_scaled

def _get_galsim_jacobian(wcs, ra, dec):
    """Get galsim jacobian.
    This produces a galsim jacobian at a point.  We call it local_wcs because we convert to a ngmix object to create the jacobian later.
    
    Parameters
    ----------
    wcs : astropy.wcs.WCS
        WCS object for which we want the Jacobian
    ra : float
        RA position of the center of the vignet (in degrees)
    dec : float
        Dec position of the center of the vignet (in degress)

    Returns
    -------
    galsim.wcs.BaseWCS.jacobian
        Jacobian of the WCS at the given position

    """
    g_wcs = galsim.fitswcs.AstropyWCS(wcs=wcs)
    world_pos = galsim.CelestialCoord(
        ra=ra * galsim.angle.degrees,
        dec=dec * galsim.angle.degrees,
    )
    galsim_jacob = g_wcs.jacobian(world_pos=world_pos)

    return galsim_jacob

def _get_noise(bkg_array):
    """Get Noise.
    Computes median deviation of sky background from object postage stamp.

    Parameters
    ----------
    bkg_array : numpy.ndarray
        Sextractor sky background postage stamp
    
    Returns
    -------
    float
        Sigma of the noise on the galaxy image
    """
    sig_noise = sigma_mad(bkg_array)

    return sig_noise

def _symmetrize_mask(mask):
    """Symmetrize masks.
    Symmetrizes pixels in mask image adding 90 deg rotated mask
    note: don't symmetrize the ubserseg. This effect is for bad columns, edges, bleeds

    Parameters
    ----------
    mask : numpy.ndarray
        weight image

    Returns
    -------
    np.ndarray
        symmetrized mask image

    """
    assert mask.shape[0] == mask.shape[1]

    mask_rot=np.rot90(mask)
    mask_zero = np.where(mask_rot == 0.0)

    if mask_zero[0].size > 0:
        mask[mask_zero] = 0.0
    
    return mask

def _sextractor_sky_background_dev(exp_ccd):
    #gets backdev from sextractor catalog
    raise NotImplementedError("Sextractor backdev not implemented")

