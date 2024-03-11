import re
import galsim
import numpy as np
from astropy.io import fits
from modopt.math.stats import sigma_mad
from sqlitedict import SqliteDict

class Tile_cat():
    """Postage stamps.
    we have two main types of info- sextractor info and the vignet info. we want to store the sextractor stuff in an object so it is easy to access
    I think we can ignore the tile cat stuff if we expand the fits reader, then it is an object we want to pass to the postage stamp maker

    this creates postage stamps of objects to be processed along with requested preprocessing steps:
    background subtraction
    rescaling individual epochs
    flipping megacam vignets if necessary
    remove objects that are more than 1/3 masked
    converting a weight map into a variance map by looking at the noise

    Parameters
    ----------
    paths to data, processing steps that we want
    array of postage stamps, whih are themselves arrays of epochs

    """
    def __init__(
        self, 
        cat_path,
        bkg_sub,
        megacam_flip,
    ):

        self.cat_path = cat_path
        self.bkg_sub = bkg_sub
        self.megacam_flip = megacam_flip

        # sextractor detection catalog for the tile
        self.tile_vignet
        dtype = [('obj_id','i4'),('ra','>f8'),('dec','>f8'),('flux','>f4'),('VIGNET', '>f4', (51, 51))]
        #self.tile_data = np.recarray(())

    @classmethod
    def get_data(self, cat_path):
        tile_cat = file_io.FITSCatalogue(
            cat_path,
            SEx_catalogue=True,
        )
        tile_cat.open()
        # I would like to make this into an object cat
        self.vign = np.copy(tile_cat.get_data()['VIGNET'])

        self.obj_id = np.copy(tile_cat.get_data()['NUMBER'])
        self.ra = np.copy(tile_cat.get_data()['XWIN_WORLD'])
        self.dec = np.copy(tile_cat.get_data()['YWIN_WORLD'])
        self.flux = np.copy(tile_cat.get_data()['FLUX_AUTO'])
        self.size = np.copy(tile_cat.get_data()['FWHM_WORLD'])
        self.e = np.copy(tile_cat.get_data()['ELLIPTICITY'])
        self.theta = np.copy(tile_cat.get_data()['THETA_WIN_WORLD'])

        tile_cat.close()



# we want this to inherit properties of catalog
class Postage_stamp():
    """Galaxy Postage Stamp.

    Class to hold catalog of postage stamps for a single galaxy

    Parameters
    ----------
    bkg_sub: bool

    megacam_flip: bool
    We probably want to put weight and flag options here too

    """
    def __init__(
        self,
        bkg_sub=True,
        megacam_flip=True

    ):
        self.gals = []
        self.psfs = []
        self.weights = []
        self.flags = []
        self.jacobs = []


class Vignet():
    """Vignet.

    Class to hold catalog of postage stamps

    Parameters
    ----------
    gal_vignet_path
    bkg_vignet_path
    psf_vignet_path
    weight_vignet_path
    flag_vignet_path
    f_wcs_path
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
        self.gal_vign_cat = SqliteDict(gal_vignet_path)
        self.bkg_vign_cat = SqliteDict(bkg_vignet_path)
        self.psf_vign_cat = SqliteDict(psf_vignet_path)
        self.weight_vign_cat = SqliteDict(weight_vignet_path)
        self.flag_vign_cat = SqliteDict(flag_vignet_path)

    @classmethod
    def close(self):
        self.f_wcs_file.close()
        self.gal_vign_cat.close()
        self.bkg_vign_cat.close()
        self.flag_vign_cat.close()
        self.weight_vign_cat.close()
        self.psf_vign_cat.close()

    def MegaCamFlip(self, vign, ccd_nb):
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
        
def preprocess_postage_stamps(vignet, tile_cat):
    i_tile = tile_cat.obj_id - 1
    obj_id = tile_cat.obj_id
    # define per-object lists of individual exposures to go into ngmix
    stamp = Postage_stamp()
    if (
        (vignet.psf_vign_cat[str(obj_id)] == 'empty')
        or (vignet.gal_vign_cat[str(obj_id)] == 'empty')
    ):
        raise AttributeError
    #identify exposure and ccd number from psf catalog
    psf_expccd_names = list(vignet.psf_vign_cat[str(obj_id)].keys())
    for expccd_name in psf_expccd_names:
        exp_name, ccd_n = re.split('-', expccd_name)

        gal_vign = (
            vignet.gal_vign_cat[str(obj_id)][expccd_name]['VIGNET']
        )

        if len(np.where(gal_vign.ravel() == 0)[0]) != 0:
            continue
        
        if stamp.bkg_sub:
            bkg_vign = (
                vignet.bkg_vign_cat[str(obj_id)][expccd_name]['VIGNET']
            )
            gal_vign_sub_bkg = background_subtract(
                gal_vign,
                bkg_vign
            )
        else:
            gal_vign_sub_bkg = gal_vign

        if stamp.megacam_flip:
            tile_vign = (
                Ngmix.MegaCamFlip(np.copy(tile_vign[i_tile]), int(ccd_n))
            )

        flag_vign = (
            vignet.flag_vign_cat[str(obj_id)][expccd_name]['VIGNET']
        )
        flag_vign[np.where(tile_vign == -1e30)] = 2**10
        v_flag_tmp = flag_vign.ravel()
        # remove objects that are more than 1/3 masked
        if len(np.where(v_flag_tmp != 0)[0]) / (51 * 51) > 1 / 3.0:
            continue

        weight_vign = (
            vignet.weight_vign_cat[str(obj_id)][expccd_name]['VIGNET']
        )

            ################This should be done when we make the stamps
        # prepare weight map
        gal_masked, weight_map, noise_img = prepare_ngmix_weights(
            gal,
            weight,
            flag
        )
        # WHY RECENTER???
   
        jacob = get_galsim_jacobian(
            vignet.f_wcs_file[exp_name][int(ccd_n)]['WCS'],
            tile_cat.ra[i_tile],
            tile_cat.dec[i_tile]
        )

        header = fits.Header.fromstring(
            vignet.f_wcs_file[exp_name][int(ccd_n)]['header']
        )

        # rescale by relative zero-points
        gal_vign_scaled, weight_vign_scaled = rescale_epoch_fluxes(
            gal_vign_sub_bkg,
            weight_vign,
            header
            )

        # gather postage stamps in all of the epochs
        stamp.gal_vign_list.append(gal_vign_scaled)
        stamp.psf_vign_list.append(
            vignet.psf_vign_cat[str(obj_id)][expccd_name]['VIGNET']
        )

        stamp.weight_vign_list.append(weight_vign_scaled)
        stamp.flag_vign_list.append(flag_vign)
        stamp.jacob_list.append(jacob)
                
    return stamp

def background_subtract(gal,bkg):
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

def rescale_epoch_fluxes(gal,weight,header):
    """rescale epochs by relative zeropoints to be on the same flux scale
        
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

def get_galsim_jacobian(wcs, ra, dec):
    """Get local wcs.
    This produces a galsim jacobian at a point.  We call it local_wcs because we convert to a ngmix object to create the jacobian later.
    TO DO: can we do this within ngmix?

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
        Jacobian of the WCS at the required position

    """
    g_wcs = galsim.fitswcs.AstropyWCS(wcs=wcs)
    world_pos = galsim.CelestialCoord(
        ra=ra * galsim.angle.degrees,
        dec=dec * galsim.angle.degrees,
    )
    galsim_jacob = g_wcs.jacobian(world_pos=world_pos)

    return galsim_jacob

def get_noise(background_vignet):
    """Get Noise.
    Computes sigma of sky background from object postage stamp.

    Parameters
    ----------
    background_vignet : numpy.ndarray
        Sextractor sky background postage stamp
    
    Returns
    -------
    float
        Sigma of the noise on the galaxy image
    """
    sig_noise = sigma_mad(background_vignet)

    return sig_noise

def prepare_ngmix_weights(
    gal,
    weight,
    flag,
    weight_type='megapipe',
    symmetrize_mask=False
):
    """bookkeeping for ngmix weights. runs on a single galaxy and epoch
        pixel scale and galaxy guess
        TO DO: decide if we want galaxy guess stuff

    Parameters
    ----------
    gal : numpy.ndarray
        galaxy image.  List indices run over epochs
    weight : numpy.ndarray
        weight image  List indices run over epochs
    flag : numpy.ndarray
        flag image.  List indices run over epochs   
    weight_type : str, optional
        'THELI' or 'megapipe' (default)
    symmetrize_mask : bool, optional
        'True' if mask will be symmetrized; default is ``False``

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
    if symmetrize_mask:
        #### add ask symmetrization here
        pass
    weight_map[np.where(flag != 0)] = 0.

    if weight_type == 'THELI':
        sig_noise = sextractor_sky_background_dev(filename)
    elif weight_type == 'megapipe':
        sig_noise = sigma_mad(gal)
        # note: we need to mask galaxy image here
    #    sig_noise = get_noise(
    #        gal,
    #        weight,
    #        gal_guess_tmp,
    #        pixel_scale,
    #    )
    else:
        raise TypeError(f'weight_type must be THELI or megapipe')

    # create gaussian noise image for correlated noise correction
    noise_img = np.random.randn(*gal.shape) * sig_noise
    noise_img_gal = np.random.randn(*gal.shape) * sig_noise
    
    # fill in galaxy image masked regions with noise
    gal_masked = np.copy(gal)
    if (len(np.where(weight_map == 0)[0]) != 0):
        gal_masked[weight_map == 0] = noise_img_gal[weight_map == 0]

    # convert weight map to variance map
    weight_map *= 1 / sig_noise ** 2
    
    return gal_masked, weight_map, noise_img

def sextractor_sky_background_dev(filename):

    return backdev

# Define the SExtractor parameters for a galaxy
def sextractor_e1e2(e,theta):
    """sextractor_e1e2

    computes ellipticity from sextrator quantities
    Parameters
    ----------
    stamp : Postage_stamp
        List of the galaxy vignets.  List indices run over epochs
    prior : ngmix.priors
        Priors for the fitting parameters
    flux_guess : np.ndarray
        guess for flux
    pixel_scale : float
        pixel scale in arcsec
    rng : numpy.random.RandomState
        Random state for guesses and priors    

    Returns
    -------
    np.ndarray
        ellipticity

    """
    # Convert the position angle from degrees to radians
    phi = np.radians(theta) - np.pi/2
    # Calculate the ellipticity vector
    e_vec = e * np.array([np.cos(2*phi), np.sin(2*phi)])
    return e_vec

