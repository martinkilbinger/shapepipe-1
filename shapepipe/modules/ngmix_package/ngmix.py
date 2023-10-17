"""NGMIX.

This module contains a class for ngmix shape measurement.

:Authors: Lucie Baumont, Axel Guinot

"""

import re
import ngmix
import galsim
import numpy as np
from astropy.io import fits
from modopt.math.stats import sigma_mad
from ngmix.observation import Observation, ObsList
from sqlitedict import SqliteDict

from shapepipe.pipeline import file_io

class Tile_cat(object):
 """Tile_cat.

    catalog measured on a tile

    Parameters
    ----------
    cat_path

    """
 def __init__(self, cat_path):
        self.cat_path = cat_path
      # sextractor detection catalog for the tile
        tile_cat = file_io.FITSCatalogue(
            self.cat_path,
            SEx_catalogue=True,
        )
        tile_cat.open()
        # I would like to make this into an object cat
        self.obj_id = np.copy(tile_cat.get_data()['NUMBER'])
        self.vign = np.copy(tile_cat.get_data()['VIGNET'])
        self.ra = np.copy(tile_cat.get_data()['XWIN_WORLD'])
        self.dec = np.copy(tile_cat.get_data()['YWIN_WORLD'])
        self.flux = np.copy(tile_cat.get_data()['FLUX_AUTO'])
        tile_cat.close()

class Postage_stamp(object):
    """Galaxy Postage Stamp.

    Class to hold catalog of postage stamps for a single galaxy

    Parameters
    ----------
    bkg_subtraction: bool

    megacam_flip: bool

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
        self.bkg_sub = bkg_sub
        self.megacam_flip = megacam_flip

class Vignet(object):
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

class Ngmix(object):
    """Ngmix.

    Class to handle NGMIX shapepe measurement.

    Parameters
    ----------
    input_file_list : list
        Input files
    output_dir : str
        Output directory
    file_number_string : str
        File numbering scheme
    zero_point : float
        Photometric zero point
    pixel_scale : float
        Pixel scale in arcsec
    f_wcs_path : str
        Path to merged single-exposure single-HDU headers
    w_log : logging.Logger
        Logging instance
    id_obj_min : int, optional
        First galaxy ID to process, not used if the value is set to ``-1``;
        the default is ``-1``
    id_obj_max : int, optional
        Last galaxy ID to process, not used if the value is set to ``-1``;
        the default is ``-1``

    Raises
    ------
    IndexError
        If the length of the input file list is incorrect

    """

    def __init__(
        self,
        input_file_list,
        output_dir,
        file_number_string,
        zero_point,
        pixel_scale,
        f_wcs_path,
        w_log,
        id_obj_min=-1,
        id_obj_max=-1
    ):

        if len(input_file_list) != 6:
            raise IndexError(
                f'Input file list has length {len(input_file_list)},'
                + ' required is 6'
            )

        self._tile_cat_path = input_file_list[0]

        self._gal_vignet_path = input_file_list[1]
        self._bkg_vignet_path = input_file_list[2]
        self._psf_vignet_path = input_file_list[3]
        self._weight_vignet_path = input_file_list[4]
        self._flag_vignet_path = input_file_list[5]

        self._output_dir = output_dir
        self._file_number_string = file_number_string

        self._zero_point = zero_point
        self._pixel_scale = pixel_scale

        self._f_wcs_path = f_wcs_path
        self._id_obj_min = id_obj_min
        self._id_obj_max = id_obj_max

        self._w_log = w_log

        # Initiatlise random generator
        seed = int(''.join(re.findall(r'\d+', self._file_number_string)))
        self._rng = np.random.RandomState(seed)
        self._w_log.info(f'Random generator initialisation seed = {seed}')

    @classmethod
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

    def get_prior(self, T_range=None, F_range=None):
        """Get Prior.

        get a prior for use with the maximum likelihood fitter

        Parameters
        ----------
        T_range: (float, float), optional
            The range for the prior on T
        F_range: (float, float), optional
            Fhe range for the prior on flux 

        Returns
        -------
        ngmix.priors
            Priors for the different parameters (ellipticity,center, size, flux)
        """
        # 2-d Gaussian prior on the object center
        # centered with respect to jacobian center
        # Units same as jacobian, probably arcsec
        cen_prior = ngmix.priors.CenPrior(
            cen1=0.0, 
            cen2=0.0, 
            sigma1=self.scale, 
            sigma2=self.scale, 
            rng=self._rng
        )
        
        # Prior on ellipticity. Details do not matter, as long
        # as it regularizes the fit. From Bernstein & Armstrong 2014
        g_sigma = 0.4
        g_prior = ngmix.priors.GPriorBA(sigma=g_sigma,rng=self._rng)

        if T_range is None:
            T_range = [-1.0, 1.e3]
        if F_range is None:
            F_range = [-100.0, 1.e9]

        # Flat Size prior in arcsec squared. Instead of flat, TwoSidedErf could be used
        T_prior = ngmix.priors.FlatPrior(
            minval=T_range[0], 
            maxval=T_range[1], 
            rng=self._rng
        )

        # Flat Flux prior. Bounds need to make sense for
        # images in question
        F_prior = ngmix.priors.FlatPrior(
            minval=F_range[0], 
            maxval=F_range[1],
            rng=self._rng
        )

        # Joint prior, combine all individual priors
        prior = ngmix.joint_prior.PriorSimpleSep(
            cen_prior=cen_prior,
            g_prior=g_prior,
            T_prior=T_prior,
            F_prior=F_prior,
        )

        return prior

    def compile_results(self, results):
        """Compile Results.

        Prepare the results of NGMIX before saving. TO DO: add snr_r and T_r
        This needs to be updated
        Parameters
        ----------
        results : dict
            Results of NGMIX metacal

        Returns
        -------
        dict
            Compiled results ready to be written to a file
            note: psfo is the original image psf from psfex or mccd

        Raises
        ------
        KeyError
            If SNR key not found

        """
        names = ['1m', '1p', '2m', '2p', 'noshear']
        names2 = [
            'id',
            'n_epoch_model',
            'moments_fail',
            'ntry_fit',
            'g1_psfo_ngmix',
            'g2_psfo_ngmix',
            'r50_psfo_ngmix',
            'g1_err_psfo_ngmix',
            'g2_err_psfo_ngmix',
            'r50_err_psfo_ngmix',
            'g1',
            'g1_err',
            'g2',
            'g2_err',
            'r50',
            'r50_err',
            'r50psf',
            'g1_psf',
            'g2_psf',
            'flux',
            'flux_err',
            's2n',
            'mag',
            'mag_err',
            'flags',
            'mcal_flags'
        ]
        output_dict = {k: {kk: [] for kk in names2} for k in names}
        for idx in range(len(results)):
            for name in names:

                mag = (
                    -2.5 * np.log10(results[idx][name]['flux'])
                    + self._zero_point
                )
                mag_err = np.abs(
                    -2.5 * results[idx][name]['flux_err']
                    / (results[idx][name]['flux'] * np.log(10))
                )

                output_dict[name]['id'].append(results[idx]['obj_id'])
                output_dict[name]['n_epoch_model'].append(
                    results[idx]['n_epoch_model']
                )
                output_dict[name]['moments_fail'].append(
                    results[idx]['moments_fail']
                )
                output_dict[name]['ntry_fit'].append(
                    results[idx][name]['ntry']
                )
                output_dict[name]['g1_psfo_ngmix'].append(
                    results[idx]['g_PSFo'][0]
                )
                output_dict[name]['g2_psfo_ngmix'].append(
                    results[idx]['g_PSFo'][1]
                )
                output_dict[name]['g1_err_psfo_ngmix'].append(
                    results[idx]['g_err_PSFo'][0]
                )
                output_dict[name]['g2_err_psfo_ngmix'].append(
                    results[idx]['g_err_PSFo'][1]
                )
                output_dict[name]['r50_psfo_ngmix'].append(
                    results[idx]['r50_PSFo']
                )
                output_dict[name]['r50_err_psfo_ngmix'].append(
                    results[idx]['r50_err_PSFo']
                )
                output_dict[name]['g1'].append(results[idx][name]['g'][0])
                output_dict[name]['g1_err'].append(
                    results[idx][name]['pars_err'][2]
                )
                output_dict[name]['g2'].append(results[idx][name]['g'][1])
                output_dict[name]['g2_err'].append(
                    results[idx][name]['pars_err'][3]
                )
                output_dict[name]['r50'].append(results[idx][name]['pars'][4])
                output_dict[name]['r50_err'].append(results[idx][name]['pars_err'][4])
                output_dict[name]['r50psf'].append(results[idx][name]['r50psf'])
                output_dict[name]['g1_psf'].append(
                    results[idx][name]['gpsf'][0]
                )
                output_dict[name]['g2_psf'].append(
                    results[idx][name]['gpsf'][1]
                )
                output_dict[name]['flux'].append(results[idx][name]['flux'])
                output_dict[name]['flux_err'].append(
                    results[idx][name]['flux_err']
                )
                output_dict[name]['mag'].append(mag)
                output_dict[name]['mag_err'].append(mag_err)

                if 's2n' in results[idx][name]:
                    output_dict[name]['s2n'].append(results[idx][name]['s2n'])
                elif 's2n_r' in results[idx][name]:
                    output_dict[name]['s2n'].append(
                        results[idx][name]['s2n_r']
                    )
                else:
                    raise KeyError('No SNR key (s2n, s2n_r) found in results')

                output_dict[name]['flags'].append(results[idx][name]['flags'])
                output_dict[name]['mcal_flags'].append(
                    results[idx]['mcal_flags']
                )

        return output_dict

    def save_results(self, output_dict):
        """Save Results.

        Save the results into a FITS file.

        Parameters
        ----------
        output_dict
            Dictionary containing the results

        """
        output_name = (
            f'{self._output_dir}/ngmix{self._file_number_string}.fits'
        )

        f = file_io.FITSCatalogue(
            output_name,
            open_mode=file_io.BaseCatalogue.OpenMode.ReadWrite
        )

        for key in output_dict.keys():
            f.save_as_fits(output_dict[key], ext_name=key.upper())

    def process(self):
        """Process.

        Funcion to processs NGMIX.
        organizes object cutouts from detection catalog in image, 
        weight, and flag files
        per object: 
            gathers wcs and psf info from exposures
            background subtracts (make this an option)
            scales by relative zeropoints
            runs metacal convolutions and ngmix fitting
        Returns
        -------
        dict
            Dictionary containing the NGMIX metacal results

        """
      
        tile_cat = Tile_cat('')
        # i would like to make this into an object vignet
        vignet_cat = Vignet(
            self._gal_vignet_path,
            self._bkg_vignet_path,
            self._psf_vignet_path,
            self._weight_vignet_path,
            self._flag_vignet_path,
            self._f_wcs_path
        )
  
        final_res = []
        prior = self.get_prior()

        count = 0
        id_first = -1
        id_last = -1

        for i_tile, obj_id in enumerate(tile_cat.obj_id):
            # only run on objects in config file if they are specified (-1 means not set)
            if self._id_obj_min > 0 and obj_id < self._id_obj_min:
                continue
            if self._id_obj_max > 0 and obj_id > self._id_obj_max:
                continue
            if id_first == -1:
                id_first = obj_id
            id_last = obj_id

            count = count + 1

            try:

                stamp = prepare_postage_stamps(vignet_cat)

            except Exception as ee:
                #make an explicit exception here
            #if (
            #    (psf_vign_cat[str(obj_id)] == 'empty')
            #    or (gal_vign_cat[str(obj_id)] == 'empty')
            #):
                continue
            
            #if object is observed, carry out metacal operations and run ngmix
          
            if len(stamp.gals) == 0:
                continue
            try:
                res, psf_res = do_ngmix_metacal(
                    stamp,
                    prior,
                    tile_cat.flux[i_tile],
                    self._pixel_scale,
                    self._rng
                )

            except Exception as ee:
                self._w_log.info(
                    f'ngmix failed for object ID={obj_id}.\nMessage: {ee}'
                )
                continue

            res['obj_id'] = obj_id
            res['n_epoch_model'] = len(stamp.gal_vign_list)
            final_res.append(res)

        self._w_log.info(
            f'ngmix loop over objects finished, processed {count} '
            + f'objects, id first/last={id_first}/{id_last}'
        )

        vignet_cat.close
    
        # Put all results together
        res_dict = self.compile_results(final_res,psf_res)

        # Save results
        self.save_results(res_dict)

def prepare_postage_stamps(vignet, tile_cat, backgroud_subtract=True):
    i_tile = tile_cat.obj_id - 1
    obj_id = tile_cat.obj_id
    # define per-object lists of individual exposures to go into ngmix
    stamp = Postage_stamp()
   
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

        jacob = get_jacob(
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

def get_jacob(wcs, ra, dec):
    """Get Jacobian.
    Return the Jacobian of the WCS at the required position.
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


def get_noise(gal, weight, guess, pixel_scale, thresh=1.2):
    """Get Noise.
    TO DO: modify guess, pixel scale
    Compute the sigma of the noise from an object postage stamp.
    Use a guess on the object size, ellipticity and flux to create a window
    function.

    Parameters
    ----------
    gal : numpy.ndarray
        Galaxy image
    weight : numpy.ndarray
        Weight image
    guess : list
        Gaussian parameters fot the window function
        ``[x0, y0, g1, g2, r50, flux]``
    pixel_scale : float
        Pixel scale of the galaxy image
    thresh : float, optional
        Threshold to cut the window function,
        cut = ``thresh`` * :math:`\sigma_{\rm noise}`;  the default is ``1.2``

    Returns
    -------
    float
        Sigma of the noise on the galaxy image

    """
    img_shape = gal.shape
    m_weight = weight != 0

    sig_tmp = sigma_mad(gal[m_weight])

    gauss_win = galsim.Gaussian(sigma=np.sqrt(guess[4] / 2), flux=guess[5])
    gauss_win = gauss_win.shear(g1=guess[2], g2=guess[3])
    gauss_win = gauss_win.drawImage(
        nx=img_shape[0],
        ny=img_shape[1],
        scale=pixel_scale
    ).array

    m_weight = weight[gauss_win < thresh * sig_tmp] != 0

    sig_noise = sigma_mad(gal[gauss_win < thresh * sig_tmp][m_weight])

    return sig_noise

def prepare_ngmix_weights(gal,weight,flag):
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
    weight_map[np.where(flag != 0)] = 0.
    # This code combines integrates flag information into the weights.
   
    #if gal_guess_flag:
    #    sig_noise = get_noise(
    #        gal,
    #        weight,
    #        gal_guess_tmp,
    #        pixel_scale,
    #    )
    #else:
    sig_noise = sigma_mad(gal)

    noise_img = np.random.randn(*gal.shape) * sig_noise
    noise_img_gal = np.random.randn(*gal.shape) * sig_noise

    gal_masked = np.copy(gal)
    if (len(np.where(weight_map == 0)[0]) != 0):
        gal_masked[weight_map == 0] = noise_img_gal[weight_map == 0]

    weight_map *= 1 / sig_noise ** 2
    
    return gal_masked, weight_map, noise_img

def prepare_galaxy_data(gal,weight,flag,psf,wcs):
    """single galaxy and epoch to be passed to ngmix
    TO DO: pixel scale
    Parameters
    ----------
    gal : numpy.ndarray
        List of the galaxy vignets.  List indices run over epochs
    weight : numpy.ndarray
        List of the PSF vignets
    flag : numpy.ndarray
        flag image
    psf : numpy.ndarray
        psf vignett
    wcs : numpy.ndarray
        Jacobian
    Returns
    -------
    ngmix.observation.Observation
        observation to fit using ngmix

    """
    # prepare psf
    psf_jacob = ngmix.Jacobian(
        row=(psf.shape[0] - 1) / 2,
        col=(psf.shape[1] - 1) / 2,
        wcs=wcs
    )

    psf_obs = Observation(psf, jacobian=psf_jacob)

    # prepare weight map
    gal_masked, weight_map, noise_img = prepare_ngmix_weights(
        gal,
        weight,
        flag
    )
 
    # Recenter jacobian if necessary
    gal_jacob = ngmix.Jacobian(
        row=(gal.shape[0] - 1) / 2,
        col=(gal.shape[1] - 1) / 2,
        wcs=wcs
    )
    # define ngmix observation
    gal_obs = Observation(
        gal_masked,
        weight=weight_map,
        jacobian=gal_jacob,
        psf=psf_obs,
        noise=noise_img
    )
 
    return gal_obs

def average_multiepoch_psf(obsdict,nepoch):
    """ averages psf information over multiple epochs
    we may need to do this for original psf as well
    Parameters
    ----------
    obsdict : dict
        dictionary of metacal observations after fit

    Returns
    -------
    dict
        Average psf size, shape over n_epochs

    """
    # create dictionary
    names = ['T_psf', 'T_psf_err', 'g_psf', 'g_psf_err']
    psf_dict = {k: [] for k in names}
    # include relevant psf quantities- check how they are presented for multi-epoch observations
    wsum = 0
    g_psf_sum = np.array([0., 0.])
    g_psf_err_sum = np.array([0., 0.])
    T_psf_sum = 0
    T_psf_err_sum = 0
    for n_e in np.arange(nepoch):
        T_psf=obsdict['noshear'][n_e].psf.meta['result']['T']
        T_psf_err=obsdict['noshear'][n_e].psf.meta['result']['T_err']
        g_psf=obsdict['noshear'][n_e].psf.meta['result']['g']
        g_psf_err=obsdict['noshear'][n_e].psf.meta['result']['g_err']
        ne_wsum = obsdict['noshear'][0].weight.sum()

        # we probably want to handle cases when there is no psf
        # how are we dealing with the error, what is npsf
        wsum += ne_wsum
        g_psf_sum += g_psf * ne_wsum
        g_psf_err_sum += g_psf_err * ne_wsum
        T_psf_sum += T_psf * ne_wsum
        T_psf_err_sum += T_psf_err * ne_wsum
        #npsf += 1
    if wsum == 0:
        raise ZeroDivisionError('Sum of weights = 0, division by zero')

    psf_dict['g_psf'] = g_psf_sum / wsum
    psf_dict['g_psf_err'] = g_psf_err_sum / wsum
    psf_dict['T_psf'] = T_psf_sum / wsum
    psf_dict['T_psf_err'] = T_psf_err_sum / wsum    

    return psf_dict      

def do_ngmix_metacal(
    stamp,
    prior,
    flux_guess,
    rng
):
    """Do Ngmix Metacal.

    Performs  metacalibration on a sigle multi-epoch object and returns the joint shape measurement with NGMIX.
    TO DO: get pixel scale from jacob_list
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
    dict
        Dictionary containing the results of NGMIX metacal

    """
    n_epoch = len(stamp.gals)

    # are there galaxies to fit?
    if n_epoch == 0:
        raise ValueError("0 epoch to process")

    # fitting options go here, make an option for the future
    psf_model = 'gauss'
    gal_model = 'gauss'

    # Construct multi-epoch observation object to pass to ngmix 
    gal_obs_list = ObsList()

    # create list of ngmix observations for each galaxy
    for n_e in range(n_epoch):
        gal_obs = prepare_galaxy_data(
            stamp.gals[n_e],
            stamp.weights[n_e],
            stamp.flags[n_e],
            stamp.psfs[n_e],
            stamp.jacobs[n_e]
        )
        gal_obs_list.append(gal_obs)
   
    #  decide on fitting options
    fitter = ngmix.fitting.Fitter(model=gal_model, prior=prior)
    # make parameter guesses based on a psf flux and a rough T
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
        rng=rng,
        T=0.25,
        prior=prior,
    )

    # psf fitting a gaussian
    psf_fitter  = ngmix.fitting.Fitter(model=psf_model, prior=prior)
    # TO DO! update flux to sextractor flux                               
    psf_guesser = ngmix.guessers.TFluxGuesser(
        rng=rng,
        T=0.25,
        prior=prior,
        flux=flux_guess,
    )

    # this runs the fitter. We set ntry=2 to retry the fit if it fails
    psf_runner = ngmix.runners.PSFRunner(
        fitter=psf_fitter, guesser=psf_guesser,
        ntry=2,
    )
    runner = ngmix.runners.Runner(
        fitter=fitter, guesser=guesser,
        ntry=5,
    )
    # metacal specific parameters
    metacal_pars = {
        'types': ['noshear', '1p', '1m', '2p', '2m'],
        'step': 0.01,
        'psf': 'fitgauss',
        'fixnoise': True,
        'use_noise_image': True
    }

    # this "bootstrapper" runs the metacal image shearing as well as both psf
    # and object measurements
    boot = ngmix.metacal.MetacalBootstrapper(
        metacal_pars,
        runner=runner, 
        psf_runner=psf_runner,
        ignore_failed_psf=True,
        rng=rng
    )
    # this is the actual fit
    resdict, obsdict = boot.go(gal_obs_list)
    # compile results to include psf information
    psf_res = average_multiepoch_psf(obsdict)
    return resdict, psf_res
