"""NGMIX.

This module contains a class for ngmix shape measurement.

:Authors: Lucie Baumont, Axel Guinot

To list for changes to this module
output dictionary is a mess

do tests on background- how bad is poisson noise

extract pixel scale from wcs
do extra fit to ngmix psf and combine info in post processing

we want to add a keep columns function to file.io. we need to test this. then the sextractor catalog can be an input into postage stamp class
"""

import re
import ngmix
import galsim
import numpy as np
from astropy.io import fits
from modopt.math.stats import sigma_mad
from ngmix.observation import Observation, ObsList
from sqlitedict import SqliteDict
from shapepipe.modules.ngmix_package import ngmix_postprocess
from shapepipe.modules.ngmix_package import postage_stamp as ps

from shapepipe.pipeline import file_io

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
    bkg_sub: bool, optional
        default is ``True`` for background subtraction
    megacam_flip: bool, optional
        default is ``True`` to flip megaprime coordinates with megapipe processing
    mask_frac: float
        maximum fraction of allowed masked pixels, default is 1/3
    rescale_weights: bool
        default is ``True`` to rescale weights into variance maps
    symmetrize_mask: bool
        ``True`` will symmetrize mask, default is ``False`` 

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
        f_wcs_path,
        w_log,
        id_obj_min=-1,
        id_obj_max=-1,
        bkg_sub=True,
        megacam_flip=True,
        mask_frac=1/3.,
        rescale_weights=True,
        symmetrize_mask=False
        
    ):

        if len(input_file_list) != 6:
            raise IndexError(
                f'Input file list has length {len(input_file_list)},'
                + ' required is 6'
            )

        self._tile_cat_path = input_file_list[0]
        self._vignet_cat = ps.Vignet(
            input_file_list[1],
            input_file_list[2],
            input_file_list[3],
            input_file_list[4],
            input_file_list[5],
            f_wcs_path
        )

        self._output_dir = output_dir
        self._file_number_string = file_number_string
        self._zero_point = zero_point

        
        self._id_obj_min = id_obj_min
        self._id_obj_max = id_obj_max

        self._w_log = w_log

        # Initiatlise random generator
        seed = int(''.join(re.findall(r'\d+', self._file_number_string)))
        self._rng = np.random.RandomState(seed)
        self._w_log.info(f'Random generator initialisation seed = {seed}')

    @classmethod
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

        Function to processs NGMIX.
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
      
        tile_cat = ps.Tile_cat(self._tile_cat_path)
        tile_cat.get_data()

        vignet_cat = self._vignet_cat  

        final_res = []
        prior = self.get_prior()



        ############ LOOP THROUGH OBJECTS- probably can be less dumb
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

            # used only for logfile
            count = count + 1
            
            # make postage stamp, skip if not observed
            try:
                stamp=ps.Postage_stamp(
                    self._rng,
                    bkg_sub=True,
                    megacam_flip=True,
                    mask_frac=1/3.0,
                    rescale_weights=True,
                    symmetrize_mask=False
                    )
                stamp.preprocess_postage_stamps(vignet_cat, tile_cat, obj_id)
            except AttributeError:
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
            # these things need to be considered
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

def do_ngmix_metacal(
    stamp,
    prior,
    flux_guess,
    rng
):
    """Do Ngmix Metacal.

    Performs  metacalibration on a single multi-epoch object and returns the joint shape measurement with NGMIX.
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
    model = 'gauss'
    psf_model='fitgauss'
    

    # Construct multi-epoch observation object to pass to ngmix 
    gal_obs_list = ObsList()

    # create list of ngmix observations for each galaxy- this maycd  change with stamp bookkeeping
    for n_e in range(n_epoch):
        gal_obs = _make_ngmix_observation(
            stamp.gals[n_e],
            stamp.weights[n_e],
            stamp.psfs[n_e],
            stamp.jacobs[n_e],
            stamp.noise_ims[n_e]
        )
        gal_obs_list.append(gal_obs)
   
    #  decide on fitting options
    fitter = ngmix.fitting.Fitter(model=model, prior=prior)
    # make parameter guesses based on a psf flux and a rough T
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
        rng=rng,
        T=0.25,
        prior=prior,
    )

    # psf fitting a gaussian
    psf_fitter  = ngmix.fitting.Fitter(model=model, prior=prior)
    # TO DO! what do we do about size?                              
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

    # this "bootstrapper" runs the metacal image shearing as well as both psf
    # and object measurements
    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner, 
        psf_runner=psf_runner,
        rng=rng,
        ignore_failed_psf=True,
        psf=psf_model
    )
    # this is the actual fit
    resdict, obsdict = boot.go(gal_obs_list)
    # compile results to include psf information
    psf_res = ngmix_postprocess.average_multiepoch_psf(obsdict, n_epoch)

    return resdict, obsdict, psf_res

def _make_ngmix_observation(gal,weight,psf,wcs,noise_img):
    """single galaxy and epoch to be passed to ngmix
   
    Parameters
    ----------
    gal : numpy.ndarray
        List of the galaxy vignets.  List indices run over epochs
    weight : numpy.ndarray
        List of the PSF vignets
    psf : numpy.ndarray
        psf vignett
    wcs : galsim.JacobianWCS
        Jacobian
    Returns
    -------
    ngmix.observation.Observation
        observation to fit using ngmix

    """
    # prepare psf
    # WHY RECENTER from detection center to tile center
    psf_jacob = ngmix.Jacobian(
        row=(psf.shape[0] - 1) / 2,
        col=(psf.shape[1] - 1) / 2,
        wcs=wcs
    )
     # Recenter jacobian if necessary
    gal_jacob = ngmix.Jacobian(
        row=(gal.shape[0] - 1) / 2,
        col=(gal.shape[1] - 1) / 2,
        wcs=wcs
    )

    psf_obs = Observation(psf, jacobian=psf_jacob)

    # define ngmix observation
    gal_obs = ngmix.Observation(
        gal,
        weight=weight,
        jacobian=gal_jacob,
        psf=psf_obs,
        noise=noise_img
    )
 
    return gal_obs


