"""
example fitting an exponential model.  The psf is fit
using a set of coelliptical gaussians

Despite this being a fit to single image we use the full fitting framework.  We
use a generic bootstrapper to "bootstrap" the process, first fitting the psf
and then a psf flux for the object.  Then the object is fit including the PSF,
so the inferred parameters are "pre-psf".  The guess for the fit is made
based on the psf flux fit and a generic rough guess for size.

To facilitate this bootstrapping process we define the fitters for psf and
object as well as objects to provide guesses.

Bootstrappers are especially useful when you will perform the same fit on many
objects.
"""

import sys
import numpy as np

try:
    import matplotlib.pylab as plt
except:
    print("Error, matplotlib could not be imported, continuing...")

import galsim
import ngmix
from modopt.math.stats import sigma_mad
from shapepipe.modules.ngmix_package import ngmix as spng
from shapepipe.modules.ngmix_package import postage_stamp as spng_ps

from cs_util import logging


def main():

    print(" === fitting_sp_v3.py ===")
    args = get_args()

    args_arr = [(key, value) for key, value in vars(args).items()]
    logging.log_command(args_arr, name="log_fitting_sp.txt")

    # Pixel scale in arcsec
    scale = args.scale
    
    # Postage stamp size in pixel
    stamp_size = 51

    # Number of epochs
    nepoch = args.n_epoch

    # Object's flux, apparently per epoch
    flux_arr = [3, 6, 12, 20, 30] / np.sqrt(nepoch)

    wcs = set_wcs(args.wcs, scale)

    # PSF Full-width half maximum, in arcsec
    psf_fwhm = args.psf_fwhm

    # Galaxy profile and half-light radius
    profile = args.profile

    if profile == "PointSource":
        gal_hlr = 0.0  # set to zero for point source
        g1 = -0.0002
        g2 = 0.0005
    elif profile in ("Gaussian", "Exponential"):
        gal_hlr = args.gal_hlr
        g1 = -0.02
        g2 = 0.05

    # Number of runs per galaxy
    n_run = args.n_run

    # Allowed are "sp", "ng"
    pipelines = ["sp", "ng"]

    with open("T.txt", "w") as f:
        print_header_to_file(f, pipelines)
       
        for flux in flux_arr:
            results, aux = run_object(scale, args.noise, profile, flux, gal_hlr, psf_fwhm, g1, g2, wcs, stamp_size, nepoch, n_run, pipelines, seed=args.seed)
            print_results_to_file(f, pipelines, flux, results, aux, n_run)
            
    return 0

 
def set_wcs(wcs, scale):
    """Set Wcs.
    
    Set the World Coordinate System (WCS) based on the specified type.

    This function configures the WCS transformation parameters based on the provided type.
    It supports two types: 'weird' and 'diagonal'.

    Parameters
    ----------
        wcs: str
            WCS type
        scale: float
            pixel scale in arcsec
    
    Returns
    --------
    galsim.JacobianWCS
        WCS information
    """
    if wcs == "weird":
        dudx = -0.00105142719975775
        dudy = 0.16467706437987895
        dvdx = 0.15681099855148395
        dvdy = -0.0015749298342502371
    elif wcs == "diagonal":
        dudx = scale
        dudy = 0
        dvdx = 0
        dvdy = scale
    else:
        raise ValueError(f"WCS type {wcs} not implemented")
    return galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)

   
def print_header_to_file(f, pipelines):

    print("# Flux SNR_pix SNR_pix2", file=f, end=" ")
    for key in pipelines:
        print(f"SNR_pix_{key} flux_{key} flux_err_{key} SNR_{key} T_{key} T_err_{key}", file=f, end=" ")
    for key in pipelines:
        print(f"R11_{key} R22_{key}", file=f, end=" ")
    print(file=f)
    

def print_results_to_file(f, pipelines, flux, results, aux, n_run):

    for i_run in range(n_run):
        print(f"{flux}", file=f, end=" ")
        print(f"{aux['snr_pix'][i_run]:.5g}", file=f, end=" ")
        print(f"{aux['snr_pix2'][i_run]:.5g}", file=f, end=" ")

        for key in pipelines:
            print(
                f"{aux[f'snr_{key}'][i_run]}"
                + f" {results[key][i_run]['noshear']['flux']:.5g}"
                + f" {results[key][i_run]['noshear']['flux_err']:.5g}"
                + f" {results[key][i_run]['noshear']['s2n']:.5g}"
                + f" {results[key][i_run]['noshear']['T']:.5g}"
                + f" {results[key][i_run]['noshear']['T_err']:.5g}",
                file=f,
                end=" ",
            )
        for key in pipelines:
            R11, R22 = get_Rii(results[key][i_run])
            print(f"{R11:.5g} {R22:.5g}", file=f, end=" ")
        print(file=f)


def run_object(scale, noise, profile, flux, gal_hlr, psf_fwhm, g1, g2, wcs, stamp_size, nepoch, n_run, pipelines, seed=None):

    results = {}
    aux = {}
    obsdicts = {}
    for key in pipelines:
        results[key] = []
        obsdicts[key] = []
        aux[f"snr_{key}"] = []
    aux["snr_pix"] = []
    aux["snr_pix2"] = []

    rng = np.random.default_rng(seed)
    prior = get_prior(rng=rng, scale=scale)
    boot, fitter = get_bootstrap(rng, prior, flux)


    # Initialize random generator for both pipelines with same seed
    rng_sp = np.random.default_rng(seed)
    rng_np = np.random.default_rng(seed)

    # Run shape measurement pipelines
    for i_run in range(n_run):

        print(f"===== Run {i_run}/{n_run} =====")

        stamp, obs, snr_pix, snr_pix2 = make_stamp(rng, nepoch, scale, flux, noise, stamp_size, g1, g2, wcs, profile, psf_fwhm, gal_hlr=gal_hlr)
        aux["snr_pix"].append(snr_pix)
        aux["snr_pix2"].append(snr_pix2)

        # Fit shape using ShapePipe ngmix module
        if "sp" in pipelines:
            res, obsdict, s2n = fit_shapes_sp(rng_sp, nepoch, scale, obs, stamp, boot, fitter)
            results["sp"].append(res)
            obsdicts["sp"].append(obsdict)
            aux["snr_sp"].append(s2n)

        # Fit shape using ngmix package
        if "ng" in pipelines:
            res, obsdict, s2n = fit_shapes_ngmix(rng_np, nepoch, obs, boot)
            results["ng"].append(res)
            obsdicts["ng"].append(obsdict)
            aux["snr_ng"].append(s2n)

    # Print results
    sigma = gal_hlr / np.sqrt(2 * np.log(2))
    T = 2 * sigma ** 2

    sigma_psf = psf_fwhm / (2 * np.sqrt(2 * np.log(2)))
    T_psf = 2 * sigma_psf ** 2

    obj_pars = get_obj_pars(g1, g2, flux, T, T_psf, 0, 0)

    for i_run in range(n_run):
        for key in results:
            print_results(results[key][i_run], obsdicts[key][i_run], key, obj_pars)
            pass

    summary = summary_runs_all(results)

    print("print bias")
    print_bias(summary, obj_pars)
    print("print bias done")
    if False:
        plot_g1g2(summary, obj_pars)

    return results, aux


def print_bias(summary, obj_pars):

    print("Residual shear biases:")
    print("code dg1     dg1/g1   dg2     dg2/g2")
    print("------------------------------------")
    for key in summary:
        print(key, end="   ")
        for comp in (1, 2):
            delta_g = summary[key]["g_mean"][comp - 1] - obj_pars[f"g{comp}"]
            delta_g_rel = delta_g / obj_pars[f"g{comp}"]
            print(f"{delta_g:.4f} {delta_g_rel:.4f}", end="   ")
        print()

def plot_g1g2(summary, obj_pars):

    plt.figure()

    x = []
    y = []
    dx = []
    dy = []
    symbols = ["o", "x"]
    colors = ["blue", "orange"]
    
    for idx, key in enumerate(summary):
        x = summary[key]["g_mean"][0]
        dx = summary[key]["g_std"][0]
        y = summary[key]["g_mean"][1]
        dy = summary[key]["g_std"][1]
        plt.errorbar(
            x,
            y,
            xerr=dx,
            yerr=dy,
            color=colors[idx],
            label=key
        )
        plt.plot(x, y, marker=symbols[idx], color=colors[idx])

    plt.scatter(obj_pars["g1"], obj_pars["g2"], color="k", label="truth")
    
    plt.xlabel(r"$g_1$")
    plt.ylabel(r"$g_2$")
    plt.legend()
    plt.savefig("g1g2.png")

def summary_runs(results):

    summary = {
        "g_mean" : np.zeros(2),
        "g_std" : np.zeros(2),
    }

    n_run = len(results)
    for comp in (0, 1):
        summary["g_mean"][comp] = np.mean(
            [results[idx]["noshear"]["g"][comp] for idx in range(n_run)]
        )
        summary["g_std"][comp] = np.std(
            [results[idx]["noshear"]["g"][comp] for idx in range(n_run)]
        )
        
    return summary


def summary_runs_all(results):

    summary = {}

    for key in results:
        summary[key] = summary_runs(results[key])

    return summary


def make_stamp(rng, nepoch, scale, flux, noise, stamp_size, g1, g2, wcs, profile, psf_fwhm, gal_hlr=0):

    stamp = spng_ps.Postage_stamp(rng)

    stamp_size_xy = np.array([stamp_size] * 2)
    prior = get_prior(rng=rng, scale=scale)
    obs = []

    snr_pix_sum = 0
    snr_pix2_sum = 0

    for _ in range(nepoch):
        this_psf, this_psf_obs, this_psf_im = make_psf(rng, stamp_size, wcs, psf_fwhm)

        #dy = dx = 0
        dy, dx = rng.uniform(low=-scale/10, high=scale/10, size=2)

        this_obs, this_wt, this_noise = make_data(rng, noise, this_psf, this_psf_obs, wcs, dx, dy, g1, g2, stamp_size, profile, scale=scale, flux=flux, gal_hlr=gal_hlr)

        # Make sure all postage stamps have the same size
        if any(this_obs.image.shape != stamp_size_xy):
            raise ValueError(f"gal image has wrong size: {this_obs.image.shape}")
        if any(this_psf_im.shape != stamp_size_xy):
            raise ValueError(f"psf image has wrong size: {this_psf_im.shape}")

        # Compute pixel-sum SNR, see ngmix:test_observation.py:test_observation_s2n
        snr_pix = np.sum(this_obs.image) / np.sqrt(np.sum(1.0/this_wt))
        snr_pix_sum += snr_pix ** 2

        # Compute pixel-square sum SNR, see ngmix.gmix_nb:get_loglike
        # or gmix_nb:get_model_s2n_sum
        # but with model -> pixel value
        s2n_numer = 0
        #s2n_denom = 0
        for index in np.ndindex(this_obs.image.shape):
            idx, jdx = index
            s2n_numer += this_obs.image[idx, jdx] ** 2 * this_wt[idx, jdx]
            #s2n_denom += this_obs.image[idx, jdx] ** 2 * this_wt[idx, jdx]
        #snr_pix2 = s2n_numer / np.sqrt(s2n_denom)
        snr_pix2 = np.sqrt(s2n_numer)
        snr_pix2_sum += snr_pix2 ** 2

        # obs used in ngmix package
        obs.append(this_obs)
        stamp.gals.append(this_obs.image)
        stamp.psfs.append(this_psf_im)
        stamp.weights.append(this_wt)
        stamp.jacobs.append(wcs)
        stamp.noise_ims.append(this_noise)

    snr_pix_sum = np.sqrt(snr_pix_sum)
    snr_pix2_sum = np.sqrt(snr_pix2_sum)

    return stamp, obs, snr_pix_sum, snr_pix2_sum


def get_bootstrap(rng, prior, flux):

    # fit using the levenberg marquards algorithm
    fitter = ngmix.fitting.Fitter(model='gauss', prior=prior)

    # make parameter guesses based on a psf flux and a rough T
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
        rng=rng,
        T=0.25,
        prior=prior,
    )

    # psf fitting with coelliptical gaussians # we need to change the psf fitter to FitGauss
    psf_fitter = fitter

    # special guesser for coelliptical gaussians
    psf_guesser = ngmix.guessers.TFluxGuesser(
        rng=rng,
        T=0.25,
        prior=prior,
        flux=flux,
    )

    # this runs the fitter. We set ntry=2 to retry the fit if it fails
    psf_runner = ngmix.runners.PSFRunner(
        fitter=psf_fitter,
        guesser=psf_guesser,
        ntry=2,
    )
    runner = ngmix.runners.Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=5,
    )

    # this bootstraps the process, first fitting psfs then the object
    metacal_bootstrapper = ngmix.metacal.MetacalBootstrapper(
        runner=runner,
        psf_runner=psf_runner,
        rng=rng,
        ignore_failed_psf=True,
        psf='fitgauss',
    )

    return metacal_bootstrapper, fitter


def my_make_ngmix_observation(gal,weight,psf,wcs,noise_img):                       
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
    
    psf_obs = ngmix.Observation(psf, jacobian=psf_jacob)                               
    
    # define ngmix observation                                                   
    gal_obs = ngmix.Observation(                                                 
        gal,                                                                     
        weight=weight,                                                           
        jacobian=gal_jacob,                                                      
        psf=psf_obs,                                                             
        noise=noise_img                                                          
    )                                                                            
    
    return gal_obs


def fit_shapes_sp(rng, nepoch, scale, obs, stamp, boot, fitter):

    # Create (empty) flag arrays
    for iepoch in range(nepoch):
        flag = np.zeros_like(stamp.weights[iepoch])
        stamp.flags.append(flag)

    stamp.bkg_sub = False
    stamp.rescale_weights = False
    #stamp.megacam_flip = True

    gal_obs_list = ngmix.observation.ObsList()
    #gal_obs_list = ngmix.ObsList()
    # Call ShapePipe ngmix functions 
    for n_e in range(nepoch):
        gal_obs = spng._make_ngmix_observation(
        #gal_obs = my_make_ngmix_observation(
            stamp.gals[n_e],
            stamp.weights[n_e],
            stamp.psfs[n_e],
            stamp.jacobs[n_e],
            stamp.noise_ims[n_e]
        )
        #gal_obs = obs[n_e]
        gal_obs_list.append(gal_obs)

    res, obsdict = boot.go(gal_obs_list)

    # ngmix observational pixel-sum SNR, equal to aux["snr_sp"]
    s2n_pix = gal_obs_list.get_s2n()

    # Model SNR, how can we get it?
    #fit_model = fitter._make_fit_model(obs=gal_obs_list, guess=[5, 0, 0, 0.25, 0, 0.25])
    #gmix = fit_model.get_gmix()
    #s2n = gmix.get_model_s2n(obs)
    #print("model s2n = ", s2n)

    return res, obsdict, s2n_pix


def fit_shapes_ngmix(rng, nepoch, obs, boot):

    # Local ngmix runner
    gal_obs_list = ngmix.observation.ObsList()

    for iepoch in range(nepoch):
        gal_obs_list.append(obs[iepoch])

    res, obsdict = boot.go(gal_obs_list)

    if False:
        try:
            import images
        except ImportError:
            from espy import images

        imfit = res.make_image()

        images.compare_images(obs.image, imfit)

    # ngmix observational pixel-sum SNR, equal to aux["snr_ng"]
    s2n_pix = gal_obs_list.get_s2n()

    return res, obsdict, s2n_pix


# Same function as Ngmix.get_prior()
def get_prior(*, rng, scale, T_range=None, F_range=None, nband=None):
    """
    get a prior for use with the maximum likelihood fitter

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    scale: float
        Pixel scale
    T_range: (float, float), optional
        The range for the prior on T
    F_range: (float, float), optional
        Fhe range for the prior on flux
    nband: int, optional
        number of bands
    """
    if T_range is None:
        #T_range = [-1.0, 1.e3]
        T_range = [0.0, 1.e3]
    if F_range is None:
        F_range = [-100.0, 1.e9]

    # sigma was 0.1
    g_prior = ngmix.priors.GPriorBA(sigma=0.4, rng=rng)
    cen_prior = ngmix.priors.CenPrior(
        cen1=0, cen2=0, sigma1=scale, sigma2=scale, rng=rng,
    )
    T_prior = ngmix.priors.FlatPrior(minval=T_range[0], maxval=T_range[1], rng=rng)
    F_prior = ngmix.priors.FlatPrior(minval=F_range[0], maxval=F_range[1], rng=rng)

    if nband is not None:
        F_prior = [F_prior]*nband

    return ngmix.joint_prior.PriorSimpleSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        F_prior=F_prior,
    )


def make_psf(rng, stamp_size, wcs, psf_fwhm):
    """Make PSF

    Simulate MoffatPSF.

    """
    psf_noise = 1.0e-6

    psf = galsim.Moffat(
        beta=2.5, fwhm=psf_fwhm,
    ).shear(
        g1=-0.01,
        g2=-0.01,
    )
    
    psf_im = psf.drawImage(nx=stamp_size, ny=stamp_size, wcs=wcs).array

    # add noise
    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)

    psf_wt = psf_im * 0 + 1 / psf_noise ** 2

    psf_cen = (np.array(psf_im.shape) - 1) / 2

    psf_jacobian = ngmix.Jacobian(
        x=psf_cen[1], 
        y=psf_cen[0], 
        wcs=wcs.jacobian(
            image_pos=galsim.PositionD(psf_cen[1], psf_cen[0])
        ),
    )

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=psf_jacobian,
    )

    return psf, psf_obs, psf_im


def model_func(profile, g1, g2, dx, dy, flux, half_light_radius=0.5):

    if profile == "Gaussian":
        model_func = galsim.Gaussian
    elif profile == "Exponential":
        model_func = galsim.Exponential
    else:
        model_func = None

    if profile in ("Gaussian", "Exponential"):

        obj0 = model_func(
            half_light_radius=half_light_radius,
            flux=flux,
        ).shear(
            g1=g1,
            g2=g2,
        ).shift(
            dx=dx,
            dy=dy,
        )

    elif profile == "PointSource":
        obj0 = galsim.DeltaFunction(
            flux=flux,
        ).shear(
            g1=g1,
            g2=g2
        ).shift(
            dx=dx,
            dy=dy,
        )

    else:
        raise ValueError(f"Unknown profile {profile}")

    return obj0


def make_data(rng, noise, psf, psf_obs, wcs, dx, dy, g1, g2, stamp_size, profile, scale=1.0, flux=100.0, gal_hlr=0):
    """
    simulate an exponential object convolved with the psf

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    noise: float
        Noise for the image
    g1: float
        object g1, default 0.05
    g2: float
        object g2, default -0.02
    flux: float, optional
        default 100

    Returns
    -------
    ngmix.Observation, pars dict

    """
    obj0 = model_func(profile, g1, g2, dx, dy, flux, gal_hlr)
    
    obj = galsim.Convolve(psf, obj0)

    im = obj.drawImage(nx=stamp_size, ny=stamp_size, wcs=wcs).array

    # add noise
    im += rng.normal(scale=noise, size=im.shape)

    cen = (np.array(im.shape) - 1) / 2

    noise = sigma_mad(im)
    wt = im * 0 + 1 / noise ** 2
    # can we extract noise image from postage stamp?
    noise_img = np.random.randn(*im.shape) * noise

    # No difference with or without scale (?)
    jacobian = ngmix.Jacobian(
        row=cen[0] + dy/scale,
        col=cen[1] + dx/scale,
        scale=scale,
        wcs=wcs,
    )
    # wcs and wcs.jacobian() seem to be the same

    obs = ngmix.Observation(
        im,
        weight=wt,
        jacobian=jacobian,
        psf=psf_obs,
        noise=noise_img,
    )

    return obs, wt, noise_img


def get_obj_pars(g1, g2, flux, T, T_psf, dx, dy):

    return {
        'g1': g1,
        'g2': g2,
        'flux': flux,
        'T': T,
        'T_psf': T_psf,
        'dx': dx,
        'dy': dy,
    }


def print_results(res, obsdict, key, obj_pars):

    print(f"===== results {key} =====")
    print('S/N:', res['noshear']['s2n'])
    print('true flux: %g meas flux: %g +/- %g (99.7%% conf)' % (
        obj_pars['flux'], res['noshear']['flux'], res['noshear']['flux_err']*3,
    ))
    print(f"true T: {obj_pars['T']:g} arcsec meas T: {res['noshear']['T']:g} +/- {res['noshear']['T_err']*3:g} (99.7% conf) arcsec")
    print(f"meas T/T_psf: {res['noshear']['T']/obj_pars['T_psf']:g}")
    print('true g1: %g meas g1: %g +/- %g (99.7%% conf)' % (
        obj_pars['g1'], res['noshear']['g'][0], res['noshear']['g_err'][0]*3,
    ))
    print('true g2: %g meas g2: %g +/- %g (99.7%% conf)' % (
        obj_pars['g2'], res['noshear']['g'][1], res['noshear']['g_err'][1]*3,
    ))

    R11, R22 = get_Rii(res)
    print(f"R11 = {R11:g}, R22 = {R22:g}")
    print(f"T_psf = {obj_pars['T_psf']:g} arsec^2")

    print(
        "Delta g / sig g = ("
        + f"{(obj_pars['g1'] - res['noshear']['g'][0]) / res['noshear']['g_err'][0]:.2f},"
        + f" {(obj_pars['g2'] - res['noshear']['g'][1]) / res['noshear']['g_err'][1]:.2f})"
    )
    print()

    
def get_Rii(res):

    step = 0.01
    R11 = (res["1p"]["g"][0] - res["1m"]["g"][0]) / (2 * step)
    R22 = (res["2p"]["g"][1] - res["2m"]["g"][1]) / (2 * step)

    return R11, R22

def get_args():
    
    def_noise = 0.01
    def_n_epoch = 1
    def_psf_fwhm = 0.68
    def_scale = 0.187
    def_gal_hlr = 0.5
    def_n_run = 25
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for rng')
    parser.add_argument('--show', action='store_true',
                        help='show plot comparing model and data')
    parser.add_argument('--noise', type=float, default=def_noise,
                        help=f'noise for images, default is {def_noise}')
    parser.add_argument('--psf_fwhm', type=float, default=def_psf_fwhm,
                        help=f'PSF FWHM [arcsec], default is {def_psf_fwhm}')
    parser.add_argument('--scale', type=float, default=def_scale,
                        help=f'pixel scale [arcsec], default is {def_scale}')
    parser.add_argument('--gal_hlr', type=float, default=def_gal_hlr,
                        help=f'Galaxy half-light radius [arcsec], default is {def_gal_hlr}')
    parser.add_argument('--n_epoch', type=int, default=def_n_epoch,
                        help=f'number of epochs. default is {def_n_epoch}')
    parser.add_argument('--n_run', type=int, default=def_n_run,
                        help=f'number of runs. default is {def_n_run}')
    parser.add_argument('--profile', type=str, default="Gaussian",
                        help="Object profile, allowed are 'PointSource', 'Gaussian' (default)")
    parser.add_argument('--wcs', type=str, default="weird",
                        help="WCS type, allowed are 'diagonal', 'weird' (default)")
    return parser.parse_args()


if __name__ == '__main__':
    main()
