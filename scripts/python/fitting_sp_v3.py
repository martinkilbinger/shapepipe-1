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
import matplotlib.pylab as plt

import galsim
import ngmix
from modopt.math.stats import sigma_mad
from shapepipe.modules.ngmix_package import ngmix as spng
from shapepipe.modules.ngmix_package import postage_stamp as spng_ps


def main():

    print(" === fitting_sp_v3.py ===")
    args = get_args()

    # Pixel scale in arcsec
    scale = 0.187
    
    # Postage stamp size in pixel
    stamp_size = 51

    # Number of epochs
    nepoch = 1

    # Object's flux
    #flux_arr = [1, 2, 3, 4, 5, 7.5, 10, 12.5, 15, 20, 25]
    flux_arr = [1, 2, 3, 5, 8, 10]

    if args.wcs == "weird":
        dudx = -0.00105142719975775
        dudy = 0.16467706437987895
        dvdx = 0.15681099855148395
        dvdy = -0.0015749298342502371
    elif args.wcs == "diagonal":
        dudx = scale
        dudy = 0
        dvdx = 0
        dvdy = scale
    else:
        raise ValueError(f"WCS type {args.wcs} not implemented")
    wcs = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)

    # Galaxy profile and half-light radius
    profile = "PointSource"
    gal_hlr = 0.0  # set to zero for point source
    g1 = -0.0002
    g2 = 0.0005

    #profile = "Gaussian"
    #gal_hlr = 0.5
    #g1 = -0.02
    #g2 = 0.05

    # PSF full width half maximum in arcsec
    psf_fwhm = 0.68

    # Number of runs per galaxy
    n_run = 25

    with open("T.txt", "w") as f:
        print("# Flux SNR_sp T_sp T_err_sp SNR_ng T_ng T_ng_err R11_sp R22_sp R11_nng R22_ng", file=f)
        
        for flux in flux_arr:
            results = run_object(args.seed, scale, args.noise, profile, flux, gal_hlr, psf_fwhm, g1, g2, wcs, stamp_size, nepoch, n_run)
            
            for i_run in range(n_run):
                R11 = {}
                R22 = {}
                for key in ("sp", "ng"):
                    R11[key], R22[key] = get_Rii(results[key][i_run])
                    
                print(
                    f"{flux} {results['sp'][i_run]['noshear']['s2n']} {results['sp'][i_run]['noshear']['T']:g}"
                    + f" {results['sp'][i_run]['noshear']['T_err']:g}"
                    + f" {results['ng'][i_run]['noshear']['s2n']} {results['ng'][i_run]['noshear']['T']:g} {results['ng'][i_run]['noshear']['T_err']:g} "
                    + f" {R11['ng']} {R22['ng']} {R11['ng']} {R22['ng']}",
                    file=f,
                )

    return 0
    

def run_object(seed, scale, noise, profile, flux, gal_hlr, psf_fwhm, g1, g2, wcs, stamp_size, nepoch, n_run):

    results = {}
    obsdicts = {}
    pipelines = ["sp", "ng"]
    for key in pipelines:
        results[key] = []
        obsdicts[key] = []

    rng = np.random.RandomState(seed)
    prior = get_prior(rng=rng, scale=scale)
    boot = get_bootstrap(rng, prior, flux)

    # Run shape measurement pipelines
    for i_run in range(n_run):

        print(f"===== Run {i_run}/{n_run} =====")

        stamp, obs = make_stamp(rng, nepoch, scale, flux, noise, stamp_size, g1, g2, wcs, profile, psf_fwhm, gal_hlr=gal_hlr)

        # Fit shape using ShapePipe ngmix module
        rng = np.random.RandomState(seed)
        res, obsdict = fit_shapes_sp(rng, nepoch, scale, obs, stamp, boot)
        results["sp"].append(res)
        obsdicts["sp"].append(obsdict)

        # Fit shape using ngmix package
        rng = np.random.RandomState(seed)
        res, obsdict = fit_shapes_ngmix(rng, nepoch, obs, boot)
        results["ng"].append(res)
        obsdicts["ng"].append(obsdict)

    # Print results
    sigma = gal_hlr / np.sqrt(2 * np.log(2))
    T = 2 * sigma ** 2

    sigma_psf = psf_fwhm / (2 * np.sqrt(2 * np.log(2)))
    T_psf = 2 * sigma_psf ** 2

    obj_pars = get_obj_pars(g1, g2, flux, T, T_psf, 0, 0)

    for i_run in range(n_run):
        for key in results:
            print_results(results[key][i_run], obsdicts[key][i_run], key, obj_pars)

    summary = summary_runs_all(results)
    #print(summary)

    print_bias(summary, obj_pars)
    plot_g1g2(summary, obj_pars)

    return results


def print_bias(summary, obj_pars):

    print(f"Residual shear biases:")
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

    for iepoch in range(nepoch):
        this_psf, this_psf_obs, this_psf_im = make_psf(rng, stamp_size, wcs, psf_fwhm)

        #dy = dx = 0
        dy, dx = rng.uniform(low=-scale/10, high=scale/10, size=2)

        this_obs, this_wt, this_noise = make_data(rng, noise, this_psf, this_psf_obs, wcs, dx, dy, g1, g2, stamp_size, profile, scale=scale, flux=flux, gal_hlr=gal_hlr)

        # Make sure all postage stamps have the same size
        if any(this_obs.image.shape != stamp_size_xy):
            raise ValueError(f"gal image has wrong size: {this_obs.image.shape}")
        if any(this_psf_im.shape != stamp_size_xy):
            raise ValueError(f"psf image has wrong size: {this_psf_im.shape}")

        # obs used in ngmix package
        obs.append(this_obs)
        stamp.gals.append(this_obs.image)
        stamp.psfs.append(this_psf_im)
        stamp.weights.append(this_wt)
        stamp.jacobs.append(wcs)
        stamp.noise_ims.append(this_noise)

    return stamp, obs


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
    return ngmix.metacal.MetacalBootstrapper(
        runner=runner,
        psf_runner=psf_runner,
        rng=rng,
        ignore_failed_psf=True,
        psf='fitgauss',
    )


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


def fit_shapes_sp(rng, nepoch, scale, obs, stamp, boot):

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

    return res, obsdict


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

    return res, obsdict


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

    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        F_prior=F_prior,
    )

    return prior


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

    # MKDEBUG shrink obs for ps testing
    #print("MKDEBUG shrink")
    #obj = obj.dilate(0.99)

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

    obj_pars = {
        'g1': g1,
        'g2': g2,
        'flux': flux,
        'T': T,
        'T_psf': T_psf,
        'dx': dx,
        'dy': dy,
    }

    return obj_pars



def print_results(res, obsdict, key, obj_pars):

    print(f"===== results {key} =====")
    #print(obj_pars)
    #print(res['noshear']['pars'])
    #print('obs image shape', obsdict['noshear'][0].image.shape)
    #print('obs image sum', obsdict['noshear'][0].image.sum())

    #print('obs noise std', obsdict['noshear'][0].noise.std())
    #print('obs weight std', obsdict['noshear'][0].weight.std())
    #print(f"obs weight sum {obsdict['noshear'][0].weight.sum():.2e}")
    #print('obs weight[0[0]', obsdict['noshear'][0].weight[0][0])
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
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for rng')
    parser.add_argument('--show', action='store_true',
                        help='show plot comparing model and data')
    parser.add_argument('--noise', type=float, default=def_noise,
                        help=f'noise for images, default is {def_noise}')
    parser.add_argument('--wcs', type=str, default="weird",
                        help="WCS type, allower are 'diagonal', 'weird' (default)")
    return parser.parse_args()


if __name__ == '__main__':
    main()
