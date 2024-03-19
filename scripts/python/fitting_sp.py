"""
example fitting an exponential model.  The psf is fit
using a set of coelliptical gaussians

Despite this being a fit to single image we use the full fitting framework.  We
use a generic bootstrapper to "bootstrap" the process, first fitting the psf
and then a psf flux for the object.  Then the object is fit including the PSF,
so the inferred parameters are "pre-psf".  The guess for the fit is made
based on the psf flux fit and a generic rough guess for size.

To faciliate this bootstrapping process we define the fitters for psf and
object as well as objects to provide guesses.

Bootstrappers are especially useful when you will perform the same fit on many
objects.

A run of the code should produce output something like thid

    > python fitting_bd_empsf.py

    S/N: 920.5078121454815
    true flux: 100 meas flux: 95.3763 +/- 0.653535 (99.7% conf)
    true g1: 0.05 meas g1: 0.0508 +/- 0.00960346 (99.7% conf)
    true g2: -0.02 meas g2: -0.0261123 +/- 0.0095837 (99.7% conf)
    true fracdev: 0.5 meas fracdev: 0.514028 +/- 0.011873 (99.7% conf)
"""

import numpy as np
import matplotlib.pylab as plt

import galsim
import ngmix
from modopt.math.stats import sigma_mad
from shapepipe.modules.ngmix_package import ngmix as spng
from shapepipe.modules.ngmix_package import postage_stamp as spng_ps


def main():

    print(" === fitting_sp.py ===")
    args = get_args()
    rng = np.random.RandomState(args.seed)

    nepoch = 3

    scale = 0.263
    flux = 100
    stamp_size = 53

    wcs = galsim.JacobianWCS(
        -0.00105142719975775,
        0.16467706437987895,
        0.15681099855148395,
        -0.0015749298342502371
    )

    g1 = -0.02
    g2 = 0.05


    n_run = 25

    results = {}
    obsdicts = {}
    pipelines = ["sp", "ng"]
    for key in pipelines:
        results[key] = []
        obsdicts[key] = []

    # Run shape measurement pipelines
    for i_run in range(n_run):

        print(f"===== Run {i_run}/{n_run} =====")

        prior = get_prior(rng=rng, scale=scale)

        stamp, obs = make_stamp(rng, nepoch, scale, flux, args.noise, stamp_size, g1, g2, wcs)

        # Fit shape using ShapePipe ngmix module
        res, obsdict = fit_shapes_sp(args, rng, nepoch, scale, flux, prior, stamp)
        results["sp"].append(res)
        obsdicts["sp"].append(obsdict)

        # Fit shape using ngmix package
        res, obsdict = fit_shapes_ngmix(args, rng, nepoch, scale, flux, prior, obs, stamp.psfs)
        results["ng"].append(res)
        obsdicts["ng"].append(obsdict)

    # Print results
    obj_pars = get_obj_pars(g1, g2, flux, 0, 0)
    for i_run in range(n_run):
        for key in results:
            print_results(results[key][i_run], obsdicts[key][i_run], key, obj_pars)

    summary = summary_runs_all(results)
    print(summary)

    plot_g1g2(summary, obj_pars)


def plot_g1g2(summary, obj_pars):

    plt.figure()

    x = []
    y = []
    dx = []
    dy = []

    for key in summary:
        x = summary[key]["g_mean"][0]
        dx = summary[key]["g_std"][0]
        y = summary[key]["g_mean"][1]
        dy = summary[key]["g_std"][1]
        plt.errorbar(x, y, xerr=dx, yerr=dy, label=key)

    plt.scatter(obj_pars["g1"], obj_pars["g2"])

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


def make_stamp(rng, nepoch, scale, flux, noise, stamp_size, g1, g2, wcs):

    stamp = spng_ps.Postage_stamp()

    stamp_size_xy = np.array([stamp_size] * 2)
    prior = get_prior(rng=rng, scale=scale)
    obs = []

    for iepoch in range(nepoch):
        this_psf, this_psf_obs, this_psf_im = make_psf(rng, stamp_size, wcs)

        dy, dx = rng.uniform(low=-scale/2, high=scale/2, size=2)

        this_obs, this_wt, this_jacobian = make_data(rng, noise, this_psf, this_psf_obs, wcs, dx, dy, g1, g2, stamp_size, scale=scale, flux=flux)

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
        stamp.jacobs.append(this_jacobian.get_galsim_wcs())



    return stamp, obs


def fit_shapes_sp(args, rng, nepoch, scale, flux, prior, stamp):

    # Create (empty) flag arrays
    for iepoch in range(nepoch):
        flag = np.zeros_like(stamp.weights[iepoch])
        stamp.flags.append(flag)

    stamp.bkg_sub = False
    stamp.megacam_flip = True


    # Call ShapePipe ngmix functions 

    #print('MKDEBUG 1 noise sum std', stamp.weights[0].sum(), stamp.weights[0].std())

    # Multiply weights by noise variance, to compensate inverse-variance weighting
    # in sp ngmix
    for iepoch in range(nepoch):
        # sigma_mad estimates input noise ok
        noise = sigma_mad(stamp.gals[iepoch])
        #print(iepoch, noise, args.noise)
        stamp.weights[iepoch] *= noise ** 2

    #print('MKDEBUG 2 noise sum std', stamp.weights[0].sum(), stamp.weights[0].std())

    print("Calling sp do_ngmix_metacal")
    res, obsdict, psf_res = spng.do_ngmix_metacal(stamp, prior, flux, rng)

    # Dividing weights by noise variance to undo above
    #print('MKDEBUG 3 noise sum std', stamp.weights[0].sum(), stamp.weights[0].std())
    for iepoch in range(nepoch):
        noise = sigma_mad(stamp.gals[iepoch])
        stamp.weights[iepoch] /= noise ** 2

    #print('MKDEBUG 4 noise sum std', stamp.weights[0].sum(), stamp.weights[0].std())

    return res, obsdict


def fit_shapes_ngmix(args, rng, nepoch, scale, flux, prior, obs, psf_im):

    # Local ngmix runner
    gal_obs_list = ngmix.observation.ObsList()

    for iepoch in range(nepoch):
        gal_obs_list.append(obs[iepoch])

    # fit the object
    # fit using the levenberg marquards algorithm
    fitter = ngmix.fitting.Fitter(model='gauss', prior=prior)
    # make parameter guesses based on a psf flux and a rough T
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
        rng=rng,
        T=0.25,
        prior=prior,
    )

    # psf fitting with coelliptical gaussians # we need to change the psf fitter to FitGauss
    psf_fitter  = fitter
    # special guesser for coelliptical gaussians
    psf_guesser = ngmix.guessers.TFluxGuesser(
        rng=rng,
        T=0.25,
        prior=prior,
        flux=flux,
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

    # this bootstraps the process, first fitting psfs then the object
    print("Calling ngmix MetacalBootstrapper")                          
    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner,
        psf_runner=psf_runner,
        rng=rng,
        ignore_failed_psf=True,
        psf='gauss'
    )
        #psf='fitgauss',

    res, obsdict = boot.go(gal_obs_list)


    if args.show:
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
        T_range = [-1.0, 1.e3]
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


def make_psf(rng, stamp_size, wcs):
    """Make PSF

    Simulate MoffatPSF.

    """
    psf_noise = 1.0e-6
    psf_fwhm = 0.9

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

    #psf_jacobian = ngmix.DiagonalJacobian(
    #    row=psf_cen[0], col=psf_cen[1], scale=scale,
    #)

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=psf_jacobian,
    )

    return psf, psf_obs, psf_im


def make_data(rng, noise, psf, psf_obs, wcs, dx, dy, g1, g2, stamp_size, scale=1.0, flux=100.0):
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
    gal_hlr = 0.5

    model_func = galsim.Gaussian
    # model_func = galsim.Exponential

    obj0 = model_func(
        half_light_radius=gal_hlr,
        flux=flux,
    ).shear(
        g1=g1,
        g2=g2,
    ).shift(
        dx=dx,
        dy=dy,
    )

    obj = galsim.Convolve(psf, obj0)

    im = obj.drawImage(nx=stamp_size, ny=stamp_size, wcs=wcs).array

    # add noise
    im += rng.normal(scale=noise, size=im.shape)

    cen = (np.array(im.shape) - 1) / 2

    noise = sigma_mad(im)
    wt = im * 0 + 1 / noise ** 2

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

    return obs, wt, jacobian


def get_obj_pars(g1, g2, flux, dx, dy):

    obj_pars = {
        'g1': g1,
        'g2': g2,
        'flux': flux,
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
    print('obs noise std', obsdict['noshear'][0].noise.std())
    print('obs weight std', obsdict['noshear'][0].weight.std())
    print(f"obs weight sum {obsdict['noshear'][0].weight.sum():.2e}")
    print('obs weight[0[0]', obsdict['noshear'][0].weight[0][0])
    print('S/N:', res['noshear']['s2n'])
    print('true flux: %g meas flux: %g +/- %g (99.7%% conf)' % (
        obj_pars['flux'], res['noshear']['flux'], res['noshear']['flux_err']*3,
    ))
    print('true g1: %g meas g1: %g +/- %g (99.7%% conf)' % (
        obj_pars['g1'], res['noshear']['g'][0], res['noshear']['g_err'][0]*3,
    ))
    print('true g2: %g meas g2: %g +/- %g (99.7%% conf)' % (
        obj_pars['g2'], res['noshear']['g'][1], res['noshear']['g_err'][1]*3,
    ))
    print(
        "Delta g / sig g = ("
        + f"{(obj_pars['g1'] - res['noshear']['g'][0]) / res['noshear']['g_err'][0]:.2f},"
        + f" {(obj_pars['g2'] - res['noshear']['g'][1]) / res['noshear']['g_err'][1]:.2f})"
    )
    print()


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for rng')
    parser.add_argument('--show', action='store_true',
                        help='show plot comparing model and data')
    parser.add_argument('--noise', type=float, default=0.01,
                        help='noise for images')
    return parser.parse_args()


if __name__ == '__main__':
    main()
