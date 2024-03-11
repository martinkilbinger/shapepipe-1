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
import galsim
import ngmix

from modopt.math.stats import sigma_mad
from shapepipe.modules.ngmix_package import ngmix as spng
from shapepipe.modules.ngmix_package import postage_stamp as spng_ps



def main():

    args = get_args()
    rng = np.random.RandomState(args.seed)

    stamp = spng_ps.Postage_stamp()

    nepoch = 3

    scale = 0.263
    flux = 100

    for iepoch in range(nepoch):
        psf, psf_obs, psf_im = make_psf(rng=rng, scale=scale)

        obs, obj_pars, wt, obs_im, jacobian = make_data(psf=psf, psf_obs=psf_obs, scale=scale, rng=rng, noise=args.noise)
        flag = np.zeros_like(wt)

        stamp.gals.append(obs_im)
        stamp.weights.append(wt)
        stamp.flags.append(flag)
        stamp.psfs.append(psf_im)
        wcs = jacobian.get_galsim_wcs()
        stamp.jacobs.append(wcs)

    stamp.bkg_sub = False
    stamp.megacam_flip = False

    prior = get_prior(rng=rng, scale=obs.jacobian.scale)

    # Call ShapePipe ngmix function 
    print('MKDEBUG 1 noise sum std', stamp.weights[0].sum(), stamp.weights[0].std())
    for iepoch in range(nepoch):
        noise = sigma_mad(stamp.gals[iepoch])
        #noise = args.noise
        stamp.weights[iepoch] *= noise ** 2
    print('MKDEBUG 2 noise sum std', stamp.weights[0].sum(), stamp.weights[0].std())
    print("Calling sp ngmix MetacalBootstrapper")
    res, obsdict, psf_res = spng.do_ngmix_metacal(stamp, prior, flux, rng)
    print('MKDEBUG 3 noise sum std', stamp.weights[0].sum(), stamp.weights[0].std())
    for iepoch in range(nepoch):
        noise = sigma_mad(stamp.gals[iepoch])
        #noise = args.noise
        stamp.weights[iepoch] /= noise ** 2
    print('MKDEBUG 4 noise sum std', stamp.weights[0].sum(), stamp.weights[0].std())
    print_results(res, obsdict, obj_pars)

    # Local ngmix runner
    gal_obs_list = ngmix.observation.ObsList()
    for i in np.arange(nepoch):
        gal_obs_list.append(obs)
    print(len(gal_obs_list))

    # fit the object to an exponential disk
    # fit using the levenberg marquards algorithm
    fitter = ngmix.fitting.Fitter(model='gauss', prior=prior)
    # make parameter guesses based on a psf flux and a rough T
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
        rng=rng,
        T=0.25,
        prior=prior,
    )

    # psf fitting with coelliptical gaussians
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
    print("Calling fitting_sp MetacalBootstrapper")                          
    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner,
        psf_runner=psf_runner,
        rng=rng,
        ignore_failed_psf=True,
    )

    res, obsdict = boot.go(gal_obs_list)
    print_results(res, obsdict, obj_pars)


    if args.show:
        try:
            import images
        except ImportError:
            from espy import images

        imfit = res.make_image()

        images.compare_images(obs.image, imfit)


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

    g_prior = ngmix.priors.GPriorBA(sigma=0.1, rng=rng)
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


def make_psf(rng, scale=1.0):
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

    psf_im = psf.drawImage(scale=scale).array
    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)

    psf_wt = psf_im*0 + 1.0/psf_noise**2

    psf_cen = (np.array(psf_im.shape)-1.0)/2.0
    psf_jacobian = ngmix.DiagonalJacobian(
        row=psf_cen[0], col=psf_cen[1], scale=scale,
    )

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=psf_jacobian,
    )

    return psf, psf_obs, psf_im


def make_data(rng, noise, psf, psf_obs, scale=1.0, g1=-0.02, g2=+0.05, flux=100.0):
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
    dy, dx = rng.uniform(low=-scale/2, high=scale/2, size=2)

    #obj0 = galsim.Gaussian(
    obj0 = galsim.Exponential(
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

    im = obj.drawImage(scale=scale).array
    im += rng.normal(scale=noise, size=im.shape)

    cen = (np.array(im.shape)-1.0)/2.0

    jacobian = ngmix.DiagonalJacobian(
        row=cen[0] + dy/scale, col=cen[1] + dx/scale, scale=scale,
    )

    noise = sigma_mad(im)
    wt = im*0 + 1.0 / noise ** 2

    noise_img = np.random.randn(*im.shape) * noise

    obs = ngmix.Observation(
        im,
        weight=wt,
        jacobian=jacobian,
        psf=psf_obs,
        noise=noise_img,
    )

    obj_pars = {
        'g1': g1,
        'g2': g2,
        'flux': flux,
        'dx': dx,
        'dy': dy,
    }

    return obs, obj_pars, wt, im, jacobian


def print_results(res, obsdict, obj_pars):

    print("===== results =====")
    print(obj_pars)
    print(res['noshear']['pars'])
    print('obs image shape', obsdict['noshear'][0].image.shape)
    print('obs image sum', obsdict['noshear'][0].image.sum())
    print('obs noise std', obsdict['noshear'][0].noise.std())
    print('obs weight std', obsdict['noshear'][0].weight.std())
    print(f"obs weight sum {obsdict['noshear'][0].weight.sum():.2e}")
    print('obs weight 0', obsdict['noshear'][0].weight[0][0])
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
    print("===== done =====")


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
