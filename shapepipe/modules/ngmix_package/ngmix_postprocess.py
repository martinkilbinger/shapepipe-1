import numpy as np


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

    if wsum == 0:
        raise ZeroDivisionError('Sum of weights = 0, division by zero')

    psf_dict['g_psf'] = g_psf_sum / wsum
    psf_dict['g_psf_err'] = g_psf_err_sum / wsum
    psf_dict['T_psf'] = T_psf_sum / wsum
    psf_dict['T_psf_err'] = T_psf_err_sum / wsum    

    return psf_dict      

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

        # looping through this is weird
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

