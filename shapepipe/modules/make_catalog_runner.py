# -*- coding: utf-8 -*-

"""MAKE CATALOG RUNNER

This module merge different catalog to make the final product.

:Author: Axel Guinot

"""

from shapepipe.modules.module_decorator import module_runner
from shapepipe.pipeline import file_io as io
from sqlitedict import SqliteDict

import numpy as np


def remove_field_name(arr, name):
    """ Remove field name

    Remove a column of a structured array from the given name.

    Parameters
    ----------
    a : numpy.ndarray
        A numpy strucured array.
    name : str
        Name of the field to remove.

    Returns
    -------
    numpy.ndarray
        The structured with the field removed.

    """
    names = list(arr.dtype.names)
    if name in names:
        names.remove(name)
    arr2 = arr[names]
    return arr2


def save_sextractor_data(final_cat_file, sexcat_path, remove_vignet=True):
    """ Save SExtractor data

    Save the SExtractor catalog into the final one.

    Parameters
    ----------
    final_cat_file : io.FITSCatalog
        Final catalog.
    sexcat_path : str
        Path to SExtractor catalog to save.
    remove_vignet : bool
        If True will not save the 'VIGNET' field into the final catalog.

    """

    sexcat_file = io.FITSCatalog(sexcat_path, SEx_catalog=True)
    sexcat_file.open()
    data = np.copy(sexcat_file.get_data())
    if remove_vignet:
        data = remove_field_name(data, 'VIGNET')

    final_cat_file.save_as_fits(data, ext_name='RESULTS')

    sexcat_file.close()


def save_sm_data(final_cat_file, sexcat_sm_path, do_classif=True, star_thresh=0.003, gal_thresh=0.01):
    """ Save spread-model data

    Save the spread-model data into the final catalog.

    Parameters
    ----------
    final_cat_file : io.FITSCatalog
        Final catalog.
    sexcat_sm_path : str
        Path to spread-model catalog to save.
    do_classif : bool
        If True will make a star/galaxy classification. Based on : class = sm + 5/3 * sm_err
    star_thresh : float
        Threshold for star selection. |class| < star_thresh
    gal_thresh : float
        Threshold for galaxy selection. class > gal_thresh

    """

    final_cat_file.open()

    sexcat_sm_file = io.FITSCatalog(sexcat_sm_path, SEx_catalog=True)
    sexcat_sm_file.open()

    sm = np.copy(sexcat_sm_file.get_data()['SPREAD_MODEL'])
    sm_err = np.copy(sexcat_sm_file.get_data()['SPREADERR_MODEL'])

    sexcat_sm_file.close()

    final_cat_file.add_col('SPREAD_MODEL', sm)
    final_cat_file.add_col('SPREADERR_MODEL', sm_err)

    if do_classif:
        obj_flag = np.ones_like(sm, dtype='int16') * 2
        classif = sm + (5. / 3.) * sm_err
        obj_flag[np.where(np.abs(classif) < star_thresh)] = 0
        obj_flag[np.where(classif > gal_thresh)] = 1

        final_cat_file.add_col('SPREAD_CLASS', obj_flag)

    final_cat_file.close()


def save_ngmix_data(final_cat_file, ngmix_cat_path):
    """ Save ngmix data

    Save the ngmix catalog into the final one.

    Parameters
    ----------
    final_cat_file : io.FITSCatalog
        Final catalog.
    ngmix_cat_path : str
        Path to ngmix catalog to save.

    """

    final_cat_file.open()
    obj_id = np.copy(final_cat_file.get_data()['NUMBER'])

    ngmix_cat_file = io.FITSCatalog(ngmix_cat_path)
    ngmix_cat_file.open()
    ngmix_n_epoch = ngmix_cat_file.get_data()['n_epoch_model']
    ngmix_mcal_flags = ngmix_cat_file.get_data()['mcal_flags']
    ngmix_id = ngmix_cat_file.get_data()['id']
    # max_epoch = np.max(ngmix_n_epoch)

    keys = ['1M', '1P', '2M', '2P', 'NOSHEAR']
    output_dict = {'NGMIX_ELL_{}'.format(i): np.ones((len(obj_id), 2)) * -10. for i in keys}
    output_dict = {**output_dict, **{'NGMIX_ELL_ERR_{}'.format(i): np.ones((len(obj_id), 2)) * -10. for i in keys}}
    output_dict = {**output_dict, **{'NGMIX_T_{}'.format(i): np.zeros(len(obj_id)) for i in keys}}
    output_dict = {**output_dict, **{'NGMIX_T_ERR_{}'.format(i): np.ones(len(obj_id)) * 1e30 for i in keys}}
    output_dict = {**output_dict, **{'NGMIX_Tpsf_{}'.format(i): np.zeros(len(obj_id)) for i in keys}}
    output_dict = {**output_dict, **{'NGMIX_SNR_{}'.format(i): np.zeros(len(obj_id)) for i in keys}}
    output_dict = {**output_dict, **{'NGMIX_FLAGS_{}'.format(i): np.ones(len(obj_id), dtype='int16') for i in keys}}
    output_dict['NGMIX_N_EPOCH'] = np.zeros(len(obj_id))
    output_dict['NGMIX_MCAL_FLAGS'] = np.zeros(len(obj_id))
    for i, id_tmp in enumerate(obj_id):
        ind = np.where(id_tmp == ngmix_id)[0]
        if len(ind) > 0:
            for key in keys:
                output_dict['NGMIX_ELL_{}'.format(key)][i][0] = ngmix_cat_file.get_data(key)['g1'][ind[0]]
                output_dict['NGMIX_ELL_{}'.format(key)][i][1] = ngmix_cat_file.get_data(key)['g2'][ind[0]]
                output_dict['NGMIX_ELL_ERR_{}'.format(key)][i][0] = ngmix_cat_file.get_data(key)['g1_err'][ind[0]]
                output_dict['NGMIX_ELL_ERR_{}'.format(key)][i][1] = ngmix_cat_file.get_data(key)['g2_err'][ind[0]]
                output_dict['NGMIX_T_{}'.format(key)][i] = ngmix_cat_file.get_data(key)['T'][ind[0]]
                output_dict['NGMIX_T_ERR_{}'.format(key)][i] = ngmix_cat_file.get_data(key)['T_err'][ind[0]]
                output_dict['NGMIX_Tpsf_{}'.format(key)][i] = ngmix_cat_file.get_data(key)['Tpsf'][ind[0]]
                output_dict['NGMIX_SNR_{}'.format(key)][i] = ngmix_cat_file.get_data(key)['s2n'][ind[0]]
                output_dict['NGMIX_FLAGS_{}'.format(key)][i] = ngmix_cat_file.get_data(key)['flags'][ind[0]]

            output_dict['NGMIX_N_EPOCH'][i] = ngmix_n_epoch[ind[0]]
            output_dict['NGMIX_MCAL_FLAGS'][i] = ngmix_mcal_flags[ind[0]]

    for key in output_dict.keys():
        final_cat_file.add_col(key, output_dict[key])

    final_cat_file.close()
    ngmix_cat_file.close()


def save_galsim_shapes(final_cat_file, galsim_cat_path):
    """ Save ngmix data

    Save the ngmix catalog into the final one.

    Parameters
    ----------
    final_cat_file : io.FITSCatalog
        Final catalog.
    ngmix_cat_path : str
        Path to ngmix catalog to save.

    """

    final_cat_file.open()
    obj_id = np.copy(final_cat_file.get_data()['NUMBER'])

    galsim_cat_file = io.FITSCatalog(galsim_cat_path)
    galsim_cat_file.open()
    galsim_id = galsim_cat_file.get_data()['id']
    # max_epoch = np.max(ngmix_n_epoch)

    output_dict = {'GALSIM_GAL_ELL': np.ones((len(obj_id), 2)) * -10.,
                   'GALSIM_GAL_SIGMA': np.zeros(len(obj_id)),
                   'GALSIM_GAL_FLAG': np.ones(len(obj_id), dtype='int16'),
                   'GALSIM_GAL_ELL_U': np.ones((len(obj_id), 2)) * -10.,
                   'GALSIM_GAL_RES': np.ones(len(obj_id)) * -1.,
                   'GALSIM_PSF_ELL': np.ones((len(obj_id), 2)) * -10.,
                   'GALSIM_PSF_SIGMA': np.zeros(len(obj_id)),
                   'GALSIM_PSF_FLAG': np.ones(len(obj_id), dtype='int16')}
    for i, id_tmp in enumerate(obj_id):
        ind = np.where(id_tmp == galsim_id)[0]
        if len(ind) > 0:
            output_dict['GALSIM_GAL_ELL'][i][0] = galsim_cat_file.get_data()['gal_g1'][ind[0]]
            output_dict['GALSIM_GAL_ELL'][i][1] = galsim_cat_file.get_data()['gal_g2'][ind[0]]
            output_dict['GALSIM_GAL_SIGMA'][i] = galsim_cat_file.get_data()['gal_sigma'][ind[0]]
            output_dict['GALSIM_GAL_FLAG'][i] = galsim_cat_file.get_data()['gal_flag'][ind[0]]
            output_dict['GALSIM_GAL_RES'][i] = galsim_cat_file.get_data()['gal_resolution'][ind[0]]
            output_dict['GALSIM_GAL_ELL_U'][i][0] = galsim_cat_file.get_data()['gal_uncorr_g1'][ind[0]]
            output_dict['GALSIM_GAL_ELL_U'][i][1] = galsim_cat_file.get_data()['gal_uncorr_g2'][ind[0]]
            output_dict['GALSIM_PSF_ELL'][i][0] = galsim_cat_file.get_data()['psf_g1'][ind[0]]
            output_dict['GALSIM_PSF_ELL'][i][1] = galsim_cat_file.get_data()['psf_g2'][ind[0]]
            output_dict['GALSIM_PSF_SIGMA'][i] = galsim_cat_file.get_data()['psf_sigma'][ind[0]]

    for key in output_dict.keys():
        final_cat_file.add_col(key, output_dict[key])

    final_cat_file.close()
    galsim_cat_file.close()



def save_psf_data(final_cat_file, galaxy_psf_path, w_log):
    """ Save PSF data

    Save the PSF catalog into the final one.

    Parameters
    ----------
    final_cat_file : io.FITSCatalog
        Final catalog.
    galaxy_psf_path : str
        Path to the PSF catalog to save.

    """

    final_cat_file.open()
    obj_id = np.copy(final_cat_file.get_data()['NUMBER'])
    #max_epoch = np.max(final_cat_file.get_data()['NGMIX_N_EPOCH'])
    max_epoch = 10

    w_log.info('here')
    # galaxy_psf_cat = np.load(galaxy_psf_path).item()
    galaxy_psf_cat = SqliteDict(galaxy_psf_path)
    w_log.info('there')
    print('a')
    output_dict = {'PSF_ELL_{}'.format(i+1): np.ones((len(obj_id), 2)) * -10. for i in range(max_epoch)}
    output_dict = {**output_dict, **{'PSF_FWHM_{}'.format(i+1): np.zeros(len(obj_id)) for i in range(max_epoch)}}
    output_dict = {**output_dict, **{'PSF_FLAG_{}'.format(i+1): np.ones(len(obj_id), dtype='int16') for i in range(max_epoch)}}
    print('b')
    for i, id_tmp in enumerate(obj_id):
        w_log.info('{}'.format(id_tmp))
        if galaxy_psf_cat[str(id_tmp)] == 'empty':
            continue
        for epoch, key in enumerate(galaxy_psf_cat[str(id_tmp)].keys()):
            if galaxy_psf_cat[str(id_tmp)][key]['SHAPES']['FLAG_PSF_HSM'] != 0:
                continue
            output_dict['PSF_ELL_{}'.format(epoch+1)][i][0] = galaxy_psf_cat[str(id_tmp)][key]['SHAPES']['E1_PSF_HSM']
            output_dict['PSF_ELL_{}'.format(epoch+1)][i][1] = galaxy_psf_cat[str(id_tmp)][key]['SHAPES']['E2_PSF_HSM']
            output_dict['PSF_FWHM_{}'.format(epoch+1)][i] = galaxy_psf_cat[str(id_tmp)][key]['SHAPES']['SIGMA_PSF_HSM'] * 2.355
            output_dict['PSF_FLAG_{}'.format(epoch+1)][i] = galaxy_psf_cat[str(id_tmp)][key]['SHAPES']['FLAG_PSF_HSM']
    print('c')
    for key in output_dict.keys():
        final_cat_file.add_col(key, output_dict[key])
    print('d')
    final_cat_file.close()
    galaxy_psf_cat.close()
    print('e')

@module_runner(input_module=['sextractor_runner', 'spread_model_runner', 'psfexinterp_runner', 'ngmix_runner'],
               version='1.0', file_pattern=['tile_sexcat', 'sexcat_sm', 'galaxy_psf', 'ngmix'],
               file_ext=['.fits', '.fits', '.npy', '.fits'],
               depends=['numpy'])
def make_catalog_runner(input_file_list, output_dir, file_number_string,
                        config, w_log):

    tile_sexcat_path, sexcat_sm_path, galaxy_psf_path, ngmix_cat_path = input_file_list

    do_classif = config.getboolean("MAKE_CATALOG_RUNNER", "SM_DO_CLASSIFICATION")
    star_thresh = config.getfloat("MAKE_CATALOG_RUNNER", "SM_STAR_STRESH")
    gal_thresh = config.getfloat("MAKE_CATALOG_RUNNER", "SM_GAL_THRESH")

    output_name = output_dir + '/final_cat' + file_number_string + '.fits'
    final_cat_file = io.FITSCatalog(output_name, open_mode=io.BaseCatalog.OpenMode.ReadWrite)

    w_log.info('Save SExtractor data')
    save_sextractor_data(final_cat_file, tile_sexcat_path)
    
    w_log.info('Save spread-model data')
    save_sm_data(final_cat_file, sexcat_sm_path, do_classif, star_thresh, gal_thresh)

    w_log.info('Save ngmix data')
    #save_ngmix_data(final_cat_file, ngmix_cat_path)
    save_galsim_shapes(final_cat_file, ngmix_cat_path)

    w_log.info('Save PSF data')
    save_psf_data(final_cat_file, galaxy_psf_path, w_log)

    return None, None
