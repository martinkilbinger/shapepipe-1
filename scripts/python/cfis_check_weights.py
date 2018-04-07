#!/usr/bin/env python
  

"""Script cfis_check_weights.py

Checking weight files.

:Authors: Martin Kilbinger

:Date: 13/02/2018
"""


# Compability with python2.x for x>6
from __future__ import print_function


import sys
import os
import re
import copy
import glob

import numpy as np
import pylab as plt

from astropy.io import ascii
from astropy.table import Table, Column
from astropy import units
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits

from optparse import OptionParser, IndentedHelpFormatter, OptionGroup
import datetime

import cfis
import stuff



def get_file_list(pattern, verbose=False):
    """Read present working directory and return list of files accounting for input pattern.
    """

    files = glob.glob('*')

    pattern = cfis.get_file_pattern(pattern, 'r', 'weight')

    dst_list = []
    for f in files:
        m = re.findall(pattern, f)
        if len(m) != 0:
            dst_list.append(f)

    if verbose:
        print('Found {} matching files'.format(len(dst_list)))

    return dst_list


def diagnostics(files, verbose=False):

    if verbose:
        print('Checking files...')

    limits = (0.0, 0.5)
    nbin   = 10

    diagn = []

    count = 0
    for f in files:

        hdu   = fits.open(f)
        dat   = hdu[0].data
        date  = hdu[0].header['DATE']

        hdu.close()

        #y, x = np.histogram(dat, bins=nbin, range=limits)  
        # Ratio of numer of close-to-zero pixels to total
        #ratio = float(y[0]) / sum(y)
        ratio = 0

        # Histogram without fixed bin limits
        #y, x = np.histogram(dat)
        #ratio2 = float(y[0]) / sum(y)

        # Number of zero pixels where image pixels are zero

        # Get image name from weight file name
        img_name = re.sub('weight\.', '', f) 

        hdu_img  = fits.open(img_name)
        dat_img  = hdu_img[0].data
        hdu_img.close()

        # Indices where images has zero pixel values
        idx_img_zero = np.where(dat_img == 0)

        # Count non-zero weight pixels at these indices
        ratio2 = np.count_nonzero(dat[idx_img_zero])

        # MKDEBUG
        #y = 8500
        #x = 4000
        #print('Image[y={}][x={}] = {}'.format(y, x, dat_img[y][x]))
        #print('Weight[y={}][x={}] = {}'.format(y, x, dat[y][x]))

        # Number of exact zero pixels
        ratio3 = 1.0 - float(np.count_nonzero(dat)) / float(dat.size)


        # Diagnostic
        ok    = ratio > 0.001
        ok2   = ratio2 == 0
        ok3   = ratio3 > 0.02



        diagn.append([f, ratio, ratio2, ratio3, ok, ok2, ok3, date])

        if verbose:
            print_diagn(sys.stdout, f, ratio, ratio2, ratio3, str(ok), str(ok2), str(ok3), date)

        count += 1

    # Sort according to date in header
    if len(diagn) > 1:
        diagn_s = sorted(diagn, key=lambda x: datetime.datetime.strptime(x[-1], '%Y-%m-%dT%H:%M:%S'))
    else:
        diagn_s = diagn

    return diagn_s



def print_diagn(f, name, diagn, diagn2, diagn3, pf, pf2, pf3, date):

    print('{:30s} {:13.5f} {:13d} {:13.5f} {:5s} {:5s} {:5s} {:20s}'.format(name, diagn, diagn2, diagn3, pf, pf2, pf3, date), file=f)



def output(diagn, output, verbose=False):

    if output == 'stdout':
        f = sys.stdout
    else:
        f = open(output, 'w')

    if verbose:
        print('Writing diagnostics to {}'.format(output))

    print('{:30s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:20s}'.
            format('# name', 'na      ', '#{w!=0&&img=0)', 'ratio[0/all]', 'pass/fail', 'pass/fail_2', 'pass/fail_3', 'date'), file=f)
    for d in diagn:
        print_diagn(f, d[0], d[1], d[2], d[3], str(d[4]), str(d[5]), str(d[6]), d[7])



def params_default():
    """Set default parameter values.

    Parameters
    ----------
    None

    Returns
    -------
    p_def: class tuff.param
        parameter values
    """

    p_def = stuff.param(
        output   = 'stdout',
        pattern  = '.',
    )

    return p_def



def parse_options(p_def):
    """Parse command line options.

    Parameters
    ----------
    p_def: class tuff.param
        parameter values

    Returns
    -------
    options: tuple
        Command line options
    args: string
        Command line string
    """

    usage  = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage, formatter=stuff.IndentedHelpFormatterWithNL())

    parser.add_option('-o', '--output', dest='output', type='string', default=p_def.output,
        help='output file name, default=stdout')

    parser.add_option('-p', '--pattern', dest='pattern', type='string', default=p_def.pattern,
         help='file pattern to match, e.g.~\'^21\d{5}p\', default=none (=all match)')

    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', help='verbose output')

    options, args = parser.parse_args()

    return options, args



def check_options(options):
    """Check command line options.

    Parameters
    ----------
    options: tuple
        Command line options

    Returns
    -------
    erg: bool
        Result of option check. False if invalid option value.
    """

    see_help = 'See option \'-h\' for help.'

    return True



def update_param(p_def, options):
    """Return default parameter, updated and complemented according to options.
    
    Parameters
    ----------
    p_def:  class mkstuff.param
        parameter values
    optiosn: tuple
        command line options
    
    Returns
    -------
    param: class mkstuff.param
        updated paramter values
    """

    param = copy.copy(p_def)

    # Update keys in param according to options values
    for key in vars(param):
        if key in vars(options):
            setattr(param, key, getattr(options, key))

    # Add remaining keys from options to param
    for key in vars(options):
        if not key in vars(param):
            setattr(param, key, getattr(options, key))

    return param



def main(argv=None):
    """Main program.
    """


    # Set default parameters
    p_def = params_default()

    # Command line options
    options, args = parse_options(p_def)
    # Without option parsing, this would be: args = argv[1:]

    if check_options(options) is False:
        return 1

    param = update_param(p_def, options)


    # Save calling command
    stuff.log_command(argv)
    if param.verbose:
        stuff.log_command(argv, name='sys.stderr')


    if param.verbose is True:
        print('Start of program {}'.format(os.path.basename(argv[0])))


    ### Start main program ###

    files = get_file_list(param.pattern, verbose=param.verbose)

    diagn = diagnostics(files, verbose=param.verbose)

    output(diagn, param.output, verbose=param.verbose)

    ### End main program

    if param.verbose is True:
        print('End of program {}'.format(os.path.basename(argv[0])))

    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv))

