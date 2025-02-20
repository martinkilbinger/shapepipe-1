# Post-processing

This page shows the required steps for post-processing the results from one or
more `ShapePipe` runs. Post-processing includes tow following tasks:  
1. Merge `ShapePipe` output files and create joint catalogues over a large sky patch area.
   The output contains all required information to create a calibrated shear catalogue via
   _metacalibration_).
2. Create merged PSF and star catalogues in pixel and WCS coordinates.
3. Compute basic PSF diagnostics such as focal-plane residual plots and rho-statistics. 

## 1. Merge `ShapePipe` output files.

The script to peform this task is `create_final_cat.py`. An example call from the base directory, in
which the patch subdirectories `P?` are found, is
```bash
create_final_cat.py -m final_cat_P3.hdf5 -i P3 -p P3/cfis/final_cat.param -o P3/n_tiles_final.txt -P 3 -v
```

This creates the merged file `final_cat_P3.hdf5' from all final `ShapePipe` catalogues found (recursively)
in input directory `P3`. Only columns are merged indicated in the parameter file `P3/cfis/final_cat.param`.
The number of merged tiles is written to `P3/n_tiles_final.txt`.

## 2. Create PSF and star catalogues.

First, project PSF and star quantities measures in pixel coordinates to spherical world (WCS) coordinates, using
`convert_psf_pix2world.py`. For example, from the same base directory as above:
```bash
mkdir star_cat
cd star_cat
convert_psf_pix2world.py -i .. -P 3 -v -p psfex -m merge 
```
converts all star and PSF catalogues found in ../P3/output.

Second, create a single output directory for `ShapePipe` with symbolic links to all projected PSF and star files with `combin_runs.bash`.
For example,
```bash
cd star_cat/P3
combine_runs.bash -p psfex -c psf_conv
```

In the case of UNIONS `v1.4`, only one symbolic link is created.

Third, create the merged PSF and star catalogues by running the `ShapePipe` module `merge_starcat_runner`. For example,
cd star_cat/P3
```bash
export SP_RUN=`pwd`
shapepipe_run -c config_Ms_psfex_conv.ini
```

## 3. Compute basic diagnostics

Basic diagnostics are created with the `mccd_plots_runner` module (for both `psfex` and `MCCD` PSF models). Type
```bash
export SP_RUN=`pwd`
shapepipe_run -c config_Pl_psfex.ini
```

If the main ShapePipe processing happened at the old canfar VM system (e.g. CFIS v0 and v1), go
[here](vos_retrieve.md) for details how to retrieve the ShapePipe output files.

---

The following steps are required for pre-v1.4 runs performed on the canfar VM system.

1. Optional: Split output into sub-samples

   An optional intermediate step is to create directories for sub-samples, for example one directory
   for each patch on the sky. This will create symbolic links to the results `.tgz` files downloaded in
   the previous step. For example, to create the subdir `tiles_W3` with links to result files to `all` for
   those tiles contained in the list `tiles_W3.txt`, do:
   ```bash
    create_sample_results --input_IDs tiles_W3.txt -i . all -o tiles_W3 -v
    ```
    The following steps will then be done in the directory `tiles_W3`.

2. Run PSF diagnostics, create merged catalogue

   Type
   ```bash
   post_proc_sp -p PSF
   ```
   to automatically perform a number of post-processing steps. Choose the PSF model with the option
   `-p psfex|mccd`. In detail, these are (and can also be done individually
   by hand):
   
   1. Analyse psf validation files
   
      ```bash
      combine_runs -t psf -p PSF
      ```
      with options as for `post_proc_sp`.
      This script creates a new combined psf run in the ShapePipe `output` directory, by identifying all psf validation files
      and creating symbolic links. The run log file is updated.

   3. Merge individual psf validation files into one catalogue. Create plots of the PSF and their residuals in the focal plane,
      as a diagnostic of the overall PSF model.
      As a scale-dependend test, which propagates directly to the shear correlation function, the rho statistics are computed,
      see {cite:p}`rowe:10` and {cite:p}`jarvis:16`,
      ```bash
      shapepipe_run -c /path/to/shapepipe/example/cfis/config_MsPl_PSF.ini
      ``` 

   4. Prepare output directory
   
      Create links to all 'final_cat' result files with 
      ```bash
      prepare_tiles_for_final
      ```
      The corresponding output directory that is created is `output/run_sp_combined/make_catalog_runner/output`.
      On success, it contains links to all `final_cat` output catalogues

   5. Merge final output files
   
      Create a single main shape catalog:
      ```bash
      merge_final_cat -i <input_dir> -p <param_file> -v
      ```
      Choose as input directory `input_dir` the output of step C. A default
      parameter file `<param_file>` is `/path/to/shapepipe/example/cfis/final_cat.param`. 
      On success, the file `./final_cat.npy` is created. Depending on the number of
      input tiles, this file can be several tens of Gb large. 
