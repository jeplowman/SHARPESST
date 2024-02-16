import copy, os, resource, numpy as np
from sys import path
import time, numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import lgmres
from sharpesst.solver import sparse_nlmap_solver
from sharpesst.element_functions import (nd_voigt_psf, bin_function, get_2d_cov, get_3d_cov,
                               nd_gaussian_psf, nd_powgaussian_psf, spike_function,
                               flattop_guassian_psf, spice_spectrograph_psf)
from sharpesst.coord_transform import coord_transform, trivialframe
from sharpesst.element_grid import detector_grid, source_grid
from sharpesst.coord_grid import coord_grid
from sharpesst.element_source_responses import element_source_responses as esr
from sharpesst.util import get_mask_errs
from astropy.io import fits

# This is a lower level routine that will do the correction once the matrices are all set up.
# See next two routines for end user usage.
def correct_2d_psf(det_dims, dat_mask, dat, err, amat_remove, amat_restore, reg_fac=.1, 
					chi2_th = 1.0, niter=40, src_dims=None, solver_tol=2.0e-4):
	recon_cube = np.zeros(det_dims)
	if(not(src_dims is None)): recon_cube0 = np.zeros([det_dims[0],src_dims[1],src_dims[2]])
	chi2s = np.zeros(det_dims[0])
	t0=time.time()
	for i in range(0,det_dims[0]):
		flatdat = dat[i,:,:].flatten()
		flaterr = err[i,:,:].flatten()
		reg_guess = amat_remove.transpose()*flatdat

		good_data = dat_mask[i,:,:] == 0
		good_inds_in = np.arange(dat_mask.size,dtype=np.uint64)
		good_inds_in = good_inds_in[np.where(good_data.flatten())]
		good_inds_out = np.arange(good_inds_in.size,dtype=np.uint64)
		good_ind_vals = np.ones(good_inds_in.size,dtype=np.int32)
		good_ind_maskmat = csc_matrix((good_ind_vals,(good_inds_out,good_inds_in)),shape=[good_inds_in.size,good_data.size])
		regmat = diags(np.ones(amat_remove.shape[1]))
		amat_remove_masked = good_ind_maskmat*amat_remove
		src_weights = np.clip(np.sum(amat_remove_masked,axis=0),1,None).A1
		amat_remove_masked = amat_remove_masked*diags(1.0/src_weights)
		solution = sparse_nlmap_solver(good_ind_maskmat*flatdat, good_ind_maskmat*flaterr, amat_remove_masked, conv_chi2=1.0e-4,
										reg_fac=reg_fac, solver_tol=2.0e-4, solver=lgmres, niter=niter, dtype='float64', chi2_th=chi2_th,
										sqrmap=False, flatguess=False, silent=True)#, regmat=regmat)#, guess = np.ones(reg_guess.size))
		recon_cube[i,:,:] = (amat_restore*(solution[0]*src_weights)).reshape(recon_cube[i,:,:].shape)
		if(not(src_dims is None)): recon_cube0[i] = solution[0].reshape(recon_cube0[i].shape)
		chi2s[i] = solution[1]
		print(time.time()-t0,'s; Finished slit index',i,'of',det_dims[0],', chi^2=',chi2s[i])

	if(src_dims is None): output = [recon_cube, chi2s]
	else: output = [recon_cube, recon_cube0, chi2s]
	return output

# Given data and header for a SPICE raster, compute the forward matrices and apply the correction. Best for
# correcting a single raster. See correct_spice_fits below for a routine which will run this on an entire
# fits file. Doing the correction takes 1-2 hours per raster, so an entire fits file can take quite a while. 
# correct_spice_fits can also be used as a guide for how to apply to a single raster, however, just look at
# one iteration of its loop; the Jupyter notebook is a more detailed version of the same...
def get_fwd_matrices(spice_dat, spice_hdr, fwhm_core0_yl, fwhm_wing0_yl, psf_yl_angle, wing_weight, 
		fwhm_symm=None, pxsz_mu=18, arcsperpx=1.1, angsperpx=0.09, spice_bin_facs = np.array([1.0,1.0,1.0]), 
		super_fac=2, yl_core_xpo=2, yl_wing_xpo=1, src_pad=0, det_subgrid_fac=3, det_subgrid_fac_wings=1,
		src_subgrid_fac=2, psf_thold_core=0.0025, psf_thold_wing=0.025, stencil_footprint=[400,1200],
		spice_err_fac=0.2, spice_mask=None, spice_err=None):

	platescale_x = pxsz_mu/arcsperpx # Micron per arcsecond
	platescale_y = pxsz_mu/arcsperpx # Micron per arcsecond
	platescale_l = pxsz_mu/angsperpx # Micron per Angstrom

	if(fwhm_symm is None): fwhm_symm = np.array([2.0,2.0])*pxsz_mu
	core_weight = 1.0-wing_weight

	# Get necessary metadata from header:
	spice_dx, spice_dy, spice_dl = spice_hdr['CDELT1'],spice_hdr['CDELT2'],10*spice_hdr['CDELT3']
	spice_wl0 = 10*spice_hdr['CRVAL3']-spice_dl*spice_hdr['CRPIX3']
	print('Correcting '+spice_hdr['EXTNAME']+'; ref. wavelength='+str(spice_wl0))
	spice_la = spice_wl0+spice_hdr['CDELT3']*10*np.arange(spice_dat.shape[2],dtype=np.float64)

	# Compute detector and PSF properties from header info and configuration inputs:
	# PSF correction is computed with source elements that are half spice pixel size.
	# This is necessary to properly resolve the linewidth, but there's not much
	# useful resolution at that scale, and lots of noise. Plus, it complicates
	# file handling and metadata. The fits output is at SPICE pixel size with
	# a 2 pixel (FWHM) symmetric PSF. This matches the design PSF listed in Table
	# 3 of the SPICE instrument paper (https://doi.org/10.1051/0004-6361/201935574)
	det_scale0 = spice_bin_facs*np.array([spice_dx*platescale_x,1.0*pxsz_mu,spice_dl*platescale_l])
	det_dims0 = np.array(spice_dat.shape)
	src_dims0 = det_dims0*super_fac
	src_scale0 = det_scale0/super_fac
	det_origin0 = np.array([0.0,0.0,spice_la[0]*platescale_l])
	src_origin0 = det_origin0-0.5*det_scale0+0.5*src_scale0 # Correct for pixel center shift with different resolutions 
	[fwhm_core_yl, fwhm_wing_yl] = [fwhm_core0_yl*[platescale_y, platescale_l], fwhm_wing0_yl*[platescale_y,platescale_l]]
	psfsigmas_core_yl = 0.5*fwhm_core_yl/(2.0*np.log(2))**0.5
	psfsigmas_wing_yl = 0.5*fwhm_wing_yl/(2.0*np.log(2))**0.5
	psfsigmas_symm = 0.5*fwhm_symm/(2.0*np.log(2))**0.5
	psfcov_core_yl = get_2d_cov(psfsigmas_core_yl,psf_yl_angle)
	psfcov_wing_yl = get_2d_cov(psfsigmas_wing_yl,psf_yl_angle)
	psfcov_symm = get_2d_cov(psfsigmas_symm,0.0)
	[ipsfcov_core_yl, ipsfcov_wing_yl] = [np.linalg.inv(psfcov_core_yl), np.linalg.inv(psfcov_wing_yl)]
	ipsfcov_symm = np.linalg.inv(psfcov_symm)
	[src_scale_yl, det_scale_yl] = [np.array([det_scale0[1],src_scale0[2]]), det_scale0[1:3]]
	[src_origin_yl, det_origin_yl] = [np.array([det_origin0[1], src_origin0[2]]), det_origin0[1:3]]
	[src_dims_yl, det_dims_yl] = [np.array([det_dims0[1], src_dims0[2]],dtype=np.int32), det_dims0[1:3]]
	[src_yl_fwdtransform, det_yl_fwdtransform] = [np.diag(np.array(src_scale_yl)), np.diag(np.array(det_scale_yl))]

	# Build objects defining coordinate systems, detector grids, and source basis function grids
	# which are used to compute forward matrices:
	[src_yl_frame, det_yl_frame] = [trivialframe(['y','l']), trivialframe(['y','l'])]
	src_yl_coords = coord_grid(src_dims_yl, src_origin_yl, src_yl_fwdtransform, src_yl_frame)
	det_yl_coords = coord_grid(det_dims_yl, det_origin_yl, det_yl_fwdtransform, det_yl_frame)
	source_yl = source_grid(src_yl_coords, None, bin_function, nsubgrid=src_subgrid_fac)

	# The modified Gaussian PSF simply adds a second exponent to the argument of the exponential.
	# Thus, a core_xpo of 2 is a exp(-x**4) profile. This makes the PSF slightly smaller near
	# the peak and much smaller further away; it has much smaller wings and 'shoulders' compared
	# to the core. The width is increased slightly from the nominal (from 0.7 Angstrom to 1.05 Angstrom)
	# along the wavelength axis in order to make the size the same overall. This is evaluated in
	# spice_spectrograph_psf and nd_powgaussian_psf.
	detector_yl_core = detector_grid(det_yl_coords, [ipsfcov_core_yl, yl_core_xpo, 9, 1.0*det_scale0[0]], spice_spectrograph_psf, nsubgrid=det_subgrid_fac, thold=psf_thold_core, footprint=stencil_footprint)
	detector_yl_wing = detector_grid(det_yl_coords, [ipsfcov_wing_yl, yl_wing_xpo, 9, 1.0*det_scale0[0]], spice_spectrograph_psf, nsubgrid=det_subgrid_fac_wings, thold=psf_thold_wing, footprint=stencil_footprint)
	detector_yl_symm = detector_grid(det_yl_coords, [ipsfcov_symm, yl_wing_xpo, 9, 1.0*det_scale0[0]], spice_spectrograph_psf, nsubgrid=det_subgrid_fac, thold=1.0e-3, footprint=stencil_footprint)

	# Compute forward matrices for core, wings, and the symmetric 'ideal' PSF used for the output:
	print('Computing PSF Core:')
	amat_yl_core = esr(source_yl, detector_yl_core, coord_transform)
	print('Computing PSF Wings:')
	amat_yl_wing = esr(source_yl, detector_yl_wing, coord_transform)
	print('Computing Symmetric PSF for output:')
	amat_yl_symm = esr(source_yl, detector_yl_symm, coord_transform)
	amat_yl = core_weight*amat_yl_core + amat_yl_wing*wing_weight/(np.prod(psfsigmas_wing_yl)/np.prod(psfsigmas_core_yl))
	
	metadict = {'det_scale0':det_scale0, 'det_dims0':det_dims0, 'det_origin0':det_origin0, 'src_dims0':src_dims0,
				'src_origin0':src_origin0, 'src_scale0':src_scale0, 'fwhm_core_yl':fwhm_core_yl, 'fwhm_wing_yl':fwhm_wing_yl,
				'psf_yl_angle':psf_yl_angle, 'fwhm_symm':fwhm_symm}
	
	return amat_yl_core, amat_yl_wing, amat_yl_symm, amat_yl, metadict

	
# Given data and header for a SPICE raster, compute the forward matrices and apply the correction. Best for
# correcting a single raster. See correct_spice_fits below for a routine which will run this on an entire
# fits file. Doing the correction takes 1-2 hours per raster, so an entire fits file can take quite a while. 
# correct_spice_fits can also be used as a guide for how to apply to a single raster, however, just look at
# one iteration of its loop; the Jupyter notebook is a more detailed version of the same...
def correct_spice_raster(spice_dat, spice_hdr, fwhm_core0_yl, fwhm_wing0_yl, psf_yl_angle, wing_weight, 
		fwhm_symm=None, pxsz_mu=18, arcsperpx=1.1, angsperpx=0.09, spice_bin_facs = np.array([1.0,1.0,1.0]), 
		super_fac=2, yl_core_xpo=2, yl_wing_xpo=1, src_pad=0, det_subgrid_fac=3, det_subgrid_fac_wings=1,
		src_subgrid_fac=2, psf_thold_core=0.0025, psf_thold_wing=0.025, stencil_footprint=[400,1200],
		spice_err_fac=0.2, spice_mask=None, spice_err=None, chi2_th=1.0):

	platescale_x = pxsz_mu/arcsperpx # Micron per arcsecond
	platescale_y = pxsz_mu/arcsperpx # Micron per arcsecond
	platescale_l = pxsz_mu/angsperpx # Micron per Angstrom

	if(fwhm_symm is None): fwhm_symm = np.array([2.0,2.0])*pxsz_mu*np.max([spice_hdr['NBIN1'],spice_hdr['NBIN2'],spice_hdr['NBIN3']])
	core_weight = 1.0-wing_weight

	# Get necessary metadata from header:
	spice_dx, spice_dy, spice_dl = spice_hdr['CDELT1'],spice_hdr['CDELT2'],10*spice_hdr['CDELT3']
	spice_wl0 = 10*spice_hdr['CRVAL3']-spice_dl*spice_hdr['CRPIX3']
	print('Correcting '+spice_hdr['EXTNAME']+'; ref. wavelength='+str(spice_wl0))
	spice_la = spice_wl0+spice_hdr['CDELT3']*10*np.arange(spice_dat.shape[2],dtype=np.float64)

	# Compute detector and PSF properties from header info and configuration inputs:
	# PSF correction is computed with source elements that are half spice pixel size.
	# This is necessary to properly resolve the linewidth, but there's not much
	# useful resolution at that scale, and lots of noise. Plus, it complicates
	# file handling and metadata. The fits output is at SPICE pixel size with
	# a 2 pixel (FWHM) symmetric PSF. This matches the design PSF listed in Table
	# 3 of the SPICE instrument paper (https://doi.org/10.1051/0004-6361/201935574)
	det_scale0 = spice_bin_facs*np.array([spice_dx*platescale_x,1.0*pxsz_mu,spice_dl*platescale_l])
	det_dims0 = np.array(spice_dat.shape)
	src_dims0 = det_dims0*super_fac
	src_scale0 = det_scale0/super_fac
	det_origin0 = np.array([0.0,0.0,spice_la[0]*platescale_l])
	src_origin0 = det_origin0-0.5*det_scale0+0.5*src_scale0 # Correct for pixel center shift with different resolutions 
	[fwhm_core_yl, fwhm_wing_yl] = [fwhm_core0_yl*[platescale_y, platescale_l], fwhm_wing0_yl*[platescale_y,platescale_l]]
	psfsigmas_core_yl = 0.5*fwhm_core_yl/(2.0*np.log(2))**0.5
	psfsigmas_wing_yl = 0.5*fwhm_wing_yl/(2.0*np.log(2))**0.5
	psfsigmas_symm = 0.5*fwhm_symm/(2.0*np.log(2))**0.5
	psfcov_core_yl = get_2d_cov(psfsigmas_core_yl,psf_yl_angle)
	psfcov_wing_yl = get_2d_cov(psfsigmas_wing_yl,psf_yl_angle)
	psfcov_symm = get_2d_cov(psfsigmas_symm,0.0)
	[ipsfcov_core_yl, ipsfcov_wing_yl] = [np.linalg.inv(psfcov_core_yl), np.linalg.inv(psfcov_wing_yl)]
	ipsfcov_symm = np.linalg.inv(psfcov_symm)
	[src_scale_yl, det_scale_yl] = [np.array([det_scale0[1],src_scale0[2]]), det_scale0[1:3]]
	[src_origin_yl, det_origin_yl] = [np.array([det_origin0[1], src_origin0[2]]), det_origin0[1:3]]
	[src_dims_yl, det_dims_yl] = [np.array([det_dims0[1], src_dims0[2]],dtype=np.int32), det_dims0[1:3]]
	[src_yl_fwdtransform, det_yl_fwdtransform] = [np.diag(np.array(src_scale_yl)), np.diag(np.array(det_scale_yl))]

	# Build objects defining coordinate systems, detector grids, and source basis function grids
	# which are used to compute forward matrices:
	[src_yl_frame, det_yl_frame] = [trivialframe(['y','l']), trivialframe(['y','l'])]
	src_yl_coords = coord_grid(src_dims_yl, src_origin_yl, src_yl_fwdtransform, src_yl_frame)
	det_yl_coords = coord_grid(det_dims_yl, det_origin_yl, det_yl_fwdtransform, det_yl_frame)
	source_yl = source_grid(src_yl_coords, None, bin_function, nsubgrid=src_subgrid_fac)

	# The modified Gaussian PSF simply adds a second exponent to the argument of the exponential.
	# Thus, a core_xpo of 2 is a exp(-x**4) profile. This makes the PSF slightly smaller near
	# the peak and much smaller further away; it has much smaller wings and 'shoulders' compared
	# to the core. The width is increased slightly from the nominal (from 0.7 Angstrom to 1.05 Angstrom)
	# along the wavelength axis in order to make the size the same overall. This is evaluated in
	# spice_spectrograph_psf and nd_powgaussian_psf.
	detector_yl_core = detector_grid(det_yl_coords, [ipsfcov_core_yl, yl_core_xpo, 9, 1.0*det_scale0[0]], 
									spice_spectrograph_psf, nsubgrid=det_subgrid_fac, thold=psf_thold_core, 
									footprint=stencil_footprint)
	detector_yl_wing = detector_grid(det_yl_coords, [ipsfcov_wing_yl, yl_wing_xpo, 9, 1.0*det_scale0[0]], 
									spice_spectrograph_psf, nsubgrid=det_subgrid_fac_wings, thold=psf_thold_wing, 
									footprint=stencil_footprint)
	detector_yl_symm = detector_grid(det_yl_coords, [ipsfcov_symm, yl_wing_xpo, 9, 1.0*det_scale0[0]], 
									spice_spectrograph_psf, nsubgrid=det_subgrid_fac, thold=1.0e-3, 
									footprint=stencil_footprint)

	# Compute forward matrices for core, wings, and the symmetric 'ideal' PSF used for the output:
	print('Computing PSF Core:')
	amat_yl_core = esr(source_yl, detector_yl_core, coord_transform)
	print('Computing PSF Wings:')
	amat_yl_wing = esr(source_yl, detector_yl_wing, coord_transform)
	print('Computing Symmetric PSF for output:')
	amat_yl_symm = esr(source_yl, detector_yl_symm, coord_transform)
	amat_yl = core_weight*amat_yl_core + amat_yl_wing*wing_weight/(np.prod(psfsigmas_wing_yl)/np.prod(psfsigmas_core_yl))

	# Renormalize the forward matrices so that a coefficient of one in the source model produces a unit count
	# summed over the data:
	amat_yl_norm = np.median(np.sum(amat_yl,axis=1).A1)
	amat_yl_symm_norm = np.median(np.sum(amat_yl_symm,axis=1).A1)
	amat_yl /= amat_yl_norm
	amat_yl_symm /= amat_yl_symm_norm

	# Estimate errors and mask off nonusable data points:
	if(spice_mask is None): spice_mask, spice_err = get_mask_errs(spice_dat, spice_err_fac, error_cube=spice_err)
	if(spice_err is None): foo, spice_err =  get_mask_errs(spice_dat, spice_err_fac)

	metadict = {'det_scale0':det_scale0, 'det_dims0':det_dims0, 'det_origin0':det_origin0, 'src_dims0':src_dims0,
				'src_origin0':src_origin0, 'src_scale0':src_scale0, 'fwhm_core_yl':fwhm_core_yl, 'fwhm_wing_yl':fwhm_wing_yl,
				'psf_yl_angle':psf_yl_angle, 'fwhm_symm':fwhm_symm}
				
	# Compute the correction:
	print('Correcting the PSF:')
	spice_corr_dat, spice_corr_chi2s = correct_2d_psf(det_dims0, spice_mask, spice_dat, spice_err, amat_yl, 
													amat_yl_symm, reg_fac=.05, niter=40, chi2_th=chi2_th)
	return spice_corr_dat, spice_corr_chi2s, metadict

# This routine should correct an entire fits file. Takes over 10 hours on my laptop, so not yet tested...
def correct_spice_fits(input_file, output_file, fwhm_core0_yl=np.array([2.0, 1.05]), fwhm_wing0_yl=np.array([10.0, 2.0]),
		psf_yl_angle=-15*np.pi/180, wing_weight=0.165, fwhm_symm=None, pxsz_mu=18, arcsperpx=1.1, angsperpx=0.09, 
		spice_bin_facs = np.array([1.0,1.0,1.0]), super_fac=2, yl_core_xpo=2, yl_wing_xpo=1, src_pad=0, 
		det_subgrid_fac=3, det_subgrid_fac_wings=1,	src_subgrid_fac=2, psf_thold_core=0.0025, psf_thold_wing=0.025,
		stencil_footprint=[400,1200], spice_err_fac=0.2, spice_mask=None, spice_err=None, subtract_min=False):
		
	hdul = fits.open(input_file)
	nraster = len(hdul)-1 # Last entry of hdul appears to contain variable_keywords...
	hdul.close()
	for i in range(0,nraster): # Python always subtracts 1 from the end of a range; convenient until it isn't...
		hdul = fits.open(input_file)
		spice_dat, spice_hdr = copy.deepcopy(hdul[i].data[0]), copy.deepcopy(hdul[i].header)
		hdul.close()
		if(subtract_min):
			specmin = np.nanmin(spice_dat,axis=2)
			nl = spice_dat.shape[2]
			for j in range(0,nl): spice_dat[:,:,j] -= specmin
		spice_dat = spice_dat.transpose([2,1,0]).astype(np.float32)
		spice_corr_dat=spice_dat
		spice_corr_dat, spice_corr_chi2s, metadict = correct_spice_raster(spice_dat,spice_hdr,fwhm_core0_yl,fwhm_wing0_yl,psf_yl_angle,wing_weight)
		hdul = fits.open(input_file)
		hdul[i].data[0][:] = spice_corr_dat.transpose([2,1,0])
		hdul.writeto(output_file)
