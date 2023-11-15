import numpy as np
import astropy.units as u
from scipy.optimize import least_squares

def fit_spice_lines(datacube, errorcube, waves, dat_mask, cen0, sig0, nmin=10, noisefloor=None, cenbound_fac = 0.2, widbound_fac=0.25, verbose=True):
	nx,ny = datacube.shape[0:2]
	fit_amps = np.zeros([nx,ny])
	fit_cens = np.zeros([nx,ny])
	fit_sigs = np.zeros([nx,ny])
	fit_cont = np.zeros([nx,ny])
	fit_chi2 = np.zeros([nx,ny])
	if(noisefloor is None): noisefloor = np.min(errorcube[np.logical_not(dat_mask)])
	dw = (waves[-1]-waves[0])
	n_status = 20 # Print this many status messages
	for i in range(0,nx):
		for j in range(0,ny):
			data = datacube[i,j,:]
			errs = errorcube[i,j,:]
			mask = dat_mask[i,j,:]
			if(np.sum(np.logical_not(mask)) >= nmin):
				dat = data[np.logical_not(mask)]#.value
				err = errs[np.logical_not(mask)]#.value
				wvl = waves[np.logical_not(mask)]
				cont = np.min(dat)
				wav_norm = np.trapz(dat-cont,x=wvl)
				amp = np.max(dat)-cont #np.trapz(dat-cont,x=wvl)
				cen = wvl[np.argmax(dat)] # np.clip(wvl[np.argmax(dat)], cen0-0.2*dw, cen0+0.2*dw)
				#cen = np.clip(np.trapz(wvl*(dat-cont),x=wvl)/wav_norm,cen0-sig0,cen0+sig0)
				sig = sig0 # np.clip((np.trapz(wvl**2*(dat-cont),x=wvl)/wav_norm-cen**2),(0.25*sig0)**2,(2.5*sig0)**2)**0.5
				bounds = (np.array([0,cen0-cenbound_fac*dw,waves[1]-waves[0],np.min(dat)]),
						np.array([np.max(dat),cen0+cenbound_fac*dw,widbound_fac*dw,np.max(dat)]))
				guess = np.clip(np.array([amp,cen,sig,noisefloor]),bounds[0],bounds[1])
				evaluator = resid_evaluator(wvl,dat,err,single_gaussian_profile)
				solution = least_squares(evaluator.evaluate,guess,bounds=bounds, x_scale=[.1,0.001,0.001,.1])#,method='dogbox')

                ## Add fits to output arrays:
				fit_amps[i,j],fit_cens[i,j],fit_sigs[i,j],fit_cont[i,j] = solution['x']
				fit_chi2[i,j] = solution['cost']
		# Print status:
		if(verbose and (np.mod(i,np.round(nx/n_status).astype(np.int32))==0 or i==nx-1)): print(i,'of',nx, np.mean(np.isnan(fit_amps)))
	return {'centers':fit_cens,'amplitudes':fit_amps,'sigmas':fit_sigs,'continuum':fit_cont,'chi2':fit_chi2}
    
def single_gaussian_profile(waves,parms):
	profile = parms[0]*np.exp(-0.5*((waves-parms[1])/parms[2])**2)
	for i in range(0,len(parms)-3): profile += parms[i+3]*waves**i
	return profile

class resid_evaluator(object):
	def __init__(self,waves,dat,err,profile):
		self.waves,self.dat,self.err,self.profile = waves,dat,err,profile		
	def evaluate(self,parms):
		return (self.dat-self.profile(self.waves,parms))/self.err

def get_overall_center(waves_in, data, wpeak=10):
    overall_data = np.nanmean(data,(0,1))
    index_peak = np.nanargmax(overall_data)
    overall_data = overall_data[index_peak-wpeak:index_peak+wpeak]
    waves = waves_in[index_peak-wpeak:index_peak+wpeak]
    index_peak = np.nanargmax(overall_data)
    errs = overall_data[index_peak]*0.01+0.0*overall_data
    mask = np.isfinite(overall_data) == False
    print(overall_data.shape)
    sigma_guess = np.trapz(overall_data[mask==False],x=waves[mask==False])/overall_data[index_peak]/(2*np.pi)**0.5
    profile = fit_spice_lines(np.expand_dims(overall_data,(0,1)), np.expand_dims(errs,(0,1)), waves, np.expand_dims(mask,(0,1)), waves[index_peak], sigma_guess)
    return profile['centers'][0][0]