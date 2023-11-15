import time
import resource
import numpy as np

from scipy.sparse.linalg import LinearOperator

# This operator implements the general linear operator for chi squared
# plus regularization with nonlinear mapping as outlined in Plowman &
# Caspi 2020. This version assumes that the 'a' matrix and regularization matrix
# are instead instances scipy.sparse matrices, which, frustratingly,
# do not behave the same way as a scipy.sparse.linalg.LinearOperator!
# Wrapping scipy.sparse matrices to behave as LinearOperators can result
# in significant performance degradations depending on the size of the problem,
# Therefore, two different versions are maintained.
class nlmap_operator_sparsemat(LinearOperator):
	def setup(self, amat, regmat, drvmat, wgtmat, reg_drvmat, dtype=np.float32, reg_fac=1):
		self.amat = amat
		self.regmat = regmat
		self.drvmat = drvmat
		self.wgtmat = wgtmat
		self.reg_drvmat = reg_drvmat
		self.dtype_internal=dtype
		self.reg_fac = reg_fac
		self.shape=regmat.shape

	def _matvec(self,vec):
		chi2term = self.drvmat*(self.amat.T*(self.wgtmat*(self.amat*(self.drvmat*vec))))
		regterm = self.reg_drvmat*(self.reg_fac*self.regmat*(self.reg_drvmat*vec))
		return (chi2term+regterm).astype(self.dtype_internal)

	def _adjoint(self):
		return self

import time
import resource
import numpy as np

# Subroutine to do the inversion. Uses the log mapping and iteration from Plowman & Caspi 2020 to ensure positivity of solutions.
def sparse_nlmap_solver(data0, errors0, amat0, guess=None, reg_fac=1, func=None, dfunc=None, ifunc=None, regmat=None, silent=False,
						solver=None, sqrmap=False, regfunc=None, dregfunc=None, iregfunc=None, map_reg=False, adapt_lam=True,
						solver_tol = 1.0e-3, niter=40, dtype=np.float32, steps=None, precompute_ata=False, flatguess=True, chi2_th=1.0,
						store_outer_Av=False):
	from scipy.sparse import diags
	from scipy.sparse.linalg import lgmres
	from scipy.linalg import cho_factor, cho_solve
	
	solver_tol = np.dtype(dtype).type(solver_tol)
	zero = np.dtype(dtype).type(0.0)
	pt5 = np.dtype(dtype).type(0.5)
	two = np.dtype(dtype).type(2.0)
	one = np.dtype(dtype).type(1.0)
	conv_chi2 = np.dtype(dtype).type(1.0e-15)

	def idnfunc(s): return s
	def iidnfunc(s): return s
	def didnfunc(s): return one + zero*s
	def expfunc(s): return np.exp(s) #
	def dexpfunc(s): return np.exp(s) #
	def iexpfunc(c): return np.log(c) #
	def sqrfunc(s): return s*s # np.exp(s) #
	def dsqrfunc(s): return two*s # np.exp(s) #
	def isqrfunc(c): return c**pt5 # np.log(c) #
	if(func is None or dfunc is None or ifunc is None): 
		if(sqrmap): [func,dfunc,ifunc] = [sqrfunc,dsqrfunc,isqrfunc]
		else: [func,dfunc,ifunc] = [expfunc,dexpfunc,iexpfunc]
	if(regfunc is None or dregfunc is None or iregfunc is None):
		if(map_reg): [regfunc,dregfunc,iregfunc] = [idnfunc,didnfunc,iidnfunc]
		else: [regfunc,dregfunc,iregfunc] = [func,dfunc,ifunc]
	if(solver is None): solver = lgmres

	flatdat = data0.flatten().astype(dtype)
	flaterrs = errors0.flatten().astype(dtype)
	flaterrs[flaterrs == 0] = (0.05*np.nanmean(flaterrs[flaterrs > 0])).astype(dtype)
	
	guess0 = amat0.T*(np.clip(flatdat,np.min(flaterrs),None))
	guess0dat = amat0*(guess0)
	guess0norm = np.sum(flatdat*guess0dat/flaterrs**2)/np.sum((guess0dat/flaterrs)**2)
	guess0 *= guess0norm
	guess0 = np.clip(guess0,0.005*np.mean(np.abs(guess0)),None).astype(dtype)
	if(guess is None): guess = guess0	
	[ndat, nsrc] = amat0.shape
	guess = ((1+np.zeros(nsrc))*np.mean(flatdat)/np.mean(amat0*(1+np.zeros(nsrc)))).astype(dtype)
	if(flatguess): guess = ((1+np.zeros(nsrc))*np.mean(flatdat)/np.mean(amat0*(1+np.zeros(nsrc)))).astype(dtype)
	svec = ifunc(guess).astype(dtype)

	# Try these step sizes at each step of the iteration. Trial Steps are fast compared to computing 
	# the matrix inverse, so having a significant number of them is not a problem.
	# Step sizes are specified as a fraction of the full distance to the solution found by the sparse
	# matrix solver (lgmres or bicgstab).
	if(steps is None): steps = np.array([0.00, 0.05, 0.15, 0.3, 0.5, 0.67, 0.85],dtype=dtype)
	nsteps = len(steps)
	step_loss = np.zeros(nsteps,dtype=dtype)
	
	if(regmat is None): regmat = diags(one/iregfunc(guess0)**two)
	if(adapt_lam):
		reglam = (np.dot(dregfunc(svec)*(regmat*(regfunc(svec))), dfunc(svec)*(amat0.T*(one/flaterrs)))/
				  np.dot(dregfunc(svec)*(regmat*(regfunc(svec))), dregfunc(svec)*(regmat*(regfunc(svec))))).astype(dtype)
	else: reglam = one
	# Still appears to be some issue with this regularization factor?
	regmat = reg_fac*regmat*reglam #*(ndat/nsrc)
	#reg_fac = reg_fac*reglam
	weights = (1.0/flaterrs**2).astype(dtype) # The weights are the errors...

	if(silent == False):
		print('Overall regularization factor:',reg_fac*reglam)

	#lo_check = linop_check(amat0) and linop_check(regmat)
	#if(lo_check):
	#	nlmo = nlmap_operator(amat0, regmat, diags(dfunc(svec)), diags(weights),
	#				diags(dregfunc(svec)), reg_fac=reg_fac, dtype=dtype)
	#else:
	#	nlmo = nlmap_operator_sparsemat(amat0, regmat, diags(dfunc(svec)), diags(weights),
	#				diags(dregfunc(svec)), reg_fac=reg_fac, dtype=dtype)

	nlmo = nlmap_operator_sparsemat(dtype=dtype,shape=(nsrc,nsrc))
	nlmo.setup(amat0,regmat,diags(dfunc(svec)),diags(weights),diags(dregfunc(svec)),reg_fac=reg_fac)
	
	# --------------------- Now do the iteration:
	tstart = time.time()
	setup_timer = 0
	solver_timer = 0
	stepper_timer = 0
	for i in range(0,niter):
		tsetup = time.time()
		# Setup intermediate matrices for solution:
		dguess = diags(dfunc(svec),dtype=dtype)
		dregguess = diags(dregfunc(svec),dtype=dtype)
		bvec = dguess*amat0.T*(diags(weights)*(flatdat-amat0*(func(svec)-svec*dfunc(svec))))
		bvec -= dregguess*(reg_fac*regmat*(regfunc(svec)-svec*dregfunc(svec)))

		setup_timer += time.time()-tsetup

		tsolver = time.time()
		# Run sparse matrix solver:
		[nlmo.drvmat,nlmo.reg_drvmat] = [dguess,dregguess]
		svec2 = solver(nlmo,bvec.astype(dtype),svec.astype(dtype),store_outer_Av=False,tol=solver_tol.astype(dtype))
		svec2 = svec2[0]
		solver_timer += time.time()-tsolver

		tstepper = time.time()
		deltas = svec2-svec
		if(np.max(np.abs(deltas)) == 0): break # This also means we've converged.
		deltas *= np.clip(np.max(np.abs(deltas)),None,0.5/steps[1])/np.max(np.abs(deltas))

		# Try the step sizes:
		for j in range(0,nsteps):
			stepguess = func(svec+steps[j]*(deltas))
			stepguess_reg = regfunc(svec+steps[j]*(deltas))
			stepresid = (flatdat-amat0*(stepguess))*weights**pt5
			#stepresid = (flatdat-amat0.matvec(stepguess))*weights**pt5
			step_loss[j] = np.dot(stepresid,stepresid)/ndat + np.sum(stepguess_reg.T*(reg_fac*regmat*(stepguess_reg)))/ndat

		best_step = np.argmin((step_loss)[1:nsteps])+1 # First step is zero for comparison purposes...
		#chi20 = np.sum(weights*(flatdat-amat0.matvec(func(svec)))**two)/ndat # step_loss[0]-step_loss[best_step]
		#reg0 = np.sum(regfunc(svec.T)*(reg_fac*regmat.matvec(regfunc(svec))))/ndat
		chi20 = np.sum(weights*(flatdat-amat0*(func(svec)))**two)/ndat # step_loss[0]-step_loss[best_step]
		reg0 = np.sum(regfunc(svec.T)*(reg_fac*regmat*(regfunc(svec))))/ndat
		
		# Update the solution with the step size that has the best Chi squared:
		svec = svec+steps[best_step]*(deltas)
		
		reg1 = np.sum(regfunc(svec.T)*(reg_fac*regmat*(regfunc(svec))))/ndat
		resids = weights*(flatdat-amat0*(func(svec)))**two
		chi21 = np.sum(weights*(flatdat-amat0*(func(svec)))**two)/ndat
		#reg1 = np.sum(regfunc(svec.T)*(reg_fac*regmat.matvec(regfunc(svec))))/ndat
		#resids = weights*(flatdat-amat0.matvec(func(svec)))**two
		#chi21 = np.sum(weights*(flatdat-amat0.matvec(func(svec)))**two)/ndat
		stepper_timer += time.time()-tstepper
		
		if(silent==False):		
			print(round(time.time()-tstart,2),'s i =',i,'chi2 =',round(chi21,2),'step size =',round(steps[best_step],3), 'reg. param. =', round(reg1,2), 'chi2 change =',round(chi20-chi21,5), 'reg. change =',round(reg0-reg1,5))
			print('New combined FOM:',chi21+reg1,'Old combined FOM:',chi20+reg0,'Change:',chi20+reg0-(chi21+reg1))
		if(np.abs(step_loss[0]-step_loss[best_step]) < conv_chi2 or chi21 < chi2_th): break # Finish the iteration if chi squared isn't changing

	return func(svec), chi21, resids
