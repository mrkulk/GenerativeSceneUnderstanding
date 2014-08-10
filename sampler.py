#Tejas D Kulkarni (tejask@mit.edu | tejasdkulkarni@gmail.com)
import numpy as np
import pdb,copy,time, pickle
from distributions import *
import bezier,distributions,scipy.stats
import skimage.color,scipy.misc
from skimage import filter
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage

RJMCMC = True
parallel = True
DELAYED_REJECTION = True
DPMIXTURE = True
BIRTH_DR_TOTAL_INTERMEDIATE_PROPOSAL = 10
MOTION = False
INF_ALL_DIM_X = False

class Sampler():
	def __init__(self):
		self.alpha = 0.1 #CRP hyper
		self.gpyramid = False
		self.gpr_layers = [4,8,10] #number of gpr layers
		self.move_probabilities = [0.90,0.05,0.05] #update,birth,death
		self.surfaces = dict()
		# self.surface = bsurface
		# self.patch_dict = dict()
		# surface.nPts = nPts
		# surface.K = 2
		#parameters
		# self.params = {
		# #hypers
		# 'lambda_x':10,
		# 'lambda_c':10,
		# 'X_var':0.2,
		# 'mu_l':-2,
		# 'mu_u':0,
		# 'X_l':0,
		# 'X_u':1,
		# 'pflip':0.01,
		# #latents
		# 'Z': np.zeros((nPts,nPts)),
		# 'theta_c': np.zeros((surface.K,3*2)),#rgb
		# 'C':np.zeros((nPts,nPts,3)),
		# # 'mu_p': np.zeros((self.divisions,self.divisions)),
		# 'X':np.zeros((nPts,nPts,3)),
		# 'mix':np.zeros(surface.K)
		# }
		# self.params['X']=copy.deepcopy(control_pts)
		self.pflip = 0.01

	def sample_prior_object(self,surface, move='update'):
		ret_ll = 0

		if move != 'birth': #we initialize block with existing one for faster convergence
			if DPMIXTURE:
				ll, surface = self.sample_CRP(surface); ret_ll+=ll
				ll, surface = self.sample_thetac_CRP(surface); ret_ll += ll
				ll, surface = self.sample_C_CRP(surface); ret_ll += ll
			else:
				ll, surface = self.sample_mixing(surface); ret_ll+=ll
				ll, surface = self.sample_Z(surface); ret_ll+=ll
				ll, surface = self.sample_thetac(surface); ret_ll+=ll
				ll, surface = self.sample_C(surface); ret_ll+=ll

		#if move != 'birth': #we initialize block with existing one for faster convergence
		ll, surface = self.sample_X(surface); ret_ll+=ll
		
		ll, surface = self.sample_tx(surface); ret_ll+=ll
		ll, surface = self.sample_ty(surface); ret_ll+=ll
		ll, surface = self.sample_tz(surface); ret_ll+=ll
		
		ll, surface = self.sample_rx(surface); ret_ll+=ll
		ll, surface = self.sample_ry(surface); ret_ll+=ll
		ll, surface = self.sample_rz(surface); ret_ll+=ll
		
		ll, surface = self.sample_sx(surface); ret_ll+=ll
		ll, surface = self.sample_sy(surface); ret_ll+=ll
		# ll, surface = self.sample_sz(surface); ret_ll+=ll			
		
		return ret_ll, surface


	def sample_prior(self):
		ret_ll = 0
		self.surfaces[0] = bezier.BezierPatch()
		ll, self.surfaces[0] = self.sample_prior_object(self.surfaces[0]); ret_ll+=ll
		ll,_ = self.logl_compute(oid=None, new_surface=None, cur_surfaces=self.surfaces); ret_ll+=ll
		return ret_ll

	def sample_tx(self, surface, sample=True):
		nparams = surface.params
		a=-0.5;b=0.5
		# a=0;b=0.1
		if sample: nparams['tx']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['tx'])
		return ll, surface

	def sample_ty(self, surface, sample=True):
		nparams = surface.params
		a=-0.5;b=0.5
		# a=0;b=0.1
		if sample: nparams['ty']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['ty'])
		return ll, surface

	def sample_tz(self, surface, sample=True):
		nparams = surface.params
		a=-0.5;b=0.5
		if sample: nparams['tz']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['tz'])
		return ll, surface

	def sample_rx(self, surface, sample=True):
		nparams = surface.params
		a=-30;b=30
		if sample: nparams['rx']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['rx'])
		return ll, surface

	def sample_ry(self, surface, sample=True):
		nparams = surface.params
		a=-30;b=30
		if sample: nparams['ry']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['ry'])
		return ll, surface

	def sample_rz(self, surface, sample=True):
		nparams = surface.params
		a=-30;b=30
		if sample: nparams['rz']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['rz'])
		return ll, surface

	def sample_sx(self, surface, sample=True):
		nparams = surface.params
		if RJMCMC:
			a=0.2;b=0.7
		else:
			a=1.0;b=1.5
		if sample: nparams['sx']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['sx'])
		return ll, surface

	def sample_sy(self, surface, sample=True):
		nparams = surface.params
		if RJMCMC:
			a=0.2;b=0.7
		else:
			a=1.0;b=1.5
		if sample: nparams['sy']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['sy'])
		return ll, surface

	# def sample_sz(self, surface, sample=True):
	# 	nparams = surface.params
	# 	a=0.6;b=0.8
	# 	if sample: nparams['sz']=np.random.uniform(a,b)
	# 	ll = distributions.uniform_logpdf(a,b,nparams['sz'])
	# 	return ll, surface

	def sample_pflip(self, sample=True):
		a=1;b=2
		if sample:	self.pflip = np.random.beta(a,b)
		ll = scipy.stats.beta.logpdf(self.pflip,a,b)
		return ll, self.pflip

	def sample_mixing(self, surface, sample=True):
		nparams = surface.params
		if sample:	nparams['mix'] = np.random.dirichlet(2*np.ones(surface.K),1)[0]
		ll = distributions.dirichlet_logpdf(2*np.ones(surface.K),nparams['mix'])
		return ll, surface


	def sample_mixing(self, surface, sample=True):
		if sample:	self.rot_angle = np.random.dirichlet(2*np.ones(surface.K),1)[0]
		ll = distributions.dirichlet_logpdf(2*np.ones(surface.K),nparams['mix'])
		return ll, surface

	def sample_Z(self, surface, sample=True):
		nparams = surface.params
		if sample:
			z_tmp = np.where(np.random.multinomial(1,nparams['mix'],surface.nPts*surface.nPts)==1)[1]
			nparams['Z'] = np.reshape(z_tmp,(surface.nPts,surface.nPts))
		else:
			z_tmp = nparams['Z'].flatten()
		ll = distributions.multinomial_logpdf(z_tmp, nparams['mix'],surface.K)

		#MRF potentials
		for ii in range(surface.nPts):
			for jj in range(surface.nPts):
				if jj > 0:	ll+=-nparams['lambda_c']*(nparams['Z'][jj,ii]-nparams['Z'][jj-1,ii])**2
				if jj < surface.nPts-1:	ll+=-nparams['lambda_c']*(nparams['Z'][jj,ii]-nparams['Z'][jj+1,ii])**2
				if ii > 0: ll+=-nparams['lambda_c']*(nparams['Z'][jj,ii]-nparams['Z'][jj,ii-1])**2
				if ii < surface.nPts-1: ll+=-nparams['lambda_c']*(nparams['Z'][jj,ii]-nparams['Z'][jj,ii+1])**2

				if ii > 0 and jj>0: ll+=-nparams['lambda_c']*(nparams['Z'][ii,jj]-nparams['Z'][ii-1,jj-1])**2
				if ii < surface.nPts-1 and jj>0: ll+=-nparams['lambda_c']*(nparams['Z'][ii,jj]-nparams['Z'][ii+1,jj-1])**2
				if ii < surface.nPts-1 and jj<surface.nPts-1: ll+=-nparams['lambda_c']*(nparams['Z'][ii,jj]-nparams['Z'][ii+1,jj+1])**2
				if ii > 0 and jj<surface.nPts-1: ll+=-nparams['lambda_c']*(nparams['Z'][ii,jj]-nparams['Z'][ii-1,jj+1])**2

		return ll, surface

	def sample_CRP(self, surface, sample=True):
		nparams = surface.params
		if sample:
			N = surface.nPts*surface.nPts
			crp = distributions.CRP(N, self.alpha)
			z_tmp = crp.sample_prior()
			nparams['Z'] = np.reshape(z_tmp,(surface.nPts,surface.nPts))
			nparams['Z_CRPOBJ'] = crp
		else:
			z_tmp = nparams['Z'].flatten()
			crp = nparams['Z_CRPOBJ']
			crp.Z = z_tmp

		ll = crp.logpdf()

		#MRF potentials
		for ii in range(surface.nPts):
			for jj in range(surface.nPts):
				if jj > 0:	ll+=-nparams['lambda_c']*(nparams['Z'][jj,ii]-nparams['Z'][jj-1,ii])**2
				if jj < surface.nPts-1:	ll+=-nparams['lambda_c']*(nparams['Z'][jj,ii]-nparams['Z'][jj+1,ii])**2
				if ii > 0: ll+=-nparams['lambda_c']*(nparams['Z'][jj,ii]-nparams['Z'][jj,ii-1])**2
				if ii < surface.nPts-1: ll+=-nparams['lambda_c']*(nparams['Z'][jj,ii]-nparams['Z'][jj,ii+1])**2

				if ii > 0 and jj>0: ll+=-nparams['lambda_c']*(nparams['Z'][ii,jj]-nparams['Z'][ii-1,jj-1])**2
				if ii < surface.nPts-1 and jj>0: ll+=-nparams['lambda_c']*(nparams['Z'][ii,jj]-nparams['Z'][ii+1,jj-1])**2
				if ii < surface.nPts-1 and jj<surface.nPts-1: ll+=-nparams['lambda_c']*(nparams['Z'][ii,jj]-nparams['Z'][ii+1,jj+1])**2
				if ii > 0 and jj<surface.nPts-1: ll+=-nparams['lambda_c']*(nparams['Z'][ii,jj]-nparams['Z'][ii-1,jj+1])**2
		return ll, surface


	def sample_thetac(self, surface, sample=True):
		ll=0
		a=surface.params['thetac_a']; b=surface.params['thetac_b']
		nparams = surface.params
		for k in range(surface.K):
			if sample:
				#nparams['theta_c'][k,0]=1;nparams['theta_c'][k,2]=1;nparams['theta_c'][k,4]=1;
				#nparams['theta_c'][k,1]=np.random.uniform(a,b);nparams['theta_c'][k,3]=np.random.uniform(a,b);nparams['theta_c'][k,5]=np.random.uniform(a,b)		
				nparams['theta_c'][k,0]=np.random.gamma(a,b);nparams['theta_c'][k,2]=np.random.gamma(a,b);nparams['theta_c'][k,4]=np.random.gamma(a,b)		
				nparams['theta_c'][k,1]=np.random.gamma(a,b);nparams['theta_c'][k,3]=np.random.gamma(a,b);nparams['theta_c'][k,5]=np.random.gamma(a,b)		
				
			# ll+=distributions.uniform_logpdf(a,b,nparams['theta_c'][k,1])
			# ll+=distributions.uniform_logpdf(a,b,nparams['theta_c'][k,3])
			# ll+=distributions.uniform_logpdf(a,b,nparams['theta_c'][k,5])

			ll+=distributions.gamma_logpdf(nparams['theta_c'][k,0],a,b)
			ll+=distributions.gamma_logpdf(nparams['theta_c'][k,2],a,b)
			ll+=distributions.gamma_logpdf(nparams['theta_c'][k,4],a,b)

			ll+=distributions.gamma_logpdf(nparams['theta_c'][k,1],a,b)
			ll+=distributions.gamma_logpdf(nparams['theta_c'][k,3],a,b)
			ll+=distributions.gamma_logpdf(nparams['theta_c'][k,5],a,b)

		return ll, surface

	def sample_thetac_CRP(self, surface, sample=True):
		ll = 0
		a=surface.params['thetac_a']; b=surface.params['thetac_b']
		nparams = surface.params
		allcids = nparams['Z_CRPOBJ'].countvec.keys()

		#remove stale params
		for cid in nparams['theta_c_crp'].keys():
			if nparams['Z_CRPOBJ'].countvec.has_key(cid) == False:
				del nparams['theta_c_crp'][cid]

		for cid in allcids:
			if sample or nparams['theta_c_crp'].has_key(cid) == False:
				if nparams['theta_c_crp'].has_key(cid) == False:
					nparams['theta_c_crp'][cid]=np.zeros(6)
				#nparams['theta_c_crp'][cid][0]=1;nparams['theta_c_crp'][cid][2]=1;nparams['theta_c_crp'][cid][4]=1;
				#nparams['theta_c_crp'][cid][1]=np.random.uniform(a,b);nparams['theta_c_crp'][cid][3]=np.random.uniform(a,b);nparams['theta_c_crp'][cid][5]=np.random.uniform(a,b)		
				nparams['theta_c_crp'][cid][0]=np.random.gamma(a,b);nparams['theta_c_crp'][cid][2]=np.random.gamma(a,b);nparams['theta_c_crp'][cid][4]=np.random.gamma(a,b)				
				nparams['theta_c_crp'][cid][1]=np.random.gamma(a,b);nparams['theta_c_crp'][cid][3]=np.random.gamma(a,b);nparams['theta_c_crp'][cid][5]=np.random.gamma(a,b)		
				
			# ll+=distributions.uniform_logpdf(a,b,nparams['theta_c_crp'][cid][1])
			# ll+=distributions.uniform_logpdf(a,b,nparams['theta_c_crp'][cid][3])
			# ll+=distributions.uniform_logpdf(a,b,nparams['theta_c_crp'][cid][5])

			ll+=distributions.gamma_logpdf(nparams['theta_c_crp'][cid][0],a,b)
			ll+=distributions.gamma_logpdf(nparams['theta_c_crp'][cid][2],a,b)
			ll+=distributions.gamma_logpdf(nparams['theta_c_crp'][cid][4],a,b)

			ll+=distributions.gamma_logpdf(nparams['theta_c_crp'][cid][1],a,b)
			ll+=distributions.gamma_logpdf(nparams['theta_c_crp'][cid][3],a,b)
			ll+=distributions.gamma_logpdf(nparams['theta_c_crp'][cid][5],a,b)
			
		return ll, surface


	# def sample_mu(self, nparams, sample=True):
	# 	if sample:	nparams['mu_p'] = np.random.uniform(nparams['mu_l'],nparams['mu_u'],(self.divisions,self.divisions))
	# 	ll = np.sum(scipy.stats.uniform.logpdf(nparams['mu_l'],nparams['mu_u'],nparams['mu_p']))

	# 	#MRF potentials
	# 	ht=np.shape(nparams['mu_p'])[1]
	# 	wd=np.shape(nparams['mu_p'])[0]
	# 	for ii in range(ht):
	# 		for jj in range(wd):
	# 			if jj > 0:	ll+=(nparams['mu_p'][jj,ii]-nparams['mu_p'][jj-1,ii])**2
	# 			if jj < wd-1:	ll+=(nparams['mu_p'][jj,ii]-nparams['mu_p'][jj+1,ii])**2
	# 			if ii > 0: ll+=(nparams['mu_p'][jj,ii]-nparams['mu_p'][jj,ii-1])**2
	# 			if ii < ht-1: ll+=(nparams['mu_p'][jj,ii]-nparams['mu_p'][jj,ii+1])**2

	# 	return ll, nparams

	# def sample_X(self, nparams, sample=True):
	# 	ll=0
	# 	for jj in range(self.divisions):
	# 		for ii in range(self.divisions):
	# 			patch = self.patch_dict[ii,jj]	
	# 			xinds = patch['xr']
	# 			yinds = patch['yr']
	# 			for yy in range(yinds[0],yinds[1]):
	# 				for xx in range(xinds[0],xinds[1]):
	# 					if sample:	nparams['X'][xx,yy,2]=np.random.normal(nparams['mu_p'][ii,jj],nparams['X_var'])
	# 					ll+= distributions.normal_logpdf(nparams['mu_p'][ii,jj],nparams['X_var'],nparams['X'][xx,yy,2])
	# 	return ll, nparams

	def sample_X(self, surface, sample=True):
		nparams = surface.params
		ll=0
		for jj in range(surface.nPts):
			for ii in range(surface.nPts):
				if sample:	
					if INF_ALL_DIM_X:
						nparams['X'][ii,jj,0]=np.random.normal(surface.controlPoints[ii,jj,0],nparams['X_var'])
						nparams['X'][ii,jj,1]=np.random.normal(surface.controlPoints[ii,jj,1],nparams['X_var'])		
					nparams['X'][ii,jj,2]=np.random.uniform(nparams['X_l'],nparams['X_u'])
				
				if INF_ALL_DIM_X:
					for dd in range(2):	ll+=distributions.normal_logpdf(surface.controlPoints[ii,jj,dd],nparams['X_var'], nparams['X'][ii,jj,dd])

				ll+= distributions.uniform_logpdf(nparams['X_l'],nparams['X_u'],nparams['X'][ii,jj,2])
				
				#MRF costs
				if INF_ALL_DIM_X:
					DIMS = [0,1,2]
				else:
					DIMS=[2]
				for dd in DIMS:
					if ii > 0:	ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,dd]-nparams['X'][ii-1,jj,dd])**2
					if ii < surface.nPts-1:	ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,dd]-nparams['X'][ii+1,jj,dd])**2
					if jj > 0: ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,dd]-nparams['X'][ii,jj-1,dd])**2
					if jj < surface.nPts-1: ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,dd]-nparams['X'][ii,jj+1,dd])**2

					if ii > 0 and jj>0: ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,dd]-nparams['X'][ii-1,jj-1,dd])**2
					if ii < surface.nPts-1 and jj>0: ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,dd]-nparams['X'][ii+1,jj-1,dd])**2
					if ii < surface.nPts-1 and jj<surface.nPts-1: ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,dd]-nparams['X'][ii+1,jj+1,dd])**2
					if ii > 0 and jj<surface.nPts-1: ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,dd]-nparams['X'][ii-1,jj+1,dd])**2

		return ll, surface


	def sample_C(self, surface, sample=True):
		nparams = surface.params
		ll=0
		for k in range(surface.K):
			indxs = np.where(nparams['Z']==k)
			xindxs = indxs[0]; yindxs = indxs[1]
			if sample:	nparams['C'][xindxs,yindxs,0]=np.random.beta(nparams['theta_c'][k,0],nparams['theta_c'][k,1],size=len(nparams['C'][xindxs,yindxs,0]))
			ll += np.sum(distributions.beta_logpdf(nparams['C'][xindxs,yindxs,0],nparams['theta_c'][k,0],nparams['theta_c'][k,1]))
			# if np.isinf(ll):
				# pdb.set_trace()
			if sample:	nparams['C'][xindxs,yindxs,1]=np.random.beta(nparams['theta_c'][k,2],nparams['theta_c'][k,3],size=len(nparams['C'][xindxs,yindxs,0]))
			ll += np.sum(distributions.beta_logpdf(nparams['C'][xindxs,yindxs,1],nparams['theta_c'][k,2],nparams['theta_c'][k,3]))		
			# if np.isinf(ll):
				# pdb.set_trace()
			if sample:	nparams['C'][xindxs,yindxs,2]=np.random.beta(nparams['theta_c'][k,4],nparams['theta_c'][k,5],size=len(nparams['C'][xindxs,yindxs,0]))
			ll += np.sum(distributions.beta_logpdf(nparams['C'][xindxs,yindxs,2],nparams['theta_c'][k,4],nparams['theta_c'][k,5]))
			# if np.isinf(ll):
				# pdb.set_trace()
		return ll, surface


	def sample_C_CRP(self, surface, sample=True):
		nparams = surface.params
		ll=0
		for k in surface.params['Z_CRPOBJ'].countvec.keys():
			indxs = np.where(nparams['Z']==k)
			xindxs = indxs[0]; yindxs = indxs[1]

			if sample:	nparams['C'][xindxs,yindxs,0]=scipy.stats.beta.rvs(nparams['theta_c_crp'][k][0],nparams['theta_c_crp'][k][1],size=len(nparams['C'][xindxs,yindxs,0]))
			if sample:	nparams['C'][xindxs,yindxs,1]=scipy.stats.beta.rvs(nparams['theta_c_crp'][k][2],nparams['theta_c_crp'][k][3],size=len(nparams['C'][xindxs,yindxs,0]))
			if sample:	nparams['C'][xindxs,yindxs,2]=scipy.stats.beta.rvs(nparams['theta_c_crp'][k][4],nparams['theta_c_crp'][k][5],size=len(nparams['C'][xindxs,yindxs,0]))

			ll += np.sum(distributions.beta_logpdf(nparams['C'][xindxs,yindxs,0],nparams['theta_c_crp'][k][0],nparams['theta_c_crp'][k][1]))	
			ll += np.sum(distributions.beta_logpdf(nparams['C'][xindxs,yindxs,1],nparams['theta_c_crp'][k][2],nparams['theta_c_crp'][k][3]))		
			ll += np.sum(distributions.beta_logpdf(nparams['C'][xindxs,yindxs,2],nparams['theta_c_crp'][k][4],nparams['theta_c_crp'][k][5]))

		return ll, surface


	def logl_compute(self, oid, new_surface,cur_surfaces, mode='update'):
		new_ll=0
		if mode == 'death' or mode == 'birth':
			surfaces = cur_surfaces
		else:
			if oid == None:
				#surfaces = copy.deepcopy(self.surfaces)
				surfaces = cur_surfaces
			else:
				surfaces = copy.deepcopy(cur_surfaces)
				surfaces[oid] = new_surface

		for i in surfaces.keys():
	
			if DPMIXTURE:
				ll, surfaces[i] = self.sample_CRP(surfaces[i], sample=False); new_ll+=ll
				if np.isinf(ll):	pdb.set_trace()
				ll, surfaces[i] = self.sample_thetac_CRP(surfaces[i], sample=False); new_ll+=ll
				if np.isinf(ll):	pdb.set_trace()
				ll, surfaces[i] = self.sample_C_CRP(surfaces[i],sample=False); new_ll += ll
				if np.isinf(ll):	pdb.set_trace()
			else:
				ll, surfaces[i] = self.sample_mixing(surfaces[i],sample=False);new_ll+=ll
	
				ll, surfaces[i] = self.sample_Z(surfaces[i],sample=False);new_ll+=ll
	
				ll, surfaces[i] = self.sample_thetac(surfaces[i],sample=False);new_ll+=ll
				ll, surfaces[i] = self.sample_C(surfaces[i],sample=False);new_ll+=ll
	
			# ll, nparams = self.sample_mu(nparams,sample=False); new_ll+=ll
	
			ll, surfaces[i] = self.sample_X(surfaces[i],sample=False); new_ll+=ll
			if np.isinf(ll):	pdb.set_trace()
			ll, self.pflip = self.sample_pflip(sample=False); new_ll += ll
			if np.isinf(ll):	pdb.set_trace()

			ll, surfaces[i] = self.sample_tx(surfaces[i],sample=False); new_ll+=ll
			ll, surfaces[i] = self.sample_ty(surfaces[i],sample=False); new_ll+=ll
			ll, surfaces[i] = self.sample_tz(surfaces[i],sample=False); new_ll+=ll
			
			ll, surfaces[i] = self.sample_rx(surfaces[i],sample=False); new_ll+=ll
			ll, surfaces[i] = self.sample_ry(surfaces[i],sample=False); new_ll+=ll
			ll, surfaces[i] = self.sample_rz(surfaces[i],sample=False); new_ll+=ll
			
			ll, surfaces[i] = self.sample_sx(surfaces[i],sample=False); new_ll+=ll
			ll, surfaces[i] = self.sample_sy(surfaces[i],sample=False); new_ll+=ll
			# ll, surfaces[i] = self.sample_sz(surfaces[i],sample=False); new_ll+=ll

			#pass params to renderer
			# surfaces[i].updateColors(surfaces[i].params['C'])
			# surfaces[i].updateControlPoints(surfaces[i].params['X'])
		
		variance = 0.01

		#render all surfaces
		if MOTION:
			rendering = dict()
			for kk in range(len(self.obs)):
				rendering[kk] = bezier.display({'surfaces':surfaces,'rot_angle':self.rot_angles[kk]}, capture=True)/255.0
				for ii in range(3):
					new_ll += np.sum(distributions.normal_logpdf(mu=self.obs[kk][:,:,ii],var=variance,x=rendering[kk][:,:,ii]))
			return new_ll, rendering
		else:
			rendering = bezier.display({'surfaces':surfaces,'rot_angle':None}, capture=True)/255.0
			
			if self.gpyramid:
				for ii in range(0,len(self.gpr_layers)+1):
					if ii == 0:
						for jj in range(3):
							new_ll += np.sum(distributions.normal_logpdf(mu=self.obs[ii][:,:,jj],var=variance,x=rendering[:,:,jj]))
					else:
						sigma = self.gpr_layers[ii-1]
						for jj in range(3):
							ren_gpr = scipy.ndimage.gaussian_filter(rendering[:,:,i],sigma=sigma)
							new_ll += np.sum(distributions.normal_logpdf(mu=self.obs[ii][:,:,jj],var=variance,x=ren_gpr))
			else:
				for ii in range(3):
					new_ll += np.sum(distributions.normal_logpdf(mu=self.obs[:,:,ii],var=variance,x=rendering[:,:,ii]))

			return new_ll, rendering


	def MCMC(self,ll,new_ll, surface, nsurface, rendering):
		if np.isinf(new_ll):
			#reject
			self.ll = ll
			return surface
		if log(np.random.rand()) < new_ll - ll: #accepted
			self.ll = new_ll
			return nsurface
		else:
			self.ll = ll
			return surface


	def propose_translation(self,oid, surface, cur_surfaces, dim):
		nsurface = copy.deepcopy(surface)
		if dim == 0:
			_,nsurface = self.sample_tx(nsurface,sample=True)
		elif dim == 1:
			_,nsurface = self.sample_ty(nsurface,sample=True)
		elif dim == 2:
			_,nsurface = self.sample_tz(nsurface, sample=True)
		new_ll,rendering = self.logl_compute(oid=oid,new_surface=nsurface, cur_surfaces=cur_surfaces)
		return self.MCMC(self.ll, new_ll, surface, nsurface, rendering)

	def propose_rotation(self,oid, surface, cur_surfaces,  dim):
		nsurface = copy.deepcopy(surface)
		if dim == 0:
			_,nsurface = self.sample_rx(nsurface,sample=True)
		elif dim == 1:
			_,nsurface = self.sample_ry(nsurface,sample=True)
		elif dim == 2:
			_,nsurface = self.sample_rz(nsurface, sample=True)
		new_ll,rendering = self.logl_compute(oid=oid,new_surface=nsurface, cur_surfaces=cur_surfaces)
		return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)

	def propose_scale(self, oid, surface, cur_surfaces,  dim):
		nsurface = copy.deepcopy(surface)
		if dim == 0:
			_,nsurface = self.sample_sx(nsurface,sample=True)
		elif dim == 1:
			_,nsurface = self.sample_sy(nsurface,sample=True)
		elif dim == 2:
			_,nsurface = self.sample_sz(nsurface, sample=True)
		new_ll,rendering = self.logl_compute(oid=oid,new_surface=nsurface, cur_surfaces=cur_surfaces)
		return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)

	def propose_pflip(self, oid, surface, cur_surfaces, dim):
		npflip = copy.deepcopy(pflip)
		_,npflip = self.sample_pflip(sample=True)
		new_ll,rendering = self.logl_compute(oid=None,new_surface=surface, cur_surfaces=cur_surfaces)
		return self.MCMC(self.ll, new_ll, pflip, npflip,rendering)

	def propose_mixing(self, oid, surface, cur_surfaces):
		nsurface = copy.deepcopy(surface)
		_,nsurface = self.sample_mixing(nsurface,sample=True)
		new_ll,rendering = self.logl_compute(oid=oid,new_surface=nsurface, cur_surfaces=cur_surfaces)
		return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)

	def propose_Z(self,oid, surface, cur_surfaces, xind, yind):
		nsurface = copy.deepcopy(surface)
		current_Zij_sample = np.where(np.random.multinomial(1,nsurface.params['mix'])==1)[0][0]
		nsurface.params['Z'][xind,yind]=current_Zij_sample
		new_ll,rendering = self.logl_compute(oid=oid,new_surface=nsurface, cur_surfaces=cur_surfaces)
		return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)

	def propose_CRP(self,oid, surface,cur_surfaces, xind, yind):
		nsurface = copy.deepcopy(surface)
		height=np.shape(nsurface.params['Z'])[0]
		width=np.shape(nsurface.params['Z'])[1]
		current_Zij_sample = nsurface.params['Z_CRPOBJ'].sample_pt((xind*width)+yind)#unflatten
		#only evaluate if sampled value is different otherwise just return for now
		if surface.params['Z'][xind,yind] == current_Zij_sample:
			return surface
		else:		
			nsurface.params['Z'][xind,yind]=current_Zij_sample
			new_ll,rendering = self.logl_compute(oid=oid,new_surface=nsurface, cur_surfaces=cur_surfaces)
			return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)

	def propose_thetac(self, oid, surface,cur_surfaces, k, ii):
		a=surface.params['thetac_a']; b=surface.params['thetac_b']
		nsurface = copy.deepcopy(surface)
		nsurface.params['theta_c'][k,ii]=np.random.gamma(a,b)#np.random.uniform(a,b)
		new_ll,rendering = self.logl_compute(oid=oid,new_surface=nsurface, cur_surfaces=cur_surfaces)
		return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)
		# for k in range(surface.K):
		# 	for repeat in range(1):
		# 		nparams['theta_c']=theta_c
		# 		for ii in [1,3,5]:
		# 			nparams['theta_c'][k,ii] = np.random.uniform(0.2,1)
		# 		new_ll, nparams = self.logl_compute(nparams)

	def propose_thetac_crp(self, oid, surface, cur_surfaces, k, ii):
		a=surface.params['thetac_a']; b=surface.params['thetac_b']
		nsurface = copy.deepcopy(surface)
		nsurface.params['theta_c_crp'][k][ii] = np.random.gamma(a,b)#np.random.uniform(a,b)
		new_ll,rendering = self.logl_compute(oid=oid,new_surface=nsurface, cur_surfaces=cur_surfaces)
		return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)


	def propose_C(self,oid, surface,cur_surfaces,xind,yind,channel):
		nsurface = copy.deepcopy(surface)
		cid = nsurface.params['Z'][xind,yind]
		if channel == 0:
			a = nsurface.params['theta_c'][cid,0]
			b = nsurface.params['theta_c'][cid,1]
		elif channel == 1:
			a = nsurface.params['theta_c'][cid,2]
			b = nsurface.params['theta_c'][cid,3]
		else:
			a = nsurface.params['theta_c'][cid,4]
			b = nsurface.params['theta_c'][cid,5]
		sample=np.random.beta(a,b)
		if sample == 1.0:	sample = 0.99
		if sample == 0.0:	sample = 0.01

		nsurface.params['C'][xind,yind,channel] = sample
		new_ll,rendering = self.logl_compute(oid=oid,new_surface=nsurface, cur_surfaces=cur_surfaces)
		return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)

	def propose_C_CRP(self,oid, surface,cur_surfaces,xind,yind,channel):
		nsurface = copy.deepcopy(surface)
		cid = nsurface.params['Z'][xind,yind]
		if channel == 0:
			a = nsurface.params['theta_c_crp'][cid][0]
			b = nsurface.params['theta_c_crp'][cid][1]
		elif channel == 1:
			a = nsurface.params['theta_c_crp'][cid][2]
			b = nsurface.params['theta_c_crp'][cid][3]
		else:
			a = nsurface.params['theta_c_crp'][cid][4]
			b = nsurface.params['theta_c_crp'][cid][5]
		nsurface.params['C'][xind,yind,channel]=np.random.beta(a,b)
		new_ll,rendering = self.logl_compute(oid=oid,new_surface=nsurface, cur_surfaces=cur_surfaces)
		return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)


	# def propose_mu(self,params,xind,yind):
	# 	nparams = copy.deepcopy(params)
	# 	nparams['mu_p'][xind,yind] = np.random.uniform(nparams['mu_l'],nparams['mu_u'])
	# 	new_ll, nparams = self.logl_compute(nparams)
	# 	return self.MCMC(self.ll, new_ll, nparams)		

	# def propose_X(self,params,xind,yind):
	# 	nparams = copy.deepcopy(params)
	# 	pdb.set_trace()
	# 	nparams['X'][xind,yind,2]=np.random.normal(nparams['mu_p'][xind,yind],nparams['X_var'])
	# 	new_ll, nparams = self.logl_compute(nparams)
	# 	return self.MCMC(self.ll, new_ll, nparams)	

	def propose_X(self,oid, surface,cur_surfaces,xind,yind,dim, _type):
		nsurface = copy.deepcopy(surface)
		#mixture proposal
		prop_ll = 0
		if _type == 'local':
			sigma=0.05
			old_val = nsurface.params['X'][xind,yind,2]
			done=False
			while done == False:
				new_val=np.random.normal(nsurface.params['X'][xind,yind,2], sigma)
				if new_val <= nsurface.params['X_u'] and new_val >= nsurface.params['X_l']:
					done=True
			nsurface.params['X'][xind,yind,dim] = new_val		
			prop_ll += scipy.stats.multivariate_normal.logpdf(nsurface.params['X'][xind,yind,dim],old_val,sigma) - scipy.stats.multivariate_normal.logpdf(old_val,nsurface.params['X'][xind,yind,dim],sigma)
		else:
			nsurface.params['X'][xind,yind,dim]=np.random.uniform(nsurface.params['X_l'],nsurface.params['X_u'])
		new_ll,rendering = self.logl_compute(oid=oid,new_surface=nsurface, cur_surfaces=cur_surfaces)
		new_ll += prop_ll
		return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)

	def set_observation(self,fname):
		obs = scipy.misc.imread(fname)/255.0
		if self.gpyramid:
			self.obs=dict()
			self.obs[0]=obs
			for ii in range(len(self.gpr_layers)):
				self.obs[ii+1]=np.zeros(np.shape(self.obs[0]))
				sigma = self.gpr_layers[ii]
				self.obs[ii+1][:,:,0] = scipy.ndimage.gaussian_filter(self.obs[0][:,:,0],sigma=sigma)
				self.obs[ii+1][:,:,1] = scipy.ndimage.gaussian_filter(self.obs[0][:,:,1],sigma=sigma)
				self.obs[ii+1][:,:,2] = scipy.ndimage.gaussian_filter(self.obs[0][:,:,2],sigma=sigma)
		else:
			self.obs = obs			

	def set_observation_sequence(self,dir):
		import glob
		allfiles = glob.glob(dir+'/*.png')
		self.obs = dict()
		cnt=0
		for _f in allfiles:
			self.obs[cnt]=scipy.misc.imread(_f)/255.0
			cnt+=1
		self.rot_angles = np.zeros(len(self.obs))

	def infer_transforms(self,surfaces,ii,repeats):
		for repeat in range(repeats):
			for j in range(3):
				surfaces[ii] = self.propose_translation(ii,surfaces[ii],surfaces,j)
			for j in range(3):
				surfaces[ii] = self.propose_rotation(ii,surfaces[ii],surfaces,j)
			for j in range(2):#skip inference on z=axis
				surfaces[ii] = self.propose_scale(ii,surfaces[ii],surfaces,j)

	def RJMCMC_UPDATE(self, surfaces):
		for ii in surfaces.keys():

			self.infer_transforms(surfaces,ii,repeats=1)

			if DPMIXTURE:
				for repeat in range(surfaces[ii].nPts/3):
					xind = np.random.randint(0,surfaces[ii].nPts)
					yind = np.random.randint(0,surfaces[ii].nPts)				
					surfaces[ii] = self.propose_CRP(ii,surfaces[ii],surfaces, xind, yind)

				for repeat in range(2):
					for k in surfaces[ii].params['Z_CRPOBJ'].countvec.keys():
						for dd in [1,3,5]:
							surfaces[ii] = self.propose_thetac_crp(ii,surfaces[ii],surfaces,k=k,ii=dd)

				self.infer_transforms(surfaces,ii,repeats=1)

				for repeat in range(surfaces[ii].nPts/3):
					xind = np.random.randint(0,surfaces[ii].nPts)
					yind = np.random.randint(0,surfaces[ii].nPts)					
					# self.pflip = self.propose_pflip(self.pflip)
					for channel in range(3):
						surfaces[ii] = self.propose_C_CRP(ii,surfaces[ii],surfaces,xind,yind,channel)
			else:
				######################## GMM ##############################
				surfaces[ii] = self.propose_mixing(ii, surfaces[ii], surfaces)
				for repeat in range(surfaces[ii].nPts/3):
					xind = np.random.randint(0,surfaces[ii].nPts)
					yind = np.random.randint(0,surfaces[ii].nPts)				
					surfaces[ii] = self.propose_Z(ii,surfaces[ii],surfaces, xind, yind)

				for repeat in range(2):
					for k in range(surfaces[ii].K):
						for dd in [1,3,5]:
							surfaces[ii] = self.propose_thetac(ii,surfaces[ii],surfaces,k=k,ii=dd)

				self.infer_transforms(surfaces,ii,repeats=1)

				for repeat in range(surfaces[ii].nPts/3):
					xind = np.random.randint(0,surfaces[ii].nPts)
					yind = np.random.randint(0,surfaces[ii].nPts)					
					# self.pflip = self.propose_pflip(self.pflip)
					for channel in range(3):
						surfaces[ii] = self.propose_C(ii,surfaces[ii],surfaces,xind,yind,channel)

			# for repeat in range(3):
			# 	xind = np.random.randint(0,self.divisions)
			# 	yind = np.random.randint(0,self.divisions)
			# 	self.params = self.propose_mu(self.params,xind,yind)

			for _type in ['local','prior']:
				for repeat in range(surfaces[ii].nPts/1):
					xind = np.random.randint(0,surfaces[ii].nPts)
					yind = np.random.randint(0,surfaces[ii].nPts)
					if INF_ALL_DIM_X:
						DIMS = [0,1,2]
					else:
						DIMS=[2]
					for dim in DIMS:
						surfaces[ii] = self.propose_X(ii,surfaces[ii],surfaces,xind,yind,dim,_type)

			self.infer_transforms(surfaces,ii,repeats=1)
			
			print self.ll
			# if DPMIXTURE:
				# print 'CRP-K(OBJ[',ii,'] => ', len(self.surfaces[ii].params['Z_CRPOBJ'].countvec)
		print

	def RJMCMC_BIRTH(self):
		nsurfaces = copy.deepcopy(self.surfaces)
		old_ll = copy.deepcopy(self.ll)
		if len(nsurfaces) == 0:
			max_oid = 0
		else:
			max_oid = max(nsurfaces.keys())+1
		nsurfaces[max_oid] = bezier.BezierPatch()

		cp_id = min(nsurfaces.keys())
		nsurfaces[max_oid].params['Z'] = copy.deepcopy(self.surfaces[cp_id].params['Z'])
		nsurfaces[max_oid].params['theta_c'] = copy.deepcopy(self.surfaces[cp_id].params['theta_c'])
		nsurfaces[max_oid].params['theta_c_crp'] = copy.deepcopy(self.surfaces[cp_id].params['theta_c_crp'])
		nsurfaces[max_oid].params['C'] = copy.deepcopy(self.surfaces[cp_id].params['C'])
		nsurfaces[max_oid].params['Z_CRPOBJ'] = copy.deepcopy(self.surfaces[cp_id].params['Z_CRPOBJ'])
		# nsurfaces[max_oid].params['X'] = copy.deepcopy(self.surfaces[cp_id].params['X'])
		# nsurfaces[max_oid].params['X'][:,:,2] *= 0

		_, nsurfaces[max_oid] = self.sample_prior_object(nsurfaces[max_oid], 'birth')

		if DELAYED_REJECTION:
			for dr_repeats in range(BIRTH_DR_TOTAL_INTERMEDIATE_PROPOSAL):
				self.RJMCMC_UPDATE(nsurfaces)
			new_ll, rendering = self.logl_compute(oid=None,new_surface=None, cur_surfaces=nsurfaces)
		else:
			# _, nsurfaces[max_oid] = self.sample_prior_object(nsurfaces[max_oid])
			new_ll,rendering = self.logl_compute(oid=max_oid, new_surface=None, cur_surfaces=nsurfaces, mode='birth');
		
		new_ll += log(self.move_probabilities[2])-log(self.move_probabilities[1]) #p_death-p_birth:correction for dim change due to RJMCMC
		self.surfaces=self.MCMC(old_ll, new_ll, self.surfaces, nsurfaces,rendering)
		print 'birth attempt'

	def RJMCMC_DEATH(self):
		old_ll = copy.deepcopy(self.ll)
		if len(self.surfaces.keys()) == 0:
			return
		nsurfaces = copy.deepcopy(self.surfaces)
		keys = nsurfaces.keys()
		indx = np.random.randint(0,len(keys))
		remove_oid = keys[indx]
		del nsurfaces[remove_oid]

		if DELAYED_REJECTION:
			for dr_repeats in range(1):
				self.RJMCMC_UPDATE(nsurfaces)
			new_ll, rendering = self.logl_compute(oid=None,new_surface=None, cur_surfaces=nsurfaces)
		else:
			new_ll,rendering = self.logl_compute(oid=None, new_surface=None, cur_surfaces=nsurfaces, mode='death');

		new_ll += log(self.move_probabilities[1])-log(self.move_probabilities[2]) #p_birth-p_death:correction for dim change due to RJMCMC
		self.surfaces=self.MCMC(old_ll, new_ll, self.surfaces, nsurfaces,rendering)
		print 'death attempt'

	def infer(self,ITERS):
		global parallel
		self.ll = self.sample_prior()

		for itr in range(ITERS):
			t1=time.time()

			if RJMCMC:
				chosen = np.where(np.random.multinomial(1,self.move_probabilities)==1)[0][0]
			else:
				chosen = 0

			if chosen == 0:#update
				self.RJMCMC_UPDATE(self.surfaces)
			elif chosen == 1: #birth
				self.RJMCMC_BIRTH()
			elif chosen == 2:
				self.RJMCMC_DEATH()

			if (itr%35) == 0:
				print 'saving ...'
				if MOTION:
					rendering=dict()
					for kk in range(len(self.obs)):
						rendering[kk] = bezier.display({'surfaces':self.surfaces,'rot_angle':self.rot_angles[kk]}, capture=True)/255.0
						scipy.misc.imsave('image_dump/'+str(itr)+'_s'+str(kk)+'.png', rendering[kk])
					pickle.dump({'params':self.surfaces}, open('params.pkl','wb'))
				else:
					rendering = bezier.display({'surfaces':self.surfaces,'rot_angle':None}, capture=True)/255.0
					scipy.misc.imsave('image_dump/'+str(itr)+'.png', rendering)
					pickle.dump({'params':self.surfaces}, open('params.pkl','wb'))



if __name__ == "__main__":
	bezier.setup()
	S = Sampler()
	if MOTION:
		S.set_observation_sequence("data/seq1")
	else:
		S.set_observation("obs_pot.png")
	S.infer(10000)



