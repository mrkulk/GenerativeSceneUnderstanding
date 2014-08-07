#Tejas D Kulkarni (tejask@mit.edu | tejasdkulkarni@gmail.com)
import numpy as np
import pdb,copy,time, pickle
from distributions import *
import bezier,distributions,scipy.stats
import skimage.color,scipy.misc
from skimage import filter
import matplotlib.pyplot as plt
from PIL import Image


parallel = True

class Sampler():
	def __init__(self):
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
		self.movep = [0.9,0.1,0.1] #update/birth/death
		self.pflip = 0.01

	def sample_prior_object(self,surface):
		ret_ll = 0
		ll, surface = self.sample_mixing(surface); ret_ll+=ll
		ll, surface = self.sample_Z(surface); ret_ll+=ll
		ll, surface = self.sample_thetac(surface); ret_ll+=ll
		ll, surface = self.sample_C(surface); ret_ll+=ll
		# ll,surface = self.sample_mu(surface)
		# ret_ll+=ll
		ll, surface = self.sample_X(surface); ret_ll+=ll
		
		ll, surface = self.sample_tx(surface); ret_ll+=ll
		ll, surface = self.sample_ty(surface); ret_ll+=ll
		ll, surface = self.sample_tz(surface); ret_ll+=ll
		
		ll, surface = self.sample_rx(surface); ret_ll+=ll
		ll, surface = self.sample_ry(surface); ret_ll+=ll
		ll, surface = self.sample_rz(surface); ret_ll+=ll
		
		ll, surface = self.sample_sx(surface); ret_ll+=ll
		ll, surface = self.sample_sy(surface); ret_ll+=ll
		ll, surface = self.sample_sz(surface); ret_ll+=ll
		
		return ret_ll, surface


	def sample_prior(self):
		ret_ll = 0
		self.surfaces[0] = bezier.BezierPatch()
		ll, self.surfaces[0] = self.sample_prior_object(self.surfaces[0]); ret_ll+=ll
		ll,_ = self.logl_compute(oid=None, surface=self.surfaces); ret_ll+=ll
		return ret_ll

	def sample_tx(self, surface, sample=True):
		nparams = surface.params
		a=-1;b=1
		if sample: nparams['tx']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['tx'])
		return ll, surface

	def sample_ty(self, surface, sample=True):
		nparams = surface.params
		a=-1;b=1
		if sample: nparams['ty']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['ty'])
		return ll, surface

	def sample_tz(self, surface, sample=True):
		nparams = surface.params
		a=-1;b=1
		if sample: nparams['tz']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['tz'])
		return ll, surface

	def sample_rx(self, surface, sample=True):
		nparams = surface.params
		a=-45;b=45
		if sample: nparams['rx']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['rx'])
		return ll, surface

	def sample_ry(self, surface, sample=True):
		nparams = surface.params
		a=-45;b=45
		if sample: nparams['ry']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['ry'])
		return ll, surface

	def sample_rz(self, surface, sample=True):
		nparams = surface.params
		a=-45;b=45
		if sample: nparams['rz']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['rz'])
		return ll, surface

	def sample_sx(self, surface, sample=True):
		nparams = surface.params
		a=0.6;b=1.2
		if sample: nparams['sx']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['sx'])
		return ll, surface

	def sample_sy(self, surface, sample=True):
		nparams = surface.params
		a=0.6;b=1.2
		if sample: nparams['sy']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['sy'])
		return ll, surface

	def sample_sz(self, surface, sample=True):
		nparams = surface.params
		a=0.6;b=1.2
		if sample: nparams['sz']=np.random.uniform(a,b)
		ll = distributions.uniform_logpdf(a,b,nparams['sz'])
		return ll, surface

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

	def sample_thetac(self, surface, sample=True):
		ll=0
		nparams = surface.params
		for k in range(surface.K):
			if sample:
				nparams['theta_c'][k,0]=1;nparams['theta_c'][k,2]=1;nparams['theta_c'][k,4]=1;
				nparams['theta_c'][k,1]=np.random.uniform(0.2,1);nparams['theta_c'][k,3]=np.random.uniform(0.2,1);nparams['theta_c'][k,5]=np.random.uniform(0.2,1)		
			ll+=distributions.uniform_logpdf(0.2,1,nparams['theta_c'][k,1])
			ll+=distributions.uniform_logpdf(0.2,1,nparams['theta_c'][k,3])
			ll+=distributions.uniform_logpdf(0.2,1,nparams['theta_c'][k,5])
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
				if sample:	nparams['X'][ii,jj,2]=np.random.uniform(nparams['X_l'],nparams['X_u'])
				ll+= distributions.uniform_logpdf(nparams['X_l'],nparams['X_u'],nparams['X'][ii,jj,2])
				#MRF costs
				if ii > 0:	ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,2]-nparams['X'][ii-1,jj,2])**2
				if ii < surface.nPts-1:	ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,2]-nparams['X'][ii+1,jj,2])**2
				if jj > 0: ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,2]-nparams['X'][ii,jj-1,2])**2
				if jj < surface.nPts-1: ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,2]-nparams['X'][ii,jj+1,2])**2

				if ii > 0 and jj>0: ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,2]-nparams['X'][ii-1,jj-1,2])**2
				if ii < surface.nPts-1 and jj>0: ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,2]-nparams['X'][ii+1,jj-1,2])**2
				if ii < surface.nPts-1 and jj<surface.nPts-1: ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,2]-nparams['X'][ii+1,jj+1,2])**2
				if ii > 0 and jj<surface.nPts-1: ll+=-nparams['lambda_x']*(nparams['X'][ii,jj,2]-nparams['X'][ii-1,jj+1,2])**2

		return ll, surface


	def sample_C(self, surface, sample=True):
		nparams = surface.params
		ll=0
		for k in range(surface.K):
			indxs = np.where(nparams['Z']==k)
			xindxs = indxs[0]; yindxs = indxs[1]
			if sample:	nparams['C'][xindxs,yindxs,0]=np.random.beta(nparams['theta_c'][k,0],nparams['theta_c'][k,1],len(nparams['C'][xindxs,yindxs,0]))
			ll += np.sum(scipy.stats.beta.logpdf(nparams['C'][xindxs,yindxs,0],nparams['theta_c'][k,0],nparams['theta_c'][k,1]))
			# if np.isinf(ll):
				# pdb.set_trace()
			if sample:	nparams['C'][xindxs,yindxs,1]=np.random.beta(nparams['theta_c'][k,2],nparams['theta_c'][k,3],len(nparams['C'][xindxs,yindxs,0]))
			ll += np.sum(scipy.stats.beta.logpdf(nparams['C'][xindxs,yindxs,1],nparams['theta_c'][k,2],nparams['theta_c'][k,3]))		
			# if np.isinf(ll):
				# pdb.set_trace()
			if sample:	nparams['C'][xindxs,yindxs,2]=np.random.beta(nparams['theta_c'][k,4],nparams['theta_c'][k,5],len(nparams['C'][xindxs,yindxs,0]))
			ll += np.sum(scipy.stats.beta.logpdf(nparams['C'][xindxs,yindxs,2],nparams['theta_c'][k,4],nparams['theta_c'][k,5]))
			# if np.isinf(ll):
				# pdb.set_trace()
		return ll, surface


	def logl_compute(self, oid, surface):
		new_ll=0
		if oid == None:
			surfaces = copy.deepcopy(self.surfaces)
		else:
			surfaces = copy.deepcopy(self.surfaces)
			surfaces[oid] = surface
		for i in range(len(surfaces)):
			ll, surfaces[i] = self.sample_mixing(surfaces[i],sample=False);new_ll+=ll
			ll, surfaces[i] = self.sample_Z(surfaces[i],sample=False);new_ll+=ll
	
			ll, surfaces[i] = self.sample_thetac(surfaces[i],sample=False);new_ll+=ll
			ll, surfaces[i] = self.sample_C(surfaces[i],sample=False);new_ll+=ll
			# ll, nparams = self.sample_mu(nparams,sample=False); new_ll+=ll
	
			ll, surfaces[i] = self.sample_X(surfaces[i],sample=False); new_ll+=ll
			ll, self.pflip = self.sample_pflip(sample=False); new_ll += ll
			
			ll, surfaces[i] = self.sample_tx(surfaces[i],sample=False); new_ll+=ll
			ll, surfaces[i] = self.sample_ty(surfaces[i],sample=False); new_ll+=ll
			ll, surfaces[i] = self.sample_tz(surfaces[i],sample=False); new_ll+=ll
			
			ll, surfaces[i] = self.sample_rx(surfaces[i],sample=False); new_ll+=ll
			ll, surfaces[i] = self.sample_ry(surfaces[i],sample=False); new_ll+=ll
			ll, surfaces[i] = self.sample_rz(surfaces[i],sample=False); new_ll+=ll
			
			ll, surfaces[i] = self.sample_sx(surfaces[i],sample=False); new_ll+=ll
			ll, surfaces[i] = self.sample_sy(surfaces[i],sample=False); new_ll+=ll
			ll, surfaces[i] = self.sample_sz(surfaces[i],sample=False); new_ll+=ll

			#pass params to renderer
			surfaces[i].updateColors(surfaces[i].params['C'])
			surfaces[i].updateControlPoints(surfaces[i].params['X'])

		#render all surfaces
		rendering = bezier.display(surfaces, capture=True)/255.0
	
		for i in range(3):
			#new_ll += np.sum(distributions.normal_logpdf(self.obs[:,:,i],rendering[:,:,i],self.pflip))
			new_ll += np.sum(distributions.normal_logpdf(self.obs[:,:,i],rendering[:,:,i],0.05))
			
		return new_ll, rendering


	def MCMC(self,ll,new_ll, surface, nsurface, rendering):
		if np.isinf(new_ll):
			#reject
			self.ll = ll
			return surface
		if log(np.random.rand()) < new_ll - ll: #accepted
			self.ll = new_ll
			print 'accepted'
			return nsurface
		else:
			self.ll = ll
			return surface


	def propose_translation(self,oid, surface, dim):
		nsurface = copy.deepcopy(surface)
		if dim == 0:
			_,nsurface = self.sample_tx(nsurface,sample=True)
		elif dim == 1:
			_,nsurface = self.sample_ty(nsurface,sample=True)
		elif dim == 2:
			_,nsurface = self.sample_tz(nsurface, sample=True)
		new_ll,rendering = self.logl_compute(oid,nsurface)
		return self.MCMC(self.ll, new_ll, surface, nsurface, rendering)

	def propose_rotation(self,oid, surface, dim):
		nsurface = copy.deepcopy(surface)
		if dim == 0:
			_,nsurface = self.sample_rx(nsurface,sample=True)
		elif dim == 1:
			_,nsurface = self.sample_ry(nsurface,sample=True)
		elif dim == 2:
			_,nsurface = self.sample_rz(nsurface, sample=True)
		new_ll,rendering = self.logl_compute(oid,nsurface)
		return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)

	def propose_scale(self, oid, surface, dim):
		nsurface = copy.deepcopy(surface)
		if dim == 0:
			_,nsurface = self.sample_sx(nsurface,sample=True)
		elif dim == 1:
			_,nsurface = self.sample_sy(nsurface,sample=True)
		elif dim == 2:
			_,nsurface = self.sample_sz(nsurface, sample=True)
		new_ll,rendering = self.logl_compute(oid, nsurface)
		return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)

	def propose_pflip(self, pflip):
		npflip = copy.deepcopy(pflip)
		_,npflip = self.sample_pflip(sample=True)
		new_ll,rendering = self.logl_compute(oid=None,surface=None)
		return self.MCMC(self.ll, new_ll, pflip, npflip,rendering)

	def propose_mixing(self, oid, surface):
		nsurface = copy.deepcopy(surface)
		_,nsurface = self.sample_mixing(nsurface,sample=True)
		new_ll,rendering = self.logl_compute(oid, nsurface)
		return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)

	def propose_Z(self,oid, surface, xind, yind):
		nsurface = copy.deepcopy(surface)
		current_Zij_sample = np.where(np.random.multinomial(1,nsurface.params['mix'],1)==1)[1][0]
		nsurface.params['Z'][xind,yind]=current_Zij_sample
		new_ll,rendering = self.logl_compute(oid, nsurface)
		return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)

	def propose_thetac(self, oid, surface, k, ii):
		nsurface = copy.deepcopy(surface)
		nsurface.params['theta_c'][k,ii]=np.random.uniform(0.2,1)
		new_ll,rendering = self.logl_compute(oid, nsurface)
		return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)
		# for k in range(surface.K):
		# 	for repeat in range(1):
		# 		nparams['theta_c']=theta_c
		# 		for ii in [1,3,5]:
		# 			nparams['theta_c'][k,ii] = np.random.uniform(0.2,1)
		# 		new_ll, nparams = self.logl_compute(nparams)


	def propose_C(self,oid, surface,xind,yind,channel):
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
		nsurface.params['C'][xind,yind,channel]=np.random.beta(a,b)
		new_ll,rendering = self.logl_compute(oid, nsurface)
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

	def propose_X(self,oid, surface,xind,yind):
		nsurface = copy.deepcopy(surface)
		nsurface.params['X'][xind,yind,2]=np.random.uniform(nsurface.params['X_l'],nsurface.params['X_u'])
		new_ll,rendering = self.logl_compute(oid, nsurface)
		return self.MCMC(self.ll, new_ll, surface, nsurface,rendering)

	def set_observation(self,fname):
		self.obs = scipy.misc.imread(fname)/255.0


	def infer_transforms(self,ii,repeats):
		for repeat in range(repeats):
			for j in range(3):
				self.surfaces[ii] = self.propose_translation(ii,self.surfaces[ii],j)
			for j in range(3):
				self.surfaces[ii] = self.propose_rotation(ii,self.surfaces[ii],j)
			for j in range(3):
				self.surfaces[ii] = self.propose_scale(ii,self.surfaces[ii],j)


	def infer(self,ITERS):
		global parallel
		self.ll = self.sample_prior()

		for itr in range(ITERS):
			t1=time.time()
			for ii in range(len(self.surfaces)):

				self.infer_transforms(ii,repeats=1)

				self.surfaces[ii] = self.propose_mixing(ii, self.surfaces[ii])

				for repeat in range(self.surfaces[ii].nPts/2):
					xind = np.random.randint(0,self.surfaces[ii].nPts)
					yind = np.random.randint(0,self.surfaces[ii].nPts)				
					self.surfaces[ii] = self.propose_Z(ii,self.surfaces[ii], xind, yind)

				for repeat in range(2):
					for k in range(self.surfaces[ii].K):
						for dd in [1,3,5]:
							self.surfaces[ii] = self.propose_thetac(ii,self.surfaces[ii],k=k,ii=dd)

				self.infer_transforms(ii,repeats=1)

				for repeat in range(self.surfaces[ii].nPts/2):
					xind = np.random.randint(0,self.surfaces[ii].nPts)
					yind = np.random.randint(0,self.surfaces[ii].nPts)					
					self.pflip = self.propose_pflip(self.pflip)
					for channel in range(3):
						self.surfaces[ii] = self.propose_C(ii,self.surfaces[ii],xind,yind,channel)

				# for repeat in range(3):
				# 	xind = np.random.randint(0,self.divisions)
				# 	yind = np.random.randint(0,self.divisions)
				# 	self.params = self.propose_mu(self.params,xind,yind)

				for repeat in range(self.surfaces[ii].nPts/2):
					xind = np.random.randint(0,self.surfaces[ii].nPts)
					yind = np.random.randint(0,self.surfaces[ii].nPts)
					self.surfaces[ii] = self.propose_X(ii,self.surfaces[ii],xind,yind)

				self.infer_transforms(ii,repeats=1)
				
				print 'LOGL:', self.ll, ' pflip:', self.pflip
				if (itr%10) == 0:
					print 'saving ...'
					rendering = bezier.display(self.surfaces, capture=True)/255.0
					scipy.misc.imsave('image_dump/'+str(itr)+'.png', rendering)
					pickle.dump({'params':self.surfaces}, open('params.pkl','wb'))



if __name__ == "__main__":
	bezier.setup()
	S = Sampler()
	S.set_observation("obs_cone.png")
	S.infer(10000)



