import numpy as np
import pdb,math,copy
from scipy.special import gammaln
from scipy.special import betaln
from numpy import array, log, exp

class CRP():

	def __init__(self,N,alpha):
		self.N=N
		self.alpha = alpha

	def sample_prior(self):
		N = self.N; alpha = self.alpha
		if N<=0:
			return []
		table_assignments = [1] # first customer sits at table 1

		for i in range(1,N):
			allcids = list(np.unique(table_assignments))
			ii=0
			probabilities=list(np.zeros(len(allcids)+1))
			for cid in allcids:
				probabilities[ii]=len(np.where(table_assignments==cid)[0])/(N-1.0+alpha)
				ii+=1
			allcids.append(max(allcids)+1)
			probabilities[-1]=alpha/(N-1.0+alpha)
			probabilities = np.array(probabilities)
			probabilities = probabilities / sum(probabilities)
			indx = np.random.multinomial(1,probabilities)
			indx = np.where(indx==1)[0][0]
			chosen_cid = allcids[indx]		
			table_assignments.append(chosen_cid)

		self.Z = table_assignments
		
		self.uniques = np.unique(self.Z)
		self.countvec = dict()
		for i in range(len(self.uniques)):
			cid = self.uniques[i]
			self.countvec[cid]=len(np.where(self.Z == cid)[0])

		return table_assignments

	def sample_pt(self,indx):
		N = self.N; alpha = self.alpha
		cur_cid = self.Z[indx]

		# print 'ZZ:',np.unique(self.Z)
		# print 'CV:',self.countvec.keys()
		# print self.countvec

		countvec = self.countvec
		countvec[cur_cid] -= 1
		if countvec[cur_cid] <= 0:
			del countvec[cur_cid]

		allcids = countvec.keys()
		ii=0
		probabilities=list(np.zeros(len(allcids)+1))
		for cid in allcids:
			probabilities[ii]=countvec[cid]/(N-1.0+alpha)
			ii+=1
		allcids.append(max(allcids)+1)
		probabilities[-1]=alpha/(N-1.0+alpha)
		probabilities = np.array(probabilities)
		probabilities = probabilities / sum(probabilities)
		indx = np.random.multinomial(1,probabilities)
		indx = np.where(indx==1)[0][0]
		chosen_cid = allcids[indx]

		#remove and add in counvec dict
		if countvec.has_key(chosen_cid):
			countvec[chosen_cid]+=1
		else:
			countvec[chosen_cid]=1

		self.Z[indx] = chosen_cid
		# print 'chosen:', chosen_cid
		# print '------'
		return chosen_cid

	def logpdf(self):
		ll = 0
		for i in self.countvec.keys():
			ll += log_factorial(self.countvec[i])
		ll = ll - log_factorial(len(self.Z)-1)
		return ll


def dirichlet_logpdf(a,x):
		B = sum(gammaln(a)) - gammaln(sum(a))
		return np.sum(np.multiply(a-1,np.log(x))) - B

def log_factorial(x):
    """Returns the logarithm of x!
    Also accepts lists and NumPy arrays in place of x."""
    return gammaln(array(x)+1)

def multinomial_logpdf(z, ps, K):
	n=len(z)
	xs=np.zeros(K)
	for i in range(K):
		xs[i] = len(np.where(z==i)[0])
	result = log_factorial(n) - sum(log_factorial(xs)) + sum(xs * log(ps))
	return result


def gamma_logpdf(k,theta,x):
	return (k-1)*np.log(x) - ((1.0*x)/theta) - (gammaln(k)+k*(log(theta)))

def normal_logpdf(mu,var,x):
	return -(((x-mu)**2)/(2*var**2)) - (log(var)+0.5*log(2*math.pi))

def beta_logpdf(x,a,b):
	x_clip=np.clip(x,0.01,0.99)
	return (a-1)*log(x_clip)+(b-1)*log(1-x_clip) - betaln(a,b)
	
def uniform_logpdf(a,b,x):
	if len(np.shape(x)) == 0:
		if x>=a and x<=b:
			return -log(b-a)
		else:
			pdb.set_trace()
			return log(0)
	else:
		if (x >= a).all() and (x <= b).all():
			if len(np.shape(x)) > 0:
				return -np.shape(x)[0]*np.shape(x)[1]*log(b-a)
			else:
				return -log(b-a)
		else:
			pdb.set_trace()
			return log(0)






