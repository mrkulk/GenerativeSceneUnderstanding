import numpy as np
import pdb,math
from scipy.special import gammaln
from numpy import array, log, exp


class CRP():

	def __init__(self,N,alpha):
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


def uniform_logpdf(a,b,x):
	if len(np.shape(x)) == 0:
		if x>=a and x<=b:
			return -log(b-a)
		else:
			return log(0)
	else:
		if (x >= a).all() and (x <= b).all():
			if len(np.shape(x)) > 0:
				return -np.shape(x)[0]*np.shape(x)[1]*log(b-a)
			else:
				return -log(b-a)
		else:
			return log(0)






