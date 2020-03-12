import numpy as np
import math
import torch.nn.init as weight_init
from sklearn.mixture import GaussianMixture








def gmm_kl(gmm_p, gmm_q, n_samples=10**5):
	X = gmm_p.sample(n_samples)
	log_p_X = gmm_p.score_samples(X[0])
	log_q_X = gmm_q.score_samples(X[0])

	return log_p_X.mean() - log_q_X.mean()
	
def gmm_js(gmm_p, gmm_q, n_samples=10**5):
	X = gmm_p.sample(n_samples)
	log_p_X = gmm_p.score_samples(X[0])
	log_q_X = gmm_q.score_samples(X[0])
	log_mix_X = np.logaddexp(log_p_X, log_q_X)

	Y = gmm_q.sample(n_samples)
	log_p_Y = gmm_p.score_samples(Y[0])
	log_q_Y = gmm_q.score_samples(Y[0])
	log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

	return (log_p_X.mean() - (log_mix_X.mean() - np.log(2)) 
		+ log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2

def KL_diverge_approx(gan, xsList, ysList, rsList, dsList, ganTrainEpoch, gmm_q=None):
	# sample generated data from GAN
	actions, states, statesPrime, rewards, dones = gan.sampleFromGAN(batchSize=2000)
	genData = np.hstack((states, statesPrime, np.rint(actions), rewards))
			
	if ganTrainEpoch == 0:
		# sample real data
		realData = []
		for i, done in enumerate(dsList):
			if int(done) == 0:
				realData.append(np.hstack((xsList[i], xsList[i+1], np.array(ysList[i]), np.array(rsList[i]))))
		rand_idx = list(np.random.randint(len(realData), size=2000))
		realData = np.array([realData[i] for i in rand_idx])

	# GMM
	gmm_p = GaussianMixture(n_components=50, covariance_type='full')
	gmm_p.fit(genData)

	if ganTrainEpoch == 0:
		gmm_q = GaussianMixture(n_components=50, covariance_type='full')
		gmm_q.fit(realData)

	accKL = gmm_kl(gmm_p, gmm_q, n_samples=10**5)
	accKLjs = gmm_js(gmm_p, gmm_q, n_samples=10**5)

	return accKL, accKLjs, gmm_q
