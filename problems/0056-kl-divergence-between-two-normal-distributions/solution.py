import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
	var_p = sigma_p ** 2
	var_q = sigma_q ** 2

	term1 = np.log(sigma_q/sigma_p)
	term2 = (var_p + (mu_p - mu_q) ** 2) / (2 * var_q)

	return term1 + term2 - 0.5
