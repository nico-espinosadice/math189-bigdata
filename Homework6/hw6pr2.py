"""
Starter file for hw6pr2 of Big Data Spring 2020

Before attemping the helper functions, please familiarize with pandas and numpy
libraries. Tutorials can be found online:
http://pandas.pydata.org/pandas-docs/stable/tutorials.html
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

Please COMMENT OUT any steps in main driver before you finish the corresponding
functions for that step. Otherwise, you won't be able to run the program
because of errors.

Note:
1. When filling out the functions below, note that
	1) Let k be the rank for approximation

2. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each part.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
import imageio
import urllib

if __name__ == '__main__':

	# =============STEP 0: LOADING DATA=================
	print('==> Loading image data...')
	img = ndimage.imread(urllib.request.urlopen('http://i.imgur.com/X017qGH.jpg'), flatten=True)

	"*** YOUR CODE HERE ***"
	shuffle_img = img.copy()
	shuffle_img = shuffle_img.flatten()
	np.random.shuffle(shuffle_img)
	"*** END YOUR CODE HERE ***"

	# reshape the shuffled image
	shuffle_img = shuffle_img.reshape(img.shape)
	print("shuffle_img shape:", shuffle_img.shape)
	print("img shape:", img.shape)

	# =============STEP 1: RUNNING SVD ON IMAGES=================
	print('==> Running SVD on images...')

	"*** YOUR CODE HERE ***"
	U, S, V = np.linalg.svd(img)
	U_s, S_s, V_s = np.linalg.svd(shuffle_img)
	"*** END YOUR CODE HERE ***"

	# =============STEP 2: SINGULAR VALUE DROPOFF=================
	print('==> Singular value dropoff plot...')
	k = 100
	plt.style.use('ggplot')

	"*** YOUR CODE HERE ***"
	orig_S_plot = plt.plot(S[0:k], color = "purple")
	shuf_S_plot = plt.plot(S_s[0:k], color = "pink")
	"*** END YOUR CODE HERE ***"

	plt.legend((orig_S_plot, shuf_S_plot), \
		('original', 'shuffled'), loc = 'best')
	plt.title('Singular Value Dropoff for Clown Image')
	plt.ylabel('singular values')
	plt.savefig('dropoff.png', format='png')
	plt.close()

	# =============STEP 3: RECONSTRUCTION=================
	print('==> Reconstruction with different ranks...')
	rank_list = [2, 10, 20]
	plt.subplot(2, 2, 1)
	plt.imshow(img, cmap='Greys_r')
	plt.axis('off')
	plt.title('Original Image')

	for index in range(len(rank_list)):
		k = rank_list[index]
		plt.subplot(2, 2, 2 + index)

		"*** YOUR CODE HERE ***"
		# Optimal reconstruction given by: X = USV^T (Murphy, Equation 12.59)
		# We want to plot the reconstructed image for a given rank (k), so we
		#	only take the first k columns of the matrix U so that we plot
		#	exactly rank k. (The colums of U are the left singular vectors).
		U_k = U[0:, 0:k]

		# We know that S has the singular values on the main diagonal (given in Murphy 12.2.3)
		# S has shape (499,), so we need to make it a diagonal matrix
		#	i.e. make its shape (499, 499) so it has the singular values on the 
		# 	main diagonal, as described in Murphy.
		# Like for U_k, we want to limit our reconstruction image to be rank k
		S_k = S[0:k]
		S_k_diag = np.diag(S_k)

		# In Murphy, Equation 12.59, we have V^T, but I ran into an error
		#	when I used the transpose of V. 
		# The columns of V are the right singular vectors
		V_k = V[0:k, 0:]

		reconstructed_image = U_k @ S_k_diag @ V_k
		plt.imshow(reconstructed_image, cmap = 'Greys_r')
		"*** END YOUR CODE HERE ***"

		plt.title('Rank {} Approximation'.format(k))
		plt.axis('off')

	plt.tight_layout()
	plt.savefig('reconstruction.png', format='png')
	plt.close()
