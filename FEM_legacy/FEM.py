import numpy as np
from scipy import linalg
from tqdm import tqdm


##========================================================================================
# FEM Inference
##========================================================================================


# Feb 25, 2024 Manu: Setting default niter_max to 100 and l2 to 100.
def FEM_fit(x,y_onehot,niter_max=100,l2=100):       

    if y_onehot.ndim == 1:
        y_onehot = np.expand_dims(y_onehot, axis=1)


    #print(niter_max)        
    l,n = x.shape
    m = y_onehot.shape[1] # number of categories (if one hot encoding)
    #print('%d states for this site'%(m))
    
    x_av = np.mean(x,axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True)

    # Vipul, Manu, 01/31/24: Explicitly symmetrizing c
    c = (c + c.T)/2
    #if not np.allclose(c, c.T, rtol=1e-05, atol=1e-08, equal_nan=False):
    #    print('sample covariance matrix is not symmetric!!')


    # 2019.07.16:  l2 = lamda/(2L)
    c += l2*np.identity(n)/(2*l)

    # Vipul, Manu, 01/31/24: Using least-square to 'invert' later
    #c_inv = linalg.pinvh(c)
    #print('c_inv shape: ', c_inv.shape)

    H0 = np.zeros(m)
    W = np.zeros((n,m))
    #print('y_onehot shape: ',y_onehot.shape)

    for i in tqdm(range(m)):
        #print(f'Doing {i} out of {m}...')
        y = y_onehot[:,i]  # y = {0,1}
        y1 = 2*y - 1       # y1 = {-1,1}

        # initial values
        h0 = 0.

        w = np.random.normal(0.0,1./np.sqrt(n),size=(n))

        for iloop in range(niter_max):
            h = h0 + x.dot(w)
            y1_model = np.tanh(h/2.)    

            # update local field
            t = h!=0    
            h[t] *= y1[t]/y1_model[t]

            # Manu: limit h -> 0
            h[~t] = 2*y1[~t]

            # find w from h    
            h_av = h.mean()
            dh = h - h_av 
            dhdx = dh[:,np.newaxis]*dx[:,:]

            dhdx_av = dhdx.mean(axis=0)

            #Manu, 01/31/24
            # Vipul suggestion: Use linalg solver instead of finding pseudoinverse a priori
            #w = c_inv.dot(dhdx_av)
            w, res, rnk, singular_vals = linalg.lstsq(c, dhdx_av)

            h0 = h_av - x_av.dot(w)

        H0[i] = h0
        W[:,i] = w

    return H0,W  
