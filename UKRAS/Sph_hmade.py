import numpy as np
from scipy.special import sph_harm

def computerealSH(L,data,reala=True):
    Nsh = []
    for l in range(0,L+1):
        if(l==0):
            fcolors = sph_harm(0,l,data[0,:] , data[1,:])
            if(reala):
                fcolors = fcolors.real
            else:
                fcolors = fcolors.imag
            # fmax, fmin = fcolors.max(), fcolors.min()
            # fcolors = (fcolors - fmin)/(fmax - fmin)
            Nsh.append(fcolors)
        else:
            for m in range(0,l+1):
                fcolors = sph_harm(m,l,data[0,:] , data[1,:])
                if(reala):
                    fcolors = fcolors.real
                else:
                    fcolors = fcolors.imag
                # fmax, fmin = fcolors.max(), fcolors.min()
                # fcolors = (fcolors - fmin)/(fmax - fmin)
                Nsh.append(fcolors)

    return np.array(Nsh)


def forwardSHT(L,data,real=True):
    Ndirs = data.shape[1]
    Nsh = (L+1)*(L+1)
    mag = data[2,:]
    Y_N = computerealSH(L,data,real)
    # invY_N = 1/Ndirs * Y_N
    invY_N  = np.linalg.pinv(np.transpose(Y_N))
    coeffs = np.dot(invY_N,mag)
    return coeffs


# var AT = numeric.transpose(A);
# numeric.dot(numeric.inv(numeric.dot(AT,A)),AT);
