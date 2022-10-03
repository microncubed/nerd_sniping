from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

def nerd_snipe(nx,ny,bx,by,cx,cy):
    '''
    Determines the potential in the case of a current source (bx,by) and a current sink (cx,cy) at different points in a square lattice of dimensions (nx,ny). Related to solving the problem in xkcd cartoon 356. A constant potential boundary condition (V = 0) is set around the domain.
    
    Parameters
    ----------
    nx,ny int,int: dimensions of the lattice
    bx,by int,int: location of the current source
    cx,cy int,int: location of the current sink
    
    Returns
    ----------
    V np.ndarray: potential
    
    '''
    
    # Builds the sparse matrix
    Arow = []
    Acol = []
    Adata =[]

    steps = nx*ny

    for i in range(nx):
        for j in range(ny):
            diag = j*nx+i
            if i == 0:
                Arow.append(diag)
                Acol.append(diag)
                Adata.append(1)
            elif j == 0:
                Arow.append(diag)
                Acol.append(diag)
                Adata.append(1)
            elif i == nx-1:
                Arow.append(diag)
                Acol.append(diag)
                Adata.append(1)
            elif j == ny -1:
                Arow.append(diag)
                Acol.append(diag)
                Adata.append(1)       
            else:
                Arow.append(diag)
                Acol.append(diag)
                Adata.append(4)

                Arow.append(diag)
                Acol.append(diag-1)
                Adata.append(-1)

                Arow.append(diag)
                Acol.append(diag+1)
                Adata.append(-1)

                Arow.append(diag)
                Acol.append(diag+nx)
                Adata.append(-1)

                Arow.append(diag)
                Acol.append(diag-nx)
                Adata.append(-1)         

    Adata_coo=coo_matrix((Adata, (Arow, Acol)), shape=(steps, steps))
    Adata_csr = Adata_coo.tocsr()
    
    # Sets up the source term
    q = np.zeros((steps))
    q[bx*nx+by]=1
    q[cx*nx+cy]=-1
    
    # Solves the matrix equation
    V_flat = spsolve(Adata_csr,q.T)
    
    # Converts the solution from a flat array to a 2d array
    V = np.zeros((nx,ny))

    for i in range(nx):
        for j in range(ny):
            V[i,j]= V_flat[i*nx+j]

    return V