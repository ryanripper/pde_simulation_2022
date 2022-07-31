import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt

### define functions

def make_grid(minx, maxx, miny, maxy, dx, dy):
   xg = range(minx, maxx, dx)
   yg = range(maxy, miny, -dy)

   grid = np.empty((len(xg) * len(yg), 2))

   k = 0
   
   for x in xg:
      for y in yg:
         grid[k, 0] = x
         grid[k, 1] = y
         k = k + 1
   
   return (grid, xg, yg)
   
def make_Dx_sparse(nx, ny):
   Nx = nx + 2
   Ny = ny + 2
   N = Nx * Ny
   n = nx * ny

   Dx = np.zeros((n, N))

   strt = 1
   cnt = 0

   for i in range(1, nx + 1):
      for j in range(1, ny + 1):
         Dx[cnt, strt + (i - 1) * ny + j - 1] = -1
         Dx[cnt, strt + (i - 1) * ny + j + (2 * Ny) - 1] = 1
         
         cnt = cnt + 1
      
      strt = strt + 2

   return Dx

def make_Dy_sparse(nx, ny):
   Nx = nx + 2
   Ny = ny + 2
   N = Nx * Ny
   n = nx * ny

   Dy = np.zeros((n,N))

   strt = ny - 2
   cnt = 0

   for i in range(1, nx + 1):
      for j in range(1, ny + 1):
         Dy[cnt, strt + (i - 1) * ny + j - 1] = 1
         Dy[cnt, strt + (i - 1) * ny + j + 2 - 1] = -1
         
         cnt = cnt + 1
      
      strt = strt + 2
   
   return Dy
   
def make_MtilI_sparse(nx, ny):
   Nx = nx + 2
   Ny = ny + 2
   N = Nx * Ny
   n = nx * ny

   MtilI = np.zeros((n, N))

   strt = Ny + 1
   cnt = 0

   for i in range(1, nx + 1):
      for j in range(1, ny + 1):
         MtilI[cnt, strt + (i - 1) * ny + j - 1] = 1
         
         cnt = cnt + 1
   
      strt = strt + 2
      
   return MtilI

def make_MtilB_sparse(nx ,ny):
   Nx = nx + 2
   Ny = ny + 2 
   N = Nx * Ny
   n = nx * ny
   NB = 2 * Nx + 2 * Ny - 4

   MtilB = np.zeros((NB, N))

   for k in range(1, ny + 1):
      MtilB[k - 1, k - 1] = 1
      MtilB[Ny + Nx - 2 + k - 1, N + 1 - k - 1] = 1

   for i in range(1, nx + 1):
      MtilB[Ny + i - 1, Ny * (i + 1) - 1] = 1
      MtilB[NB + 1 - i - 1, (Ny * i) + 1 - 1] = 1

   return MtilB

def make_abcde(dt, dx, dy, alpha, Y, Dx, Dy, MtilI):
   dx2 = dx**2
   dy2 = dy**2
   YI = np.dot(MtilI, Y)
   DxY = np.dot(Dx, Y)
   DyY = np.dot(Dy, Y)

   a = 1 - (2 * dt / dx2) * YI - (2 * dt / dy2) * YI + dt * alpha

   b = (-dt / (4 * dx2)) * (DxY) + (dt / dx2) * YI

   c = (dt / (4 * dx2)) * (DxY) + (dt / dx2) * YI

   d = (-dt / (4 * dy2)) * (DyY) + (dt / dy2) * YI

   e = (dt / (4 * dy2)) * (DyY) + (dt / dy2) * YI

   return (a, b, c, d, e)
   
def make_HBmat_sparse(nx, ny, b, c, d, e):
   Nx = nx + 2
   Ny = ny + 2
   N = 2 * Nx + 2 * Ny - 4

   cnt = 0

   HB = np.zeros((nx * ny, N))

   for j in range(ny, (ny * nx) + 1, ny):
      HB[j - 1, Ny + cnt] = d[[j - 1]]
      HB[j - ny + 1 - 1, N - cnt + 1 - 2] = e[[j - ny + 1 - 1]]
      
      cnt = cnt + 1

   for k in range(1, ny + 1):
     HB[k - 1, 1 + k - 1] = b[[k - 1]]
     HB[(nx - 1) * ny + k - 1, 2 * Ny + Nx - 2 - k - 1] = c[[(nx - 1) * ny + k - 1]]

   return HB
   
def make_HImat_sparse(nx, ny, a, b, c, d, e):
   n = nx * ny

   mdiag = a

   diag_p1 = np.zeros((n, 1))
   diag_p1[range(2 - 1, n)] = d[range(1 - 1, n - 1)]
   diag_p1[range(ny + 1 - 1, n - 1, ny)] = 0

   diag_n1 = np.zeros((n, 1))
   diag_n1[range(1 - 1, n - 1)] = e[range(2 - 1, n)]
   diag_n1[range(ny - 1, n - 1, ny)] = 0

   diag_pny = np.zeros((n, 1))
   diag_pny[range(ny + 1 - 1, ny * nx)] = c[range(1 - 1, n - ny)]
   diag_nny = np.zeros((n, 1))
   diag_nny[range(1 - 1, ny * nx - ny)] = b[range(ny + 1 - 1, n)]

   B = np.zeros((n, 5))
   B[:, 1 - 1] = mdiag[:, 0]
   B[:, 2 - 1] = diag_p1[:, 0]
   B[:, 3 - 1] = diag_n1[:, 0]
   B[:, 4 - 1] = diag_pny[:, 0]
   B[:, 5 - 1] = diag_nny[:, 0]
   
   HI = spdiags(B.T, [0, 1, -1, ny, -ny], n, n).todense()

   return HI
   
### simulation of reaction-diffusion equation

ny = 20
nx = 20
T = 30
dx = 1
dy = 1
dt = 1
alpha = 0

dx2 = dx * dx
dy2 = dy * dy

grdxy, xg, yg = make_grid(1, nx + 2, 1, ny + 2, dx, dy)

DX = make_Dx_sparse(nx, ny)

DY = make_Dy_sparse(nx, ny)

MtilI = make_MtilI_sparse(nx, ny)

MtilB = make_MtilB_sparse(nx, ny)

n = nx * ny

nB, N = MtilB.shape

D = .2 * np.ones((N, 1))

uI = np.zeros((n, T))
uB = np.zeros((nB, T))

a, b, c, d, e = make_abcde(dt, dx, dy, alpha, D, DX, DY, MtilI)

HB = make_HBmat_sparse(nx, ny, b, c, d, e)

HI = make_HImat_sparse(nx, ny, a, b, c, d, e)

ustrt = np.zeros((ny, nx))

ustrt[9, 9] = .5

uI[:, 0] = ustrt.flatten()

for t in range(2, T + 1):
   uI[:, t - 1] = np.dot(HI, uI[:, t - 1 - 1]) + np.dot(HB, uB[:, t - 1 - 1])
   
### visualize simulation

for j in range(1 - 1, T):
   tmp = uI[:, j]

   tmp2 = np.reshape(tmp, (ny, nx))

   plt.imshow(tmp2)
   plt.colorbar()
   plt.show()