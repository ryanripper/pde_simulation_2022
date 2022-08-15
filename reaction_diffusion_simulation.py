import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from scipy.sparse import spdiags
import cProfile
import pstats
import math
import io
import pandas as pd

### DEFINE FUNCTIONS

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
   
### SIMULATION OF REACTION-DIFFUSION EQUATION

st.markdown("## Simulation of Reaction-Diffusion Equation")

st.markdown("We aim to simulate the Reaction-Diffusion Equation as defined below:")

st.latex(r'''
   \frac{\partial u}{\partial t} = \frac{\partial}{\partial x} \left(\delta (x, y) \frac{\partial u}{\partial x} \right) + \frac{\partial}{\partial y} \left(\delta (x, y) \frac{\partial u}{\partial y} \right) + f(u)
   ''')
   
st.markdown(r'''
   Where $u_t(x,y)$ is a spatio–temporal process at spatial location $s = (x, y)$ in two-dimensional Euclidean space at time t and $\delta (x, y)$ is a spatially varying diffusion coefficient.
   ''')
   
st.markdown(r'''
   The “reaction” term $f(u)$ describes the population growth dynamics.
   ''')
   
st.markdown(r'''
   We can write the matrix form of the solution to the Reaction-Diffusion Equation where $f(u)$ is zero as:
   ''')
   
st.latex(r'''
   u_t = H(\delta, \Delta_t, \Delta_x)u_{t−1} + H_B(\delta, \Delta_t, \Delta_x)u_B
   ''')
   
st.markdown("Where the term on the left of the RHS represents the interior solution and the term on the right of the RHS represents the boundary solution.")

st.markdown("### Visualizations")

st.markdown("The left-hand side of this module allows the user to alter values that change the dynamics of the matrix form to the solution to the Reaction-Diffusion Equation.")

st.markdown(r'''
   Once the values have been altered, by pushing the button labeled "RUN REACTION-DIFFUSION EQUATION", the module will numerically solve the Reaction-Diffusion Equation and display both a two- and three-dimensional visualization of the solution over space and time.
   ''')

st.markdown("#### Two-Dimensional Visualization (Scalar Data)")

plot_spot_2d = st.empty()

with plot_spot_2d:
   st.markdown(r'''
      [Please click "RUN REACTION-DIFFUSION SIMULATION" to view.]
      ''')

st.markdown("#### Three-Dimensional Visualization (Surface Plot)")

plot_spot_3d = st.empty()

with plot_spot_3d:
   st.markdown(r'''
      [Please click "RUN REACTION-DIFFUSION SIMULATION" to view.]
      ''')

# mass = 1.00
mass = st.sidebar.number_input(label = "Mass of Starting Value", value = 1.00)

# ny = 20
ny = st.sidebar.number_input(label = "Number of Columns", value = 20)

# nx = 20
nx = st.sidebar.number_input(label = "Number of Rows", value = 20)

# T = 60
T = st.sidebar.number_input(label = "Number of Time Steps", value = 60)

# dx = 1
dx = st.sidebar.number_input(label = "Horizontal Diffusion Rate", value = 1)

# dy = 1
dy = st.sidebar.number_input(label = "Vertical Diffusion Rate", value = 1)

# dt = 0.10
dt = st.sidebar.number_input(label = "Time Step", value = 0.10)

# alpha = 0.00
alpha = st.sidebar.number_input(label = "Reaction Rate", value = 0.00)

def reaction_diffusion_simulation():
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

   ustrt = np.zeros((nx, ny))
   
   mid_col = math.ceil(nx / 2) - 1
   
   mid_row = math.ceil(ny / 2) - 1

   ustrt[mid_col, mid_row] = mass

   uI[:, 0] = ustrt.flatten()

   for t in range(2, T + 1):
      uI[:, t - 1] = np.dot(HI, uI[:, t - 1 - 1]) + np.dot(HB, uB[:, t - 1 - 1])
      
   min_z = np.ndarray.min(uI)
   
   max_z = np.ndarray.max(uI)

   ### VISUALIZE SIMULATION

   for j in range(1 - 1, T):
      tmp = uI[:, j]

      tmp2 = np.reshape(tmp, (nx, ny))
      
      #####
      
      fig_2d = px.imshow(tmp2, color_continuous_scale = "viridis")
      
      #####
      
      fig_3d = go.Figure(data = [go.Surface(z = tmp2, colorscale = "viridis")])
      
      fig_3d.update_layout(scene = dict(zaxis = dict(nticks = 4, range = [min_z, max_z])))
      
      #####
      
      with plot_spot_2d:
         st.plotly_chart(fig_2d)
      
      with plot_spot_3d:
         # st.pyplot(fig_3d)
         st.plotly_chart(fig_3d)
           
if st.sidebar.button("RUN REACTION-DIFFUSION SIMULATION"):
   pr = cProfile.Profile()
   
   pr.enable()
   
   reaction_diffusion_simulation()
   
   pr.disable()
   
   result = io.StringIO()
   
   pstats.Stats(pr, stream = result).print_stats()
   
   result = result.getvalue()
   
   result = "ncalls" + result.split("ncalls")[-1]
   
   result = "\n".join([",".join(line.rstrip().split(None, 5)) for line in result.split("\n")])
   
   df = pd.read_csv(io.StringIO(result), sep = ",")
   
   st.dataframe(df)