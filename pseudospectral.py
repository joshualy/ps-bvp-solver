from __future__ import division
import numpy as np
from numpy.polynomial.chebyshev import chebval
from numpy.linalg import solve
from scipy.optimize import root
from scipy.interpolate import BarycentricInterpolator


def cheb(N):
	# N = Number of gridpoints
	x =	 np.cos((np.pi/(N-1))*np.linspace(0,(N-1),N))
	x.shape = (N,1)
	lin = np.linspace(0,(N-1),N)
	lin.shape = (N,1)
	
	c = np.ones((N,1))
	c[0], c[-1] = 2., 2.
	c = c*(-1.)**lin
	X = x*np.ones(N) # broadcast along 2nd dimension (columns)
	
	dX = X - X.T
	
	D = (c*(1./c).T)/(dX + np.eye(N))
	D = D - np.diag(np.sum(D.T,axis=0))
	x = x[:,0]
	return D, x


class ps_solver():
	"""A BVP solver that uses a pseudospectral discretization on a grid of Chebychev points.
	
	The pseudospectral method can achieve high accuracy, often with errors < 1e-10, 1e-13, etc.
	
	Methods:
	__init__(F,bcs,guess,N,dim,domain) -- Initializes the BVP.
	solve() -- Uses a standard rootsolving method to solve the discretized BVP
                   on the Chebychev grid. 
	eval(x_grid) -- A barycentric interpolator is used to evaluate the polynomial 
                        interpolating the solution through the Chebychev nodes. 
	spectral_coeff() -- Computes the coefficients of the interpolating polynomial
                            when expanded using a basis of Chebychev polynomials. These
                            coefficients can then be visually inspected to check convergence
                            of the solution.  It is also recommended to check the solution 
                            by approximating the residual ( res = Y' - F(Y,x) ) using a 
                            standard finite difference scheme
	
	
	cheb_gridpoints() -- Used internally to construct a difference operator over 
                             the Chebychev grid.
	G() -- Used internally to discretize the ODE and its associated 
               boundary conditions.
	"""
	
	def __init__(self,F,bcs,guess,N=41,dim=1,domain=[-1,1]): 
		"""Defines the boundary value problem (BVP).
		
		Arguments: 
		F -- the ODE function, where Y' = F(Y,x). F must be vectorized appropriately;
                     see the example below.
		guess --  a function or an array to initialize the rootsolving process. If an 
                          array, it must provide values at the Chebychev nodes.
		bcs   --  a dictionary defining boundary conditions for the BVP. Can handle 
                          general linear boundary conditions.
		
		Keyword Arguments:
		N -- the number of Chebychev nodes to use in the discretization (default 41)
		dim -- dimension of the BVP (default 1)
		domain -- list of length two, defining the solution interval. The default is 
                          the Chebychev domain. (default [-1,1])
		
		Example: 
		BVP:  y'' = -3*epsilon*y/(epsilon + x**2)**2,  -0.1 <= x < = 0.1.
		
		BCs: y(0.1) = -y(-0.1) = 0.1/sqrt(epsilon + 0.01)
		
		BVP as a first-order system: Let y0 = y, y1 = y'.
			y0 ' = y1
			y1 ' = -3*epsilon*y0/(epsilon + x**2)**2
		
		
		Python: 
		
		epsilon = .001
		N = 41
		def F(U,x):	 
			U0, U1 = U[0], U[1]
			return np.vstack([ U1,
							  ( -3.*epsilon*U0 )/(epsilon+x**2)**2.
							 ])
		
		# A simple initial guess. Most initial guesses must be more elaborate.
		guess = 0.1*np.ones((2,N)) 
		
		
		bcs = {"ODE_Cs": { 'L': [1], 
							 # Coordinate indexes where ODE will be enforced at RHS
						  'R': [1] 
					     } , 
			  "BCs":    { 'L': np.array([ [1., 0., -.1/np.sqrt(.01 + epsilon)]   ]),
							 # Projective conditions			  
						  'R': np.array([ [1., 0., .1/np.sqrt(.01 + epsilon)]  ])  
						 }, 
		  		}
	
		bvp = ps_solver(F,bcs,guess,N=N,dim=2,domain=[-0.1,0.1])
		bvp.solve()
		
		x = np.linspace(-0.1,0.1, 201)
		solution = bvp.eval(x)
		
		# Calculate spectral coefficients for the solution.
		coeff = bvp.spectral_coeff()
		"""
		if domain == [-1,1]:
			self.F = F
		else:
			assert (len(domain)==2), "domain is list of length two."
			a,b = domain
			assert (a < b), "The left endpoint must be < the right endpoint."
			self.F = lambda U,x:(.5*(b-a))*F( U, a + (.5*(b-a))*(x+1.) )

		self.domain = domain
		self.N = N # N = number of Chebychev gridpoints
		self.guess = guess
		self.bcs = bcs
		self.dim = dim
		self.solved = False
		self.cheb_gridpoints()

		
	def solve(self):
		# Is guess a function or an array? Obtain an array
		if callable(self.guess):
			a,b = self.domain
			guess = lambda x: self.guess(a + (.5*(b-a))*(x+1.) )
			assert ( guess(-1).shape[0]==self.dim )
			
			sol = root(  self.G, np.reshape( guess(self._x), (self.dim*self.N,) )  )
		else: 
			sol = root(  self.G, np.reshape( self.guess, (self.dim*self.N,) )  )
		
		assert sol.success, "BVP solution failed."
		self.solved = True
		a,b = self.domain
		self.x = a + (.5*(b-a))*(self._x+1.)
		self.sol = np.zeros((self.dim,self.N))
		for j in range(self.dim): 
			self.sol[j] = sol.x[j*self.N:(j+1)*self.N]
		self.interpolator = BarycentricInterpolator(self._x,self.sol,axis=1)
	
	
	def G(self,U):
		N = self.N		   # N = number of gridpoints, N-1 is number of subintervals
		M = N-2			   # M = number of interior grid points
		dim = self.dim
		Mat = np.zeros((dim*N,dim*N))
		Nonlinear_Part = np.zeros(dim*N)
		U_sq = np.reshape(U,(dim,N)) # Fills out row by row.
		
		# Rows index components, columns index chebychev nodes.
		FU_sq = self.F(U_sq,self._x)
		
		for j in range(dim):
			Mat[j*M:(j+1)*M,j*N:(j+1)*N] = self._D[1:-1] 
			Nonlinear_Part[j*M:(j+1)*M] = FU_sq[j,1:-1]
		
		# Recall that the Chebyshev grid is ordered right to left, with  
		# x[0] = 1, x[-1] = -1. This is especially important when coding 
		# boundary conditions.
		
		ind = 0
		# ind needs to index rows from [dim*M:dim*(M+1)], and [dim*(M+1):]
		# [row_in_matrix is indexed by ind,
		# 				col_in_matrix signifies component of solution ]
		
		# # ODE Conditions.
		assert (len(self.bcs["ODE_Cs"]['R']) + len(self.bcs["ODE_Cs"]['L'])==dim),('Number'+
		' of ODE Conditions must equal the dimension of the ODE')

		# # RHS
		for j in range(dim):
			if j in self.bcs["ODE_Cs"]['R']:
				Mat[dim*M+ind,j*N:(j+1)*N] = self._D[0]
				Nonlinear_Part[dim*M+ind] = FU_sq[j,0]
				ind += 1
				
		# # LHS
		for j in range(dim):
			if j in self.bcs["ODE_Cs"]['L']:
				Mat[dim*M+ind,j*N:(j+1)*N] = self._D[-1]
				Nonlinear_Part[dim*M+ind] = FU_sq[j,-1]
				ind += 1
		
		# # Boundary Conditions.
		# x = np.arange(16)
		# print(x[::4]) - Views the beginning of all segments of length 4
		for j in range(self.bcs["BCs"]['R'].shape[0]):
			Mat[dim*M+ind,::N] = self.bcs["BCs"]['R'][j,:dim]
			Nonlinear_Part[dim*M+ind] = self.bcs["BCs"]['R'][j,dim]
			ind += 1
		
		# print(x[3::4]) - Views the end of all segments of length 4
		for j in range(self.bcs["BCs"]['L'].shape[0]):
			Mat[dim*M+ind,N-1::N] = self.bcs["BCs"]['L'][j,:dim]
			Nonlinear_Part[dim*M+ind] = self.bcs["BCs"]['L'][j,dim]
			ind += 1
		
		out = Mat.dot(U) - Nonlinear_Part
		return out

	
	def cheb_gridpoints(self):
		N = self.N
		# N = Number of gridpoints
		x =	 np.cos((np.pi/(N-1))*np.linspace(0,(N-1),N))
		x.shape = (N,1)
		lin = np.linspace(0,(N-1),N)
		lin.shape = (N,1)
		
		c = np.ones((N,1))
		c[0], c[-1] = 2., 2.
		c = c*(-1.)**lin
		X = x*np.ones(N) # broadcast along 2nd dimension (columns)
		
		dX = X - X.T
		
		D = (c*(1./c).T)/(dX + np.eye(N))
		self._D	= D - np.diag(np.sum(D.T,axis=0))
		self._x = x[:,0]
	
	
	def eval(self,x_grid):
		assert (self.solved), "The BVP must be solved first."
			# Evaluate the interpolating function on x_grid.
			
		a,b = self.domain
		assert (np.max(x_grid)<= b and np.min(x_grid) >=a), "Cannot evaluate outside the solution domain"
		# (2./(b-a)) *(x_grid-a) - 1 : transform x_grid to Chebychev domain
		u_vals = self.interpolator.__call__( (2./(b-a)) *(x_grid-a) - 1)
		return u_vals
		
		
	def spectral_coeff(self):
		assert self.solved, "The BVP must be solved first."

		# The jth column of M is the jth Chebychev 
		# polynomial phi_j evaluated at the Chebychev points x
		N = self.N
		M = np.zeros((N,N))
		for j in range(N):
		    c = np.zeros(N)
		    c[j] = 1.
		    M[:,j] = chebval(self._x,c) 
		
		self.coeff = np.zeros(self.sol.shape)
		for j in range(self.dim):
			self.coeff[j] = solve(M,self.sol[j])
		return self.coeff
		

def check_dimensions():
	"""
	May be helpful for debugging. 
	"""
	# pass
	# x = np.arange(1,3)
	# print(x.shape)
	# print(x)
	# print("\n")
	#
	# x = x[np.newaxis,:]
	# print(x.shape)
	# print(x)
	# print("\n")
	#
	# x_stacked = np.vstack((x,x))
	# print(x_stacked)
	# print("\n")
	#
	# z = np.reshape(x_stacked,(4,))
	# print(z.shape)
	# print(z)
	# print("\n")
	#
	# z_square = np.reshape(z,(2,2))
	# print(z_square.shape)
	# print(z_square)
	#
	# print("\n")
	# print(z_square[0,:])
	#
	# x = np.arange(16)
	# print(x)
	# print(x[::4]) # Hits the beginning of all segments of length 4
	# print(x[3::4]) # Hits the end of all segments of length 4
	#
	#
	#
	#########################################################
	# # ODE Conditions.   # BACKUP!!!
	# # RHS
	# Mat[dim*M:dim*(M+1),j*N:(j+1)*N] = self._D[0]
	# Nonlinear_Part[dim*M:dim*(M+1)] = FU_sq[:,0]
	#
	# # LHS
	# Mat[dim*(M+1):,j*N:(j+1)*N] = self._D[-1]
	# Nonlinear_Part[dim*(M+1):] = FU_sq[:,-1]
	#
	# out = Mat.dot(U) - Nonlinear_Part
	# out[dim*M] = U[0]-self.bcs[1]
	# out[dim*(M+1)] = U[N-1]-self.bcs[0]
	# # Equation for Mat[dim*M] is overwritten for boundary condition at RHS
	# # Equation for Mat[dim*(M+1)] is overwritten for boundary condition at LHS
	#########################################################
	return


	