from __future__ import division
import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.interpolate import BarycentricInterpolator
from pseudospectral import ps_solver


def test_prob_17_system_withclass(flag=False):
	N = 101	# N = number of gridpoints
	guess = .1*np.ones((2,N)) # A simple initial guess
	epsilon = .001 
	
	bcs = {"ODE_Cs": { 'L': [1], 
						 # Coordinate indexes where ODE will be enforced at RHS
					  'R': [1] 
				     } , 
		  "BCs":    { 'L': np.array([ [1., 0., -.1/np.sqrt(.01 + epsilon)]   ]),
						 # Projective conditions			  
					  'R': np.array([ [1., 0., .1/np.sqrt(.01 + epsilon)]  ])  
					 }, 
	  		}
	
	def F(U,x):	 
		return np.vstack([ U[1],
						  ( -3.*epsilon*U[0] )/(epsilon+x**2)**2.
						 ])	 
						 
	bvp = ps_solver(F,bcs,guess,N=N,dim=2,domain=[-0.1,0.1])
	bvp.solve()
	if bvp.solved==True:
		check = bvp.sol[0]
		x_vals = np.linspace(-0.1,0.1, 201)
		solution = bvp.eval(x_vals)
		u_vals = solution[0]
		
		ex = x_vals
		why = ex/np.sqrt( epsilon + ex**2. )   
	
		if flag:
			# Plot the solution. 
			import matplotlib.pyplot as plt
			plt.plot(bvp.x,check,'*k',linewidth=2.0)
			plt.plot(x_vals,u_vals,'-k',linewidth=2.0)
			plt.plot(ex,why,'-r',linewidth=2.0)
			plt.xlabel('x'); plt.ylabel('u')
			plt.title('Test Problem 17')
			plt.show()
		
			coeff = bvp.spectral_coeff()
			print("This plot graphs the spectral coefficients of the pseudospectral\n" +
			   "approximation of the soliton. ")
			for j in range(2):
				plt.loglog(np.abs(coeff[j]),'-*k',linewidth=2)
				plt.ylabel('$a$',fontsize=18);	plt.xlabel('$n$',fontsize=18)
				plt.title('Spectral Coefficients $a_n$ for solution.',fontsize=20)
				plt.show()


def test_solitons(flag=False):
	N = 121	# N = number of gridpoints
	# guess = .1*np.ones(2*N) # A simple initial guess
	s = 1 
	p = 2
	L = 12   # Numerical Infinity
	def F(U,x):	 
		return np.vstack([  U[1],
						    s*U[0]-U[0]**p/p,
						   -U[3],
						   -s*U[2]+U[2]**p/p,
						 ])	
						 
						 # Coordinate indexes where ODE will be enforced at LHS
	bcs = {"ODE_Cs": { 'L': [0,1,2,3], 
						 # Coordinate indexes where ODE will be enforced at RHS
					  'R': [] 
				     } , 
		  "BCs":    { 'L': np.array([ [1., 0., -1., 0., 0.],      # Matching condition
		  							  # [0., 0.,  0., 1., 0. ],     # Phase condition
								      [0., 1.,  0., 0., 0. ]   ]),
						 # Projective conditions			  
					  'R': np.array([ [1./np.sqrt(s), 1.,              0., 0., 0.],
		  							  [0.,            0.,  -1./np.sqrt(s), 1., 0. ]  ])  
					 }, 
	  		}
			
			
	def guess_func(x):
	    return 2./np.cosh(x)+.1*np.exp(-100.*x**2.)*np.cos(100.*x)**2.
		
	def guess_func_deriv(x):
		return -2.*np.tanh(x)/np.cosh(x) -20.*np.exp(-100.*x**2.)*(np.cos(100.*x)**2. + 
												np.cos(100.*x)*np.sin(100.*x) )
	def guess(x):
		return np.vstack([ guess_func(x),
	 					   guess_func_deriv(x),
					   	   guess_func(-x),
					   	 	guess_func_deriv(-x)
					      ])
	
	
	bvp = ps_solver(F,bcs=bcs,guess=guess,N=N,dim=4,domain=[0,L])
	bvp.solve()
	
	if bvp.solved==True:
		x_vals = np.linspace(0,L, 201)
		sol = bvp.eval(x_vals)
		u_vals = sol[0]
		
		def sol_anal(x):
			return (  p*(p+1)*.5/np.cosh( .5*(1-p)*x )**2.  )**(1./(p-1.))
			
		def sol_der(x):
			p = 2  # This simplifies the derivative quite a bit.
			b = -.5
			return 3*np.tanh(b*x)/np.cosh(b*x)**2.	
			
		if flag:
			# Plot the solution. 
			
			import matplotlib.pyplot as plt
			plt.plot(x_vals,sol[0],'-k',linewidth=2.8)
			plt.plot(-x_vals,sol[2],'-k',linewidth=2.8)
			plt.plot(x_vals,sol[1],'-k',linewidth=2.8)
			plt.plot(-x_vals,sol[3],'-k',linewidth=2.8)
			
			plt.plot(x_vals[::15],sol_anal(x_vals[::15]),'*r',markersize=5,linewidth=.8,label='Solution')
			plt.plot(-x_vals[::15],sol_anal(x_vals[::15]),'*r',markersize=5,linewidth=.8)
			plt.plot(x_vals,sol_anal(x_vals),'-r',linewidth=.8)
			plt.plot(-x_vals,sol_anal(x_vals),'-r',linewidth=.8)
			
			plt.plot(x_vals[::15],sol_der(x_vals[::15]),'*r',markersize=5,linewidth=.8,label="Derivative")
			plt.plot(-x_vals[::15],sol_der(-x_vals[::15]),'*r',markersize=5,linewidth=.8)
			plt.plot(x_vals,sol_der(x_vals),'-r',linewidth=.8)
			plt.plot(-x_vals,sol_der(-x_vals),'-r',linewidth=.8)
			plt.xlabel('x'); plt.ylabel('u')
			plt.title('Soliton')
			plt.show()
			
			# plt.plot(x_vals,np.abs(sol[0]-sol_anal(x_vals)),'-k',linewidth=2.0)
			# plt.plot(x_vals,np.abs(sol[2]-sol_anal(-x_vals)),'-b',linewidth=2.0)
			# plt.title('Error')
			# plt.show()
		
		bvp.spectral_coeff()
		print("This plot graphs the spectral coefficients of the pseudospectral\n" +
			   "approximation of the soliton. ")# +"The graph indicates extremely fast convergence.")
		for j in range(4):
			pass
			plt.loglog(np.abs(bvp.coeff[j]),'-*k',linewidth=2)
			plt.ylabel('$a$',fontsize=18);	plt.xlabel('$n$',fontsize=18)
			plt.title('Spectral Coefficients $a_n$ for solution.',fontsize=20)
			plt.show()


if __name__ == "__main__": 
	test_solitons(flag=True)
	test_prob_17_system_withclass(flag=True)
	
	