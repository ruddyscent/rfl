#!/usr/bin/python

# raman_fiber_laser.py - Raman fiber laser simulation package
# Copyright (C) 2008 Gwangju Institute of Science and Technology
#
# Author: Kyungwon Chun <kwchun@gist.ac.kr>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# This program is the implementation of the algorithm shown at 
# 'F. Castella, P. Chartier, E. Faou, D. Bayart, F. Leplingard,
# and C. Martinelli, "Raman laser: mathematical and numerical 
# analysis of a model," Math. Model. Numer. Anal., vol. 38, no. 
# 3, pp. 457-475, 2004.'
#
# n: number of Stokes level
# L: distance (m) between Bragg lattices 
# G: the matrix of Raman gain (1/(W m))
# alpha: the matrix of attenuation coefficients (dB/km)
# R0: reflectivity coefficients (%) of the Bragg lattices in x=0,
#     but the first element is the input power.
# RL: reflectivity coefficients (%) of the Bragg lattices in x=L,
#     but the last element is Rout.
# steps: number of steps for explicit Euler integration

try:
    import psyco
    psyco.profile()
    from psyco.classes import *
except:
    pass

from numpy.core import matrix, array, arange, zeros
from numpy.core import log, dot, cosh, sum, fabs, exp, sqrt, product
from scipy.linalg import inv, solve, pinv, det


class RamanFiberLaser:
    def __init__(self, L, R0, RL, G, alpha, steps):
        """
        L: distance (m) between Bragg lattices
        R0: reflectivity coefficients (%)of the Bragg lattices in x=0,
            but the first element is the input power.
        RL: reflectivity coefficients (%)of the Bragg lattices in x=L,
            but the last element is Rout.
        G: the matrix of Raman gain (1/(W m))
        alpha: the matrix of attenuation coefficients (dB/km)
        steps: number of steps for explicit Euler integration
        """
        if len(R0) % 2 == 0:
            self.castella = Even(L, R0, RL, G, alpha, steps)
        else:
            self.castella = Odd(L, R0, RL, G, alpha, steps)

    def run(self, error=None, n=None):
        if error is not None:
            self.castella.saturate_evolve(error)
        elif n is not None:
            self.castella.evolve_steps(n)

    def show(self, forward=[], backward=[]):
        from pylab import plot, show, xlabel, ylabel, title, legend
        
        x = arange(0, self.castella.L + self.castella.h, self.castella.h)
        xlabel('x (m)')
        ylabel('Power density (W)')
        titleString = 'Solution for n=' + str(self.castella.n) + ', P=' + str(self.castella.P)
        title(titleString)
        
        legendList = []
        if len(forward) == 0 and len(backward) == 0:
            forward = xrange(1, self.castella.n+1)
            backward = xrange(1, self.castella.n+1)
            
        for i in forward:
            plot(x, self.castella.F[i-1])
            tag = 'F' + str(i)
            legendList.append(tag)
            
        for i in backward:
            plot(x, self.castella.B[i-1])
            tag = 'B' + str(i)
            legendList.append(tag)
        
        legend(legendList, shadow=True)
        show()
            
    def write_hdf5(self, filename):
        from tables import openFile
        
        h5file = openFile(filename + '.h5', mode='w', title='Raman fiber laser')
        fgroup = h5file.createGroup('/', 'forward', 'forward propagating powers')
        bgroup = h5file.createGroup('/', 'backward', 'backward propagating powers')

        try:
            for i in xrange(self.castella.n):
                if i == 0:
                    ordinal = str(1) + 'st'
                if i == 1:
                    ordinal = str(2) + 'nd'
                if i == 2:
                    ordinal = str(3) + 'rd'
                else:
                    ordinal = str(i + 1) + 'th'
                
            h5file.createArray(fgroup, 'F' + str(i+1), self.castella.F[i], ordinal + ' forward propagating powers')
            h5file.createArray(bgroup, 'B' + str(i+1), self.castella.B[i], ordinal + ' backward propagating powers')
            
        except TypeError:
            print "This version of PyTables does not support NumPy array."

        h5file.close()
        
    def write_csv(self, filename):
        """write forward and backward propagating powers.
        each columns contains Fs and Bs in order.
        """
        from csv import writer
        csvwriter = writer(open(filename+'.csv', 'wb')) 
        
        for i in xrange(self.castella.F.shape[1]):
            row = self.castella.F[:, i].tolist() + self.castella.B[:, i].tolist()
            csvwriter.writerow(row)
         
    def write_diff_csv(self, filename):
        """write difference of forward and backward propagating powers.
        each columns contains F - B in order.
        """
        from csv import writer
        csvwriter = writer(open(filename+'.csv', 'wb')) 
        
        diff = self.castella.F - self.castella.B
        for i in xrange(diff.shape[1]):
            row = diff[:, i].tolist()
            csvwriter.writerow(row)


class Raman:
    def __init__(self, L, R0, RL, G, alpha, steps):
        if len(R0) != len(RL):
            raise ValueError, "R0 and RL should have the same length."
        self.n = len(R0)
        self.P = float(R0[0])

        self.R0 = array(R0, 'd')
        self.RL = array(RL, 'd')
        self.alpha = array(alpha[:self.n], 'd')
        
        self.G = matrix(G, 'd')
        self.G = self.G[:self.n, :self.n]
        print "Det(G):", det(G)
        if len(self.G.shape) != 2 or self.G.shape[0] != self.G.shape[1] != self.n:
            raise ValueError, "G should be a square matrix."

        self.L = float(L)
        self.h = self.L / float(steps)
        array_size = steps + 1
        
        self.u = zeros((self.n, array_size), 'd')
        
        # Set the initial conditions, but u[0, 0] can't not be set. 
        # Because, c didn't set yet.
        self.u[1:, 0] = .5 * log(self.R0[1:])
        self.u[:, -1] = -.5 * log(self.RL)
        
        # forward propagating powers
        self.F = zeros((self.n, array_size), 'd')
        # backward propagating powers
        self.B = zeros((self.n, array_size), 'd')
        
        # c = FB
        self.c = zeros(self.n, 'd')
    
    def get_L1_norm(self):
        return map(lambda x: self.h * sum(cosh(x)), self.u)
        
    def set_FB(self):
        for i in xrange(self.n):
            self.F[i, ] = sqrt(self.c[i]) * exp(self.u[i, ])
            self.B[i, ] = sqrt(self.c[i]) * exp(-self.u[i, ])

    def saturate_evolve(self, error):
        n = 1
        while True:
            old_u = array(self.u)
            self.evolve()
            error_rate = fabs(old_u - self.u).max() / self.u.max()
            print "iteration:", n, "error rate:", error_rate
            if error > error_rate: break
            else: n += 1

        self.set_c()
        self.set_FB()
    
    def evolve_steps(self, n):
        for i in xrange(n):
            old_u = array(self.u)
            self.evolve()
            error_rate = fabs(old_u - self.u).max() / self.u.max()
            print "iteration:", i + 1, "error rate:", error_rate
     
        self.set_c()
        self.set_FB()
        
        
class Even(Raman):
    def __init__(self, L, R0, RL, G, alpha, steps):
        print "Starting an even case..."
        Raman.__init__(self, L, R0, RL, G, alpha, steps)
        
    def get_Rin(self):
        return (2 * self.P / self.q[0] * self.L1_norm[0])**2

    def get_mu(self):
        return -.5 * log(self.RL * self.R0) + self.alpha * self.L
    
    def get_q(self):
        return dot(inv(self.G), self.mu)
    
    def set_c(self):
        print "mu:", self.mu
        print "q:", self.q 
        
        for q in self.q:
            if q < 0:
                raise ValueError, "q < 0 is detected."
           
        self.c = (.5 * self.q / self.L1_norm)**2

        print "c:", self.c
        
    def evolve(self):
        self.L1_norm = self.get_L1_norm()
        self.mu = self.get_mu()
        self.q = self.get_q()
        self.R0[0] = self.get_Rin()
        self.u[0, 0] = .5 * log(self.R0[0])

        slope = zeros(self.u.shape, 'd')
        
        for i in xrange(self.n):
            for k in xrange(self.u.shape[1]):
                second_term = sum(array(self.G[i]).flatten() * self.q * cosh(self.u[:, k]) / self.L1_norm)
                slope[i, k] = -self.alpha[i] + second_term
                
        for j in xrange(self.u.shape[1] - 1):
            self.u[:, j+1] = slope[:, j] * self.h + self.u[:, j]

        
class Odd(Raman):
    def __init__(self, L, R0, RL, G, alpha, steps):
        print "Starting an odd case..."
        Raman.__init__(self, L, R0, RL, G, alpha, steps)
        
        # a.G=0
        partA = solve(self.G.getT()[1:, 1:], -array(self.G[0, 1:]).flatten())
        self.a = array([1] + partA.tolist())
        print "a:", self.a
        
        # b.G=0
        partB = solve(self.G[1:, 1:], -array(self.G[1:, 0],).flatten())
        self.b = array([1] + partB.tolist())
        print "b:", self.b
        
        self.R0[0] = self.get_R00()
        print "Rin:", self.R0[0]
        
        self.u[0, 0] = .5 * log(self.R0[0])
        self.mu = -.5 * log(self.RL*self.R0) + self.alpha * self.L
        print "mu:", self.mu
        
        # the particular solution of G.q=mu 
        q0_part = dot(pinv(self.G[:, :-1]), self.mu)
        self.q0 = array(q0_part.tolist() + [0])
        print "q0:", self.q0
        
    def get_R00(self):
        exponent = 2 * self.L * dot(self.a, self.alpha)
        tmp = (self.RL * self.R0)**-self.a
        return exp(exponent) / self.RL[0] * product(tmp[1:])
        
    def get_lambda(self):
        return 2 * self.P * self.L1_norm[0] / sqrt(self.R0[0]) - self.q0[0]
    
    def get_q(self):
        return self.q0 + self.l * self.b
    
    def set_c(self):
        print "lambda", self.l
        self.q = self.get_q()
        print "q:", self.q
        
        for q in self.q:
            if q < 0:
                raise ValueError, "q < 0 is detected."
            
        tmp1 = log(self.P) + .5 * log(self.RL[0]) - self.alpha[0] * self.L
        tmp2 = sum(self.a[1:] / self.a[0] * (.5 * log(self.RL[1:] * self.R0[1:]) - self.alpha[1:] * self.L))
        self.c[0] = exp(2 * (tmp1 + tmp2))
        
        self.c[1:] = (.5 * self.q[1:] / self.L1_norm[1:])**2
        
        print "c:", self.c
        
    def evolve(self):
        self.L1_norm = self.get_L1_norm()
        self.l = self.get_lambda()
        slope = zeros(self.u.shape, 'd')
        
        self.q = self.q0 + self.l * self.b
        
        for i in xrange(self.n):
            for k in xrange(self.u.shape[1]):
                second_term = sum(array(self.G[i], 'd').flatten() * self.q * cosh(self.u[:, k]) / self.L1_norm)
                slope[i, k] = -self.alpha[i] + second_term
                
        for j in xrange(self.u.shape[1] - 1):
            self.u[:, j+1] = slope[:, j] * self.h + self.u[:, j]
            
            
if __name__ == '__main__':
    L = 100
    P = 5
    R5out = 0.1
    R4out = 0.1
    R0 = [P, 0.99, 0.99, 0.99, 0.99]
    RL = [0.99, 0.99, 0.99, 0.99, R5out]
    G = 10**-3 * matrix([[0, -5.354693, -0.833641, -0.165746, -0.001215],
                       [5.109551, 0, -5.09133, -0.800871, -0.246770],
                       [0.757437, 4.847864, 0, -4.883841, -0.694188],
                       [0.143011, 0.724173, 4.637914, 0, -3.546259],
                       [0.001000, 0.212878, 0.628922, 3.383213, 0]])
    alpha = 10**-3 * array([0.388799, 0.346712, 0.296873, 0.252234, 0.218211])
    steps = 1000
    
    rfl = RamanFiberLaser(L, R0, RL, G, alpha, steps)
    rfl.run(n=10)
    rfl.write_diff_csv('test')
    rfl.show()
