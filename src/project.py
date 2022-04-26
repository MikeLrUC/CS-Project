#
#       Created on Thu Apr 14 2022 19:16:01
#       Author: Gabriel Fernandes
#       email: gabrielf@student.dei.uc.pt
#

import simcx
import numpy as np


# differential equations
def brusselator(a, b):
    '''
    a > 0 e b > 0
    '''
    def brusselator_inner(x, y):
        return a - (b + 1) * x + x**2 * y, b * x - x**2 * y
    return brusselator_inner

# integration method?
# euler?
# runge-kutta4?
class EulerSimulator(simcx.simulators.Simulator):
    def __init__(self, func, init_state, Dt):
        '''
        func -> function that returns the derivatives given the previous state
        init_state -> [x1, x2]
        Dt -> delta used for the numerical integration: https://en.wikipedia.org/wiki/Numerical_methods_for_ordinary_differential_equations
        '''
        self.__func = func
        self.Dt = Dt
        self.x = [0]
        self.y = [[init_state[0]], [init_state[1]]]
        self.state = init_state
    
    def step(self, delta = 0):
        vals = self.__func(*self.state) # get the derivatives
        self.x += [self.x[-1] + self.Dt] # increment step
        self.state = []
        for i in range(len(self.y)):
            self.y[i] += [self.y[i][-1] + self.Dt * vals[i]] # calculate new values [previous + Dt * new]
            self.state.append(self.y[i][-1]) # update state
    
    def reset(self):
        self.state = [y[0] for y in self.y]
        # self.time = 0
        self.x = [0]
        self.y = [[state] for state in self.state]
    

class RungeKuttaSimulator(simcx.simulators.Simulator):
    def __init__(self, func, init_state, Dt):
        '''
        func -> function that returns the derivatives given the previous state
        init_state -> [x1, x2]
        Dt -> delta used for the numerical integration: https://en.wikipedia.org/wiki/Numerical_methods_for_ordinary_differential_equations
        '''
        self.__func = func
        self.Dt = Dt
        self.x = [0]
        self.y = [[init_state[0]], [init_state[1]]]
        self.state = init_state
    
    def step(self, delta = 0):
        vals = self.runge_kutta_4() # get the derivatives
        self.x += [self.x[-1] + self.Dt] # increment step
        self.state = []
        for i in range(len(self.y)):
            self.y[i] += [self.y[i][-1] + vals[i]] # calculate new values [previous + Dt * new]
            self.state.append(self.y[i][-1]) # update state
    
    def reset(self):
        self.state = [y[0] for y in self.y]
        # self.time = 0
        self.x = [0]
        self.y = [[state] for state in self.state]
    
    def runge_kutta_4(self):
        '''
        performs runge kutta of the fourth order and returns the value that needs to be added to the previous state per state component.
        '''
        temp = self.__func(*self.state)
        k1 = np.array([self.Dt * elem for elem in temp])
        temp = self.__func(*(self.state + k1 / 2))
        k2 = np.array([self.Dt * elem for elem in temp])
        temp = self.__func(*(self.state + k2 / 2))
        k3 = np.array([self.Dt * elem for elem in temp])
        temp = self.__func(*(self.state + k3))
        k4 = np.array([self.Dt * elem for elem in temp])

        vals = []
        for i in range(len(self.state)):
            temp = (1 / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
            vals.append(temp)
        return vals


# testar varios a e b


if '__main__' == __name__:

    x0 = [0.5, 1] # initial state
    
    # test various a's and b's
    a = [i * 0.5 for i in range(10)]
    b = [i * 0.5 for i in range(10)]

    func = brusselator(a[0], b[0])
    Dt = 0.01

    # sim = EulerSimulator(func, x0, Dt)
    sim = RungeKuttaSimulator(func, x0, Dt)
    vis = simcx.visuals.Lines(sim)
    
    display = simcx.Display()
    display.add_simulator(sim)
    display.add_visual(vis)
    simcx.run()