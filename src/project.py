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

class NumericalIntegration():
    '''
    class with the numerical integration methods.
    '''
    def euler(func, state, Dt):
        '''
        func: function that gives us the derivatives.
        state: list containing two lists with the initial states that we want to compare.
        Dt: delta used for the numerical integratio method.

        returns the new state given the previous one using the euler method.
        '''
        derivatives = func(*state)
        new_state = [state[i] + Dt * derivatives[i] for i in range(len(state))]
        return new_state
    

    def runge_kutta(func, state, Dt):
        '''
        func: function that gives us the derivatives.
        state: list containing two lists with the initial states that we want to compare.
        Dt: delta used for the numerical integratio method.

        returns the new state given the previous one using runge kutta of order 4.
        '''
        temp = func(*state)
        k1 = np.array([Dt * elem for elem in temp])
        temp = func(*(state + k1 / 2))
        k2 = np.array([Dt * elem for elem in temp])
        temp = func(*(state + k2 / 2))
        k3 = np.array([Dt * elem for elem in temp])
        temp = func(*(state + k3))
        k4 = np.array([Dt * elem for elem in temp])

        new_state = [state[i] + (1 / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(len(state))]
        return new_state


class MySimulator(simcx.simulators.Simulator):
    '''
    Simulation with one initial state (use simcx.visuals.Lines(...) to see the orbit).
    '''
    def __init__(self, func, init_state, Dt, integration_method):
        '''
        func: function that returns the derivatives given the previous state
        init_state: [x1, x2]
        Dt: delta used for the numerical integration: https://en.wikipedia.org/wiki/Numerical_methods_for_ordinary_differential_equations
        integration_method: a function that, given the previous state, returns the new state using some numerical integration method.

        Simulation of a orbit given an initial state.
        '''
        self.__func = func
        self.Dt = Dt
        self.x = [0]
        self.y = [[init_state[0]], [init_state[1]]]
        self.state = init_state
        self.integration_method = integration_method
    
    def step(self, delta = 0):
        self.state = self.integration_method(self.__func, self.state, self.Dt)
        self.x += [self.x[-1] + self.Dt] # increment step
        
        for i in range(len(self.y)):
            self.y[i] += [self.state[i]] # save the value of the new state in the respective y list so that we have the values to plot the orbits.
            print(f'y[{i}]: {self.y[i][-1]}')
    
    def reset(self):
        self.state = [y[0] for y in self.y]
        self.x = [0]
        self.y = [[state] for state in self.state]


class MySimulatorMultipleInitStates(simcx.simulators.Simulator):
    '''
    Simulation with multiple initial states (use MySimulatorMultipleInitStatesVisual(...) to see the orbits).
    Same as MySimulator if the initial state is [[component_0, ..., component_n]] (only one intial state with n components).
    '''
    def __init__(self, func, init_states, Dt, integration_method):
        '''
        func: function that gives us the derivatives.
        init_state: list containing lists with the initial states that we want to simulate.
        Dt: delta used for the numerical integratio method.
        integration_method: a function that, given the previous state, returns the new state using some numerical integration method.

        Simulation of the orbits given the initial states (n initial states).
        Same as MySimulator if the initial state is [[component_0, ..., component_n]] (only one intial state with n components).
        '''
        self.__func = func
        self.Dt = Dt
        self.x = [0]
        self.y = [[[state_component] for state_component in init_state] for init_state in init_states] # the orbits
        self.state = init_states
        self.integration_method = integration_method
    
    def step(self, delta = 0):
        self.state = [self.integration_method(self.__func, state_, self.Dt) for state_ in self.state]
        self.x += [self.x[-1] + self.Dt] # increment step
        
        for i in range(len(self.y)):  # go through the different orbits (different initial states)
            for e in range(len(self.y[i])): # go through the different components of each orbit  (different initial states)
                self.y[i][e] += [self.state[i][e]] # save the value of the new state in the respective y list so that we have the values to plot the orbits.

    def reset(self):
        self.state = [[orbit_component[0] for orbit_component in orbit] for orbit in self.y]
        self.y = [[[state_component] for state_component in state_] for state_ in self.state]
        self.x = [0]


class MySimulatorMultipleInitStatesVisual(simcx.MplVisual):
    '''
    Visual for simulation created with MySimulatorMultipleInitStates.
    '''
    def __init__(self, sim: MySimulatorMultipleInitStates):
        super(MySimulatorMultipleInitStatesVisual, self).__init__(sim)
        self.ax = self.figure.add_subplot(111)
        self.lines = []
        for i in range(len(self.sim.y)):
            for e in range(len(self.sim.y[i])):
                line, = self.ax.plot(self.sim.x, self.sim.y[i][e], '-')
                self.lines.append(line)

    def draw(self):
        line_counter = 0
        for i in range(len(self.sim.y)):
            for e in range(len(self.sim.y[i])):
                self.lines[line_counter].set_data(self.sim.x, self.sim.y[i][e])
                line_counter += 1
        self.ax.relim()
        self.ax.autoscale_view()


class OrbitDifference(simcx.simulators.Simulator):
    '''
    Simulation of the difference between two orbits (use OrbitDifferenceVisual() to see the difference with ot without the orbits).
    '''
    def __init__(self, func, init_states, Dt, integration_method):
        '''
        func: function that gives us the derivatives.
        init_state: list containing two lists with the initial states that we want to compare.
        Dt: delta used for the numerical integratio method.
        integration_method: a function that, given the previous state, returns the new state using some numerical integration method.

        Simulation of the differences between the orbits of two different initial states.
        '''
        self.__func = func
        self.Dt = Dt
        self.x = [0]
        self.orbits = [[[state_component] for state_component in init_state] for init_state in init_states] # the orbits
        self.y = [[init_states[1][i] - init_states[0][i]] for i in range(len(init_states[0]))] # difference between the two orbits
        print(self.y)
        self.state = init_states
        self.integration_method = integration_method
    
    def step(self, delta = 0):
        self.state = [self.integration_method(self.__func, state_, self.Dt) for state_ in self.state]
        self.x += [self.x[-1] + self.Dt] # increment step
        
        # save the values of each orbit
        for i in range(len(self.orbits)):  # go through the different orbits (different initial states)
            for e in range(len(self.orbits[i])): # go through the different components of each orbit  (different initial states)
                self.orbits[i][e] += [self.state[i][e]] # save the value of the new state in the respective y list so that we have the values to plot the orbits.
        
        # calculate the differences between orbits
        for i in range(len(self.y)):
            self.y[i] += [self.state[1][i] - self.state[0][i]]

    def reset(self):
        self.state = [[orbit_component[0] for orbit_component in orbit] for orbit in self.orbits]
        self.orbits = [[[state_component] for state_component in state_] for state_ in self.state]
        self.x = [0]
        self.y = [[self.state[1][i] - self.state[0][i]] for i in range(len(self.state[0]))] # difference between the two orbits


class OrbitDifferenceVisual(simcx.MplVisual):
    '''
    Visual for the OrbitDifference simulation.
    '''
    def __init__(self, sim : OrbitDifference, plot_orbits = False):
        '''
        sim: an OrbitDifference simulation.
        plot_orbits: of false it only plots the difference lines, else it also plot the orbits of the two simulations beig compared.

        The difference lines are ploted with an alpha of 0.5 and the orbit lines with an alpha of 1 (to differentiate between the two types of lines).
        '''
        super(OrbitDifferenceVisual, self).__init__(sim)
        self.plot_orbits = plot_orbits
        self.ax = self.figure.add_subplot(111)
        self.lines = []
        for i in range(len(self.sim.y)):
            line, = self.ax.plot(self.sim.x, self.sim.y[i], '-', alpha=0.5)
            self.lines.append(line)
        
        if self.plot_orbits:
            for i in range(len(self.sim.orbits)):
                for e in range(len(self.sim.orbits[i])):
                    line, = self.ax.plot(self.sim.x, self.sim.orbits[i][e], '-')
                    self.lines.append(line)

    def draw(self):
        line_counter = 0
        for i in range(len(self.sim.y)):
            self.lines[i].set_data(self.sim.x, self.sim.y[i])
            line_counter += 1
        
        if self.plot_orbits:
            for i in range(len(self.sim.orbits)):
                for e in range(len(self.sim.orbits[i])):
                    self.lines[line_counter].set_data(self.sim.x, self.sim.orbits[i][e])
                    line_counter += 1

        self.ax.relim()
        self.ax.autoscale_view()

# testar varios a e b


if '__main__' == __name__:
    
    # test various a's and b's
    a = [i * 0.5 for i in range(1, 11)]
    b = [i * 0.5 for i in range(1, 11)]

    x0 = [0.5, 1] # initial state
    # x0 = [0.5, 0.8] # initial state
    # x0 = [0.5, 0.5] # initial state
    x0 = [0, 0] # initial state
    # x0 = [a[2], b[2] / a[2]]
    # x0 = [0, 2.3] # initial state
    # x0 = [0, 1] # initial state
    # x0 = [-0.1, 0.2]
    # x0 = [-0.1, -0.2]

    # func = brusselator(a[0], b[1])
    func = brusselator(a[2], b[0])
    Dt = 0.01

    # sim = MySimulator(func, x0, Dt, NumericalIntegration.euler)
    # sim = MySimulator(func, x0, Dt, NumericalIntegration.runge_kutta)
    # sim = OrbitDifference(func, [[0, 0], [0.1, 0.1]], Dt, NumericalIntegration.euler)
    sim = MySimulatorMultipleInitStates(func, [[0, 0], [0.05, 0.05], [0.1, 0.1]], Dt, NumericalIntegration.euler)
    # sim = MySimulatorMultipleInitStates(func, [[0, 0]], Dt, NumericalIntegration.euler)
    
    # vis = simcx.visuals.Lines(sim)
    # vis = OrbitDifferenceVisual(sim, False)
    vis = MySimulatorMultipleInitStatesVisual(sim)
    
    display = simcx.Display()
    display.add_visual(vis)
    display.add_simulator(sim)
    simcx.run()

    print(f'[final values]  x: {sim.y[0][-1]}, y: {sim.y[1][-1]}')