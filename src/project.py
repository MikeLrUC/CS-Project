import simcx
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.legend import Legend


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
    def __init__(self, sim: MySimulatorMultipleInitStates, width = 1500, height = 800):
        '''
        sim: a MySimulatorMultipleInitStates simulation.
        width: width of the figure.
        height: height of the figure.
        '''
        super(MySimulatorMultipleInitStatesVisual, self).__init__(sim, width = width, height = height)
        self.ax = self.figure.add_subplot(111)
        self.lines = []
        line_styles = ['-', '--']
        for i in range(len(self.sim.y)):
            for e in range(len(self.sim.y[i])):
                line, = self.ax.plot(self.sim.x, self.sim.y[i][e], line_styles[e])
                self.lines.append(line)
        
        self.ax.set_title('Orbits')
        labels = []
        [labels.extend(pair) for pair in [[f'x0_{i}', f'y0_{i}'] for i in range(int(len(self.lines) / 2))]]
        leg = Legend(self.ax, self.lines, labels)
        self.ax.add_artist(leg);

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
    def __init__(self, sim : OrbitDifference, plot_orbits = False, width = 1500, height = 800):
        '''
        sim: an OrbitDifference simulation.
        plot_orbits: of false it only plots the difference lines, else it also plot the orbits of the two simulations beig compared.
        width: width of the figure.
        height: height of the figure.
        '''
        super(OrbitDifferenceVisual, self).__init__(sim, width = width, height = height)
        self.plot_orbits = plot_orbits
        self.ax1 = self.figure.add_subplot(1, 2, 1)
        self.ax2 = self.figure.add_subplot(1, 2, 2)
        self.lines1 = []
        self.lines2 = []
        # plot the difference lines
        for i in range(len(self.sim.y)):
            line, = self.ax1.plot(self.sim.x, self.sim.y[i], '-')
            self.lines1.append(line)
        
        # plot the orbits
        for i in range(len(self.sim.orbits)):
            for e in range(len(self.sim.orbits[i])):
                line, = self.ax2.plot(self.sim.x, self.sim.orbits[i][e], '-')
                self.lines2.append(line)
            
        # set the titles etc.
        self.ax1.set_title('Orbit Difference')
        self.ax2.set_title('Orbits')

        # add legend to the figure
        leg1 = Legend(self.ax1, self.lines1, ['difference x', 'difference y'])
        self.ax1.add_artist(leg1)
        leg2 = Legend(self.ax2, self.lines2, ['x0_0', 'y0_0', 'x0_1', 'y0_1'])
        self.ax2.add_artist(leg2)

    def draw(self):
        for i in range(len(self.sim.y)):
            self.lines1[i].set_data(self.sim.x, self.sim.y[i])
        
        line_counter = 0
        for i in range(len(self.sim.orbits)):
            for e in range(len(self.sim.orbits[i])):
                self.lines2[line_counter].set_data(self.sim.x, self.sim.orbits[i][e])
                line_counter += 1

        self.ax1.relim()
        self.ax2.relim()
        self.ax1.autoscale_view()
        self.ax2.autoscale_view()



class MyPhaseSpace2DMultipleInitialStates(simcx.MplVisual):
    '''
    Visual for the phase space.
    '''
    def __init__(self, sim: MySimulatorMultipleInitStates, name_x, name_y, width = 1000, height = 800):
        '''
        sim: a MySimulatorMultipleInitStates.
        name_x: label for the x axis.
        name_y: label for the y axis.
        width: width of the figure.
        height: height of the figure.
        '''
        super(MyPhaseSpace2DMultipleInitialStates, self).__init__(sim, width = width, height = height)
        self.ax = self.figure.add_subplot(111)
        self.lines = []
        for i in range(len(self.sim.y)):
            line, = self.ax.plot(self.sim.y[i][0], self.sim.y[i][1], '-') # var 1 in the x axis and var 2 in the y axis
            self.lines.append(line)

        self.ax.set_title(f'Phase Space\n{len(self.sim.state)} initial states')
        self.ax.set_xlabel(name_x)
        self.ax.set_ylabel(name_y)
    
    def draw(self):
        for i in range(len(self.sim.y)):
            self.lines[i].set_data(self.sim.y[i][0], self.sim.y[i][1])  # var 1 in the x axis and var 2 in the y axis
        self.ax.relim()
        self.ax.autoscale_view()


class MyFinalStateIterator(simcx.simulators.Simulator):
    '''
    Only tries the func with different parameters starting from a specific seed (initial state).
    Only works with functions that have two parameters to define.
    '''
    def __init__(self, func, seed, start, end, Dt, integration_method, discard=1000, samples=250,
                 delta=0.01):
        '''
        func: function that we want to test. It receives the parameters that we want to test and returns a function with thos parameters specified.
        seed: initial state.
        start: list with the values from where we want to start each parameter that we want to test ([a = 0.1, b = 0.2] for example)
        end: list with the values until where we want to test each parameter ([a = 10, b = 10] for example).
        Dt: delta used for the numerical integration method.
        integration_method: a function that, given the previous state, returns the new state using some numerical integration method.
        discard: the number of samples that we want to discard at the begining of the simulation.
        samples: the number of samples that we actually want to use from the simulation (the number of samples that we want after we discard the first n samples).
        delta: list with the deltas used to get the next parameters values.
        
        Note
        The discard value needs to take into account that, if the initial state is far from the fixed point, it will need to be a big value.
        '''
        super(MyFinalStateIterator, self).__init__()

        self._func = func
        self._seed = seed # initial state
        self._a = start[:] # variable that keeps track of the func parameter that we are testing
        self.start = start
        self.end = end
        self._discard = discard
        self._samples = samples
        self._delta = delta
        self.integration_method = integration_method
        self.state = seed[:] # keep track of the current state
        self.Dt = Dt

        self.x = [np.zeros(self._samples) for _ in range(len(start))]
        self.y = [np.zeros(self._samples) for _ in range(len(seed))]

    def new_parameters(self):
        '''
        Used to get the parameters combinations.
        '''
        new_a = self._a[:]
        if self._a[1] <= self.end[1]: # se o b ainda nao chegou ao fim, incrementar
            new_a[1] += self._delta[1]
        else: # se o parametro b chegou ao final, incrementar o a e resetar o b
            new_a[0] += self._delta[0]
            new_a[1] = self.start[1]
        self._a = new_a

    def step(self, delta=0):
        # per step it calculates the final state for all the simulations using "all" the possible values for a certain parameter
        if self._a <= self.end: # verify if we reached the end of the simulation
            # perform simulation for the self._a parameters
            self.state = self._seed # initial state
            func = self._func(*self._a) # get the function with the correct parameters
            for _ in range(self._discard):
                self.state = self.integration_method(func, self.state, self.Dt)
            for i in range(self._samples):
                self.state = self.integration_method(func, self.state, self.Dt)
                # save the function result values
                for e in range(len(self.state)):
                    self.y[e][i] = self.state[e]
                # save the parameters values
                for e in range(len(self.start)):
                    self.x[e][i] = self._a[e]
            
            # get the next parameters that we want to test the functio with
            self.new_parameters()


class Bifurcation3DVisual(simcx.MplVisual):
    '''
    Visual for the Bifurcation diagram in 3D.
    '''
    def __init__(self, sim: MyFinalStateIterator, name_x, name_y, name_z, width = 1000, height = 800):
        '''
        sim: a FinalStateIterator.
        name_x: label for the x axis.
        name_y: label for the y axis.
        name_z: label for the z axis.
        width: width of the figure.
        height: height of the figure.
        '''
        super(Bifurcation3DVisual, self).__init__(sim, width = width, height = height)
        self.ax = self.figure.add_subplot(111, projection = '3d')
        # self.ax.view_init(20, 20)

        self.ax.set_xlim(sim.start[0], sim.end[0])
        self.ax.set_ylim(sim.start[1], sim.end[1])

        self.ax.set_title(f"Bifurcation\nx0: {self.sim._seed}, deltas: {self.sim._delta}")
        self.ax.set_xlabel(name_x)
        self.ax.set_ylabel(name_y)
        self.ax.set_zlabel(name_z)

        # add legend to the figure
        blue_patch = mpatches.Patch(color='blue', label='x')
        orange_patch = mpatches.Patch(color='orange', label='y')
        self.ax.legend(handles=[blue_patch, orange_patch])
        
    
    def draw(self):
        self.ax.scatter(self.sim.x[0], self.sim.x[1], self.sim.y[0], c = 'blue')
        self.ax.scatter(self.sim.x[0], self.sim.x[1], self.sim.y[1], c = 'orange')
        

if '__main__' == __name__:

    # set the rng seed
    np.random.seed(0)
    
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
    # x0 = [-10, -10]
    # x0 = [-10, 10]
    # x0 = [10, -10]
    # x0 = [10, 10]
    # x0 = [10, 0]
    # x0 = [0, 10]
    # x0 = [0, -10]

    # define a and b
    a_ = a[1]
    b_ = b[0]

    # func = brusselator(a[0], b[1])
    func = brusselator(a_, b_)
    Dt = 0.01


    distance = 1 # distance from the fixed point, from which we want to generate random values
    n_init_states = 3 # number of different initial states
    initial_states = [[np.random.uniform(a_ - distance, a_ + distance), np.random.uniform(b_ / a_ - distance, b_ / a_ + distance)] for _ in range (n_init_states)]
    # print(initial_states)


    # sim = MySimulator(func, x0, Dt, NumericalIntegration.euler)
    # sim = MySimulator(func, x0, Dt, NumericalIntegration.runge_kutta)
    # sim = OrbitDifference(func, [[0, 0], [0.1, 0.1]], Dt, NumericalIntegration.euler)
    # sim = MySimulatorMultipleInitStates(func, [[0, 0], [0.05, 0.05], [0.1, 0.1]], Dt, NumericalIntegration.euler)
    # sim = MySimulatorMultipleInitStates(func, [[i * 0, i * 0.1] for i in range(25)], Dt, NumericalIntegration.euler)
    sim = MySimulatorMultipleInitStates(func, initial_states, Dt, NumericalIntegration.euler)
    # sim = MySimulatorMultipleInitStates(func, [[0, 0]], Dt, NumericalIntegration.euler)
    # sim = MyFinalStateIterator(brusselator, x0, [0.1, 0.1], [1, 1], Dt, NumericalIntegration.euler, delta = [0.05, 0.05], discard = 15000, samples = 2)

    
    # vis = simcx.visuals.Lines(sim)
    # vis = OrbitDifferenceVisual(sim, True)
    # vis = MySimulatorMultipleInitStatesVisual(sim)

    vis = MyPhaseSpace2DMultipleInitialStates(sim, 'x', 'y')
    # vis = Bifurcation3DVisual(sim, 'a', 'b', 'function_output')
    
    display = simcx.Display()
    display.add_visual(vis)
    display.add_simulator(sim)
    simcx.run()

    print(f'[final values]  x: {sim.y[0][-1]}, y: {sim.y[1][-1]}')