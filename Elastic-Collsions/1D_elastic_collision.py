import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def momentum(m, v):
    u = np.zeros(2)
    
    u[0] = v[0]*(m[0]-m[1])/(m[0]+m[1]) + 2*m[1]*v[1]/(m[0]+m[1])
    u[1] = 2*m[0]*v[0]/(m[0]+m[1]) + v[1]*(m[1]-m[0])/(m[0]+m[1])

    return u

def collided(state, R):
    if np.diff(state, axis=0)[0,0] <= np.sum(R):
        return True
    
    return False

def simulate(state, m, R, h, steps):
    simulation = [state]
    state_next = np.copy(state)
    for i in range(steps):
       
        if collided(state_next, R):
            state_next[:,2] = momentum(m, state[:,2])

        state_next[0,0] += state_next[0,2] * h
        state_next[1,0] += state_next[1,2] * h
        simulation.append(np.copy(state_next))

    return simulation

def main():
    DT = 0.001
    SIM_LEN = 100

    mass1 = 5
    velocity1 = 5
    mass2 = 1
    velocity2 = -40

    state = np.zeros((2,3))
    v = np.array([velocity1, velocity2])
    m = np.array([mass1, mass2])
    R = (m/sum(m))/10

    state[0] = [R[0], 0.5, velocity1]
    state[1] = [1-R[1], 0.5, velocity2]


    u = np.array(momentum(m, state[:,2]))
    print("m0: {:3.2f}[kg] | v0: {:3.2f}[m/s] | u0: {:3.2f}[m/s]".format(m[0], v[0], u[0]))
    print("m1: {:3.2f}[kg] | v1: {:3.2f}[m/s] | u1: {:3.2f}[m/s]".format(m[1], v[1], u[1]))

    simulation = simulate(state, m, R, DT, SIM_LEN)

    fig, ax = plt.subplots()
    partical1 = plt.Circle(state[0], R[0])
    partical2 = plt.Circle(state[1], R[1])
    ax.add_patch(partical1)
    ax.add_patch(partical2)

    def animate_func(frame):
        partical1.center = simulation[frame][0]
        partical2.center = simulation[frame][1]
        return partical1, partical2
    
    animate = animation.FuncAnimation(fig, animate_func, frames=SIM_LEN, interval=40)
    
    plt.show()

if __name__ == "__main__":
    main()