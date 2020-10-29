import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class TwoLinkSimulator:
    def __init__(self):
        self.l1 = 1 # length of joint i
        self.l2 = 2
        self.lc1 = 0.5  # COM (center of mass) of joint i
        self.lc2 = 1
        self.m1 = 1 # mass for joint i
        self.m2 = 1
        self.Ic1 = 0.083 #
        self.Ic2 = 0.33
        self.I1 = self.Ic1 + self.m1 * self.lc1 ** 2
        self.I2 = self.Ic2 + self.m2 * self.lc2 ** 2
        self.b1 = 0.1  # damping for joint i
        self.b2 = 0.1
        self.B = np.array([[1, 0], [0, 1]]) # controlability matrix
        self.g = 9.81 # gravity constant
        self.dt = 0.01 # simulated delta t
        self.MAX_VEL_1 = 100 # maximum velocity for joint i
        self.MAX_VEL_2 = 100

        # # model disturbance
        # self.wm = 1
        # self.wc = 1
        # self.wg = 1
        # self.wf = 1

        self.plot2realRATIO = 2 #  we want to plot the animation at 10 Hz


        self.INI_STATE = [pi, pi, 0, 0]
        self.state = self.INI_STATE
        self.state_history = [self.state] # a buffer to store states history




    def _dynamics_matrices(self, x):
        # INPUTS:
        #    model: struct
        #    x: [4,1] = [q1 q2 q1d q2d]
        #
        # OUTPUTS:
        #    M: [2,2] = inertia array
        #    C: [2,2] = coriolis and centrifugal terms
        #    G: [2,1] = gravitational terms
        #    F: [2,1] = Fiction force terms
        #    dGdq: [2,2] = partial G / partial q

        g = self.g

        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        lc1 = self.lc1
        l2 = self.l2
        lc2 = self.lc2
        b1 = self.b1
        b2 = self.b2
        I1 = self.I1
        I2 = self.I2

        q = x[0:2, :]
        qd = x[2:4, :]

        c = cos(q[0:2, :])
        s = sin(q[0:2, :])
        s12 = sin(q[0, :] + q[1, :])

        m2l1lc2 = m2 * l1 * lc2  # occurs often!

        ### Find M(q), inertia array
        M11 = I1 + I2 + m2 * l1 ** 2 + 2 * m2l1lc2 * c[1]
        M12 = I2 + m2l1lc2 * c[1]
        M21 = M12
        M22 = I2
        M = np.array([[M11.item(), M12.item()], [M21.item(), M22]])

        ## Find C(q,qd), coriolis and centrifugal terms
        C11 = -2 * m2l1lc2 * s[1] * qd[1]
        C12 = -m2l1lc2 * s[1] * qd[1]
        C21 = m2l1lc2 * s[1] * qd[0]
        C22 = 0

        C = np.array([[C11.item(), C12.item()], [C21.item(), C22]])

        ## Find G(q), gravitational terms
        G1 = g * (m1 * lc1 * s[0] + m2 * (l1 * s[0] + lc2 * s12))
        G2 = g * m2 * lc2 * s12
        G = np.array([[G1.item()], [G2.item()]])

        ## Find F, Fiction force terms
        F = np.array([[b1, 0], [0, b2]])
        B = self.B

        return M, C, G, F, B

    def _dynamics(self, x, u):
        # % INPUTS:
        # %    x: [4,1] = [q1 q2 q1d q2d]
        # %    u: scalar = input torque
        # %
        # % OUTPUTS:
        # %    xdot: [4,1] = [q1d q2d q1dd q2dd]
        q = x[0:2, :]
        qd = x[2:4, :]

        M, C, G, F, B = self._dynamics_matrices(x)

        inv_M = np.linalg.inv(M)
        A_bar = inv_M.dot(-C.dot(qd) - G - F.dot(qd))
        B_bar = inv_M.dot(B)

        qdd = A_bar + B_bar.dot(u)

        xdot = np.concatenate((qd, qdd), axis=0)
        return xdot

    def _dsdt(self, s_augmented):
        if len(s_augmented.shape) != 1:
            raise ValueError('s_augmented should be a 1 dimensional numpy array')
        x = s_augmented[0:4]
        u = s_augmented[4:6]

        # dynamics equation function will have input vectors and output vectors
        xdot = simulator._dynamics(x.reshape(-1, 1), u.reshape(-1, 1))
        return np.concatenate((xdot.reshape(-1), np.zeros(2)))

    def step(self, torque):
        if not isinstance(torque, (list, tuple)) and len(torque) == 2:
            raise ValueError('torque arguement should be a list: [0, 0.1] or a tuple: (0, 0.1)')
        s = np.array(self.state)
        a = np.array(torque)
        s_augmented = np.concatenate((s, a), axis=0)
        ns = self._rk4(self._dsdt, s_augmented, [0, self.dt])
        ns = ns[-1]
        ns = ns[:4]  # omit action

        ns[0] = self._wrap(ns[0], -pi, pi)
        ns[1] = self._wrap(ns[1], -pi, pi)
        ns[2] = self._bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = self._bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns.tolist()
        self.state_history.append(self.state)

        return self.state

    def reset(self):
        self.state = self.INI_STATE
        self.state_history = [self.state] # a buffer to store states history

    def _rk4(self, derivs, y0, t, *args, **kwargs):
        """ copy from gym

        Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
        This is a toy implementation which may be useful if you find
        yourself stranded on a system w/o scipy.  Otherwise use
        :func:`scipy.integrate`.
        Args:
            derivs: the derivative of the system and has the signature ``dy = derivs(yi, ti)``
            y0: initial state vector
            t: sample times
            args: additional arguments passed to the derivative function
            kwargs: additional keyword arguments passed to the derivative function
        Example 1 ::
            ## 2D system
            def derivs6(x,t):
                d1 =  x[0] + 2*x[1]
                d2 =  -3*x[0] + 4*x[1]
                return (d1, d2)
            dt = 0.0005
            t = arange(0.0, 2.0, dt)
            y0 = (1,2)
            yout = rk4(derivs6, y0, t)
        Example 2::
            ## 1D system
            alpha = 2
            def derivs(x,t):
                return -alpha*x + exp(-t)
            y0 = 1
            yout = rk4(derivs, y0, t)
        If you have access to scipy, you should probably be using the
        scipy.integrate tools rather than this function.
        Returns:
            yout: Runge-Kutta approximation of the ODE
        """

        try:
            Ny = len(y0)
        except TypeError:
            yout = np.zeros((len(t),), np.float_)
        else:
            yout = np.zeros((len(t), Ny), np.float_)

        yout[0] = y0.reshape(-1)

        for i in np.arange(len(t) - 1):
            thist = t[i]
            dt = t[i + 1] - thist
            dt2 = dt / 2.0
            y0 = yout[i]

            k1 = np.asarray(derivs(y0, *args, **kwargs))
            k2 = np.asarray(derivs(y0 + dt2 * k1, *args, **kwargs))
            k3 = np.asarray(derivs(y0 + dt2 * k2, *args, **kwargs))
            k4 = np.asarray(derivs(y0 + dt * k3, *args, **kwargs))
            yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return yout

    def _wrap(self, x, m, M):
        """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
        truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
        For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
        Args:
            x: a scalar
            m: minimum possible value in range
            M: maximum possible value in range
        Returns:
            x: a scalar, wrapped
        """
        diff = M - m
        while x > M:
            x = x - diff
        while x < m:
            x = x + diff
        return x

    def _bound(self, x, m, M=None):
        """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
        have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
        Args:
            x: scalar
        Returns:
            x: scalar, bound between min (m) and Max (M)
        """
        if M is None:
            M = m[1]
            m = m[0]
        # bound x between min (m) and Max (M)
        return min(max(x, m), M)


    def plot_simulation(self):

        fig = plt.figure(figsize=(4, 4))
        lmin, lmax = (-self.l1-self.l2)*1.2, (self.l1+self.l2)*1.2
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(lmin, lmax), ylim=(lmin, lmax))
        ax.grid(False)

        line, = ax.plot([], [], 'o-', markerfacecolor='black', markeredgecolor='black', lw=4, mew=5)
        plt.xticks([], [])
        plt.yticks([], [])

        #time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def position(state):
            """Compute x,y position of the hand"""

            x = np.cumsum([0,
                           self.l1 * np.sin(state[0]),
                           self.l2 * np.sin(state[1])])
            y = np.cumsum([0,
                           -self.l1 * np.cos(state[0]),
                           -self.l2 * np.cos(state[1])])
            return (x, y)

        def init():
            """initialize animation"""
            line.set_data([], [])
            #time_text.set_text('')
            return (line, )

        def animate(i):
            state = self.state_history[i*self.plot2realRATIO]
            line.set_data(position(state))
            #time_text.set_text('time = %.2f' % arm.time_elapsed)
            return (line,)

        # frames=None for matplotlib 1.3
        interval = self.dt * 1000 * self.plot2realRATIO;


        ani = animation.FuncAnimation(fig, animate, frames=len(self.state_history)//self.plot2realRATIO,
                                      interval=interval, blit=True,
                                      init_func=init)

        # uncomment the following line to save the video in mp4 format.
        # requires either mencoder or ffmpeg to be installed
        # ani.save('2linkarm.mp4', fps=15,
        #         extra_args=['-vcodec', 'libx264'])

        plt.show()

        return ani

if __name__ == '__main__':
    simulator = TwoLinkSimulator()
    duration = 20
    for i in range(int(duration/simulator.dt)):
        state = simulator.step([0, 0.1])
        #print(state)

    #print(simulator.state_history)
    simulator.plot_simulation()

