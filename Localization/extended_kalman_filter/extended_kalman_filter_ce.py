"""

Extended kalman filter (EKF) localization sample

author: Atsushi Sakai (@Atsushi_twi)

"""
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import math
import matplotlib.pyplot as plt
import numpy as np

from utils.plot import plot_covariance_ellipse

# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]

show_animation = True


def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud


def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F @ x + B @ u

    return x


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H @ x

    return z


def jacob_f(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH


def ekf_estimation(xEst, PEst, z, u):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    #  Update
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst

from utils.angle import rot_mat_2d

def get_cov_ellipse(x, y, cov, chi2=3.0):

    eig_val, eig_vec = np.linalg.eig(cov)

    if eig_val[0] >= eig_val[1]:
        big_ind = 0
        small_ind = 1
    else:
        big_ind = 1
        small_ind = 0
    a = math.sqrt(chi2 * eig_val[big_ind])
    b = math.sqrt(chi2 * eig_val[small_ind])
    angle = math.atan2(eig_vec[1, big_ind], eig_vec[0, big_ind])

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    px = [a * math.cos(it) for it in t]
    py = [b * math.sin(it) for it in t]
    fx = rot_mat_2d(angle) @ (np.array([px, py]))
    px = np.array(fx[0, :] + x).flatten()
    py = np.array(fx[1, :] + y).flatten()
    return px, py

def main():
    print(__file__ + " start!!")

    time = 0.0

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)

    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))
    hxCov = np.zeros((2, 1, 64))

    while SIM_TIME >= time:
        time += DT
        u = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))
        px, py = get_cov_ellipse(xEst[0, 0], xEst[1, 0], PEst)
        hxCov = np.hstack((hxCov, np.vstack((px, py)).reshape(2, 1, -1)))

        # if show_animation:
        #     plt.cla()
        #     # for stopping simulation with the esc key.
        #     plt.gcf().canvas.mpl_connect('key_release_event',
        #             lambda event: [exit(0) if event.key == 'escape' else None])
        #     plt.plot(hz[0, :], hz[1, :], ".g")
        #     plt.plot(hxTrue[0, :].flatten(),
        #              hxTrue[1, :].flatten(), "-b")
        #     plt.plot(hxDR[0, :].flatten(),
        #              hxDR[1, :].flatten(), "-k")
        #     plt.plot(hxEst[0, :].flatten(),
        #              hxEst[1, :].flatten(), "-r")
        #     plot_covariance_ellipse(xEst[0, 0], xEst[1, 0], PEst)
        #     plt.axis("equal")
        #     plt.grid(True)
        #     plt.pause(0.001)

    if show_animation:
        from matplotlib.animation import FuncAnimation
        fig, ax = plt.subplots()
        meas, = ax.plot([], [], ".g", label="Sensor Measurements")
        true, = ax.plot([], [], "-b", label="True Trajectory")
        dead, = ax.plot([], [], "-k", label="Dead Reckoning")
        est, = ax.plot([], [], "-r", label="EKF Estimate")
        cov, = ax.plot([], [], "-r", label="EKF Covariance")
        
        def anim_init():
            ax.axis("equal")
            ax.grid(True)
            ax.set_xlim(-20, 20)
            ax.set_ylim(-10, 30)
            ax.legend(ncol=2, loc="lower center")
            return meas, true, dead, est, cov,
    
        def animate(i):
            # ln.set_data(hxTrue[0, i:].flatten(), hxTrue[1, i:].flatten())
            meas.set_data(hz[0, :i].flatten(), hz[1, :i].flatten())
            true.set_data(hxTrue[0, :i].flatten(), hxTrue[1, :i].flatten())
            dead.set_data(hxDR[0, :i].flatten(), hxDR[1, :i].flatten())
            est.set_data(hxEst[0, :i].flatten(), hxEst[1, :i].flatten())
            cov.set_data(hxCov[0, i].flatten(), hxCov[1, i].flatten())
            return meas, true, dead, est, cov,
    
        anim = FuncAnimation(fig, animate, frames=range(len(hz[0, :])),
                    init_func=anim_init, blit=True, interval=10, repeat=False)
        plt.show()
        



if __name__ == '__main__':
    main()
