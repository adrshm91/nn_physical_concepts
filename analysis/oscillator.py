import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scinet import *
import scinet.ed_oscillator as edo


def osc_eqn(A_0, delta_0, b, kappa, t):
    return np.real(A_0 * np.exp(-b / 2. * t) * np.exp(1 / 2. * np.sqrt(b**2 - 4 * kappa + 0.j) * t + 1.j * delta_0))


def gen_input(A_0, delta_0, b, kappa, tt_predicted):
    tt_in = np.linspace(0, 5, 50)
    a = osc_eqn(A_0, delta_0, b, kappa, tt_in)
    ba = [osc_eqn(A_0, delta_0, b, kappa, tt_in) for _ in tt_predicted]
    in1 = np.array([osc_eqn(A_0, delta_0, b, kappa, tt_in) for _ in tt_predicted])
    in2 = np.reshape(tt_predicted, (-1, 1))
    out = in2 #dummy filler
    return [in1, in2, out]

blue_color='#000cff'
orange_color='#ff7700'

def pendulum_prediction(net, b, kappa):
    tt_given = np.linspace(0, 10, 250)
    tt_predicted = np.linspace(0, 10, 250)
    a_given = osc_eqn(1, 0, b, kappa, tt_given)
    a_precicted = net.run(gen_input(1, 0, b, kappa, tt_predicted), net.output).ravel()
    fig = plt.figure(figsize=(3.4, 2.1))
    ax = fig.add_subplot(111)
    ax.plot(tt_given, a_given, color=orange_color, label='True time evolution')
    ax.plot(tt_predicted, a_precicted, '--', color=blue_color, label='Predicted time evolution')
    ax.set_xlabel(r'$t$ [$s$]')
    ax.set_ylabel(r'$x$ [$m$]')
    handles, labels = ax.get_legend_handles_labels()
    lgd=ax.legend(handles, labels,loc='upper center', bbox_to_anchor=(0.6, 1.3), shadow=True, ncol=1)
    fig.tight_layout()
    return fig


def osc_representation_plot(net, b_range, kappa_range, step_num=100, eval_time=7.5):
    bb = np.linspace(*b_range, num=step_num)
    kk = np.linspace(*kappa_range, num=step_num)
    B, K = np.meshgrid(bb, kk)
    out = np.array([net.run(gen_input(1, 0, b, kappa, [eval_time]), net.mu)[0] for b, kappa in zip(np.ravel(B), np.ravel(K))])
    fig = plt.figure(figsize=(net.latent_size*3.9, 2.1))
    for i in range(net.latent_size):
        zs = out[:, i]
        ax = fig.add_subplot('1{}{}'.format(net.latent_size, i + 1), projection='3d')
        Z = np.reshape(zs, B.shape)
        surf = ax.plot_surface(B, K, Z, rstride=1, cstride=1, cmap=cm.inferno, linewidth=0)
        ax.set_xlabel(r'$b$ [$kg/s$]')
        ax.set_ylabel(r'$\kappa$ [$kg/s^2$]')
        ax.set_zlabel('Latent activation {}'.format(i + 1))
        if (i==2):
            ax.set_zlim(-1,1) #Fix the scale for the third plot, where the activation is close to zero
        ax.set_zticks([-1,-0.5,0,0.5,1])
    fig.tight_layout()
    return fig


net_2_latent = nn.Network.from_saved('oscillator')

pendulum_prediction(net_2_latent, 0.5, 5.);

