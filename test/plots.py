import numpy as np
import matplotlib.pyplot as plt


def autocorr(x, k_max):
    x_bar = x.mean()
    x_var = np.sum((x - x_bar)**2)
    c = np.ones((k_max,), dtype=np.float_)
    for k in range(1, k_max):
        c[k] = np.sum((x[k:] - x_bar) * (x[:-k] - x_bar)) / x_var
    return c


def create_plot(file_name):
    data = np.loadtxt(file_name)

    x = data[:, 0]
    y = data[:, 1]

    ac_x = autocorr(x, 1000)
    ac_y = autocorr(y, 1000)

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].plot(x)
    ax[0, 0].set_xlabel('Sample')
    ax[0, 0].set_ylabel('x')

    ax[1, 0].plot(y)
    ax[1, 0].set_xlabel('Sample')
    ax[1, 0].set_ylabel('y')

    ax[0, 1].plot(ac_x, label='IAC = {:.1f}'.format(1.0 + 2.0 * ac_x.sum()))
    ax[0, 1].legend()
    ax[0, 1].set_xlabel('Lag')
    ax[0, 1].set_ylabel('AC(k; x)')

    ax[1, 1].plot(ac_y, label='IAC = {:.1f}'.format(1.0 + 2.0 * ac_y.sum()))
    ax[1, 1].legend()
    ax[1, 1].set_xlabel('Lag')
    ax[1, 1].set_ylabel('AC(k; y)')

    plt.tight_layout()

    fig.savefig(file_name.replace('.dat', '.png'))
    plt.close(fig)


if __name__ == '__main__':
    create_plot('mh_test.dat')
    create_plot('am_test.dat')
    create_plot('dr_test.dat')
    create_plot('dram_test.dat')
