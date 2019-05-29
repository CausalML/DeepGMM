import numpy as np


def approx_psi_eval(epsilon_dev, f_of_z_dev_list, epsilon_dev_tilde):
    eval_list = []
    for f_of_z in f_of_z_dev_list:
        mean_moment = f_of_z.mul(epsilon_dev).mean()
        c_term = (f_of_z ** 2).mul(epsilon_dev_tilde ** 2).mean()
        psi_f = float(mean_moment - 0.25 * c_term)
        next_eval = -1.0 * psi_f
        # next_eval = mean_moment / c_term ** 0.5
        eval_list.append(next_eval)
    return float(np.min(eval_list))


def max_approx_psi_eval(epsilon_dev_list, f_of_z_dev_list, epsilon_dev_tilde,
                        burn_in, max_no_progress):
    max_eval = float("-inf")
    no_progress = 0
    for i, epsilon_dev in enumerate(epsilon_dev_list):
        psi = approx_psi_eval(epsilon_dev, f_of_z_dev_list, epsilon_dev_tilde)
        if i >= burn_in:
            if psi > max_eval:
                max_eval = psi
                no_progress = 0
            else:
                no_progress += 1
            if no_progress == max_no_progress:
                break
    return max_eval, epsilon_dev
