import matplotlib.pyplot as plt
from lib import gaussian_model as gm, utils



def lab04_analysis(D, L):

    for i in range(2):
        D0 = D[:, L == i]
        figure, axis = plt.subplots(2, 3, figsize=(15, 10))
        for dIdx in range(6):
            D0_d = utils.vrow(D0[dIdx, :])
            place = axis[dIdx // 3, dIdx % 3]
            m_ML, C_ML = gm.mu_C_matrices(D0_d)
            ll = gm.loglikelihood(D0_d, m_ML, C_ML)
            print("Log-likelihood for feature %d of class %d: %.4f" % (dIdx + 1, i, ll))
            place.set_title("Histogram for feature %d of class %d" % (dIdx + 1, i))
            gm.plot_pdf(D0_d, m_ML, C_ML, place)
        plt.savefig("output_data/lab04/hist_features_class_%d.png" % i)
        plt.show()

    gm.plot_pdf_both_classes(D, L)
