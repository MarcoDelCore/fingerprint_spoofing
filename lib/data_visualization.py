from matplotlib import pyplot as plt


def plotHist(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    for dIdx in range(0, 6, 2):
        figure, axis = plt.subplots(1, 2, figsize=(10, 5))
        axis[0].hist(D0[dIdx, :], density=True, alpha=0.4, label="Counterfeit")
        axis[0].hist(D1[dIdx, :], density=True, alpha=0.4, label="Genuine")
        axis[0].set_title("Histogram for feature %d" % (dIdx + 1))
        axis[0].legend()
        dIdx += 1
        axis[1].hist(D0[dIdx, :], density=True, alpha=0.4, label="Counterfeit")
        axis[1].hist(D1[dIdx, :], density=True, alpha=0.4, label="Genuine")
        axis[1].set_title("Histogram for feature %d" % (dIdx + 1))
        axis[1].legend()
        plt.savefig("output_data/lab02/hist_features_%d_and_%d.png" % (dIdx, dIdx+1))
        plt.show()


def plotScatter(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for dIdx1 in range(0, 6, 2):
        plt.figure()
        plt.title("Scatter plot for features %d and %d" % (dIdx1 + 1, dIdx1 + 2))
        plt.scatter(D0[dIdx1, :], D0[dIdx1+1, :], label="Counterfeit")
        plt.scatter(D1[dIdx1, :], D1[dIdx1+1, :], label="Genuine")
        plt.legend()
        plt.savefig("output_data/lab02/scatters_feature_%d_and_%d.png" % (dIdx1 + 1, dIdx1+2))
        plt.show()
        plt.close()
