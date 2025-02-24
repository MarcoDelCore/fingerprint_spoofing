import numpy
from labs import lab02, lab03, lab04, lab05, lab07, lab08, lab09, lab10, lab11, evaluation
from lib.utils import vcol

def load(fileName):
    l_attrs = []
    l_labs = []
    with (open(fileName, 'r') as f):
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = vcol(numpy.array([float(i) for i in attrs]))
                label = int(line.split(',')[-1].strip())
                l_attrs.append(attrs)
                l_labs.append(label)
            except:
                pass
    return numpy.hstack(l_attrs, dtype=numpy.float64), numpy.array(l_labs, dtype=numpy.int32)


if __name__ == '__main__':
    D, L = load("trainData.txt")
    lab02.lab02_analysis(D, L)
    lab03.lab03_analysis(D, L)
    lab04.lab04_analysis(D, L)
    lab05.lab05_analysis(D, L)
    lab07.lab07_analysis(D, L)
    lab08.lab08_analysis(D, L)
    lab09.lab09_analysis(D, L)
    lab10.lab10_analysis(D, L)
    lab11.lab11_analysis(D, L)
    DEVAL, LEVAL = load("evalData.txt")
    evaluation.evaluation_analysis(D, L, DEVAL, LEVAL)
