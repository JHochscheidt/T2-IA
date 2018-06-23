from numpy import genfromtxt, array, linalg, zeros, apply_along_axis, set_printoptions, inf, random
from minisom import MiniSom
import sys
from pylab import plot, axis, show, pcolor, colorbar, bone


set_printoptions(threshold=inf)
#tamanho conjunto de treinamento (em %)
TAM_SET_TRAIN = 70
TAM_SET_TEST = 15
TAM_SET_EVAL = 15
N_ITER_TRAIN = 10000
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python %s <dataset file name (full path)>' % sys.argv[0])
        exit(0)

    try:
        #le arquivo com dataset
        seeds_dataset = genfromtxt(sys.argv[1],delimiter=';', usecols=(0,1,2,3,4,5,6))  # open the dataset file
        target = genfromtxt(sys.argv[1], delimiter=';', usecols=(7), dtype=int)
    except:
        print('Could not open', sys.argv[1])
        exit(0)
    
    #normaliza dados
    seeds_dataset = apply_along_axis(lambda x: x/linalg.norm(x), 1, seeds_dataset)

    #embaralha os dados
    #random.shuffle(seeds_dataset)
    # #print seeds_dataset
    # #define tamanho do conjunto de treinamento
    # nsizeTr = seeds_dataset.shape[0]*TAM_SET_TRAIN/100
    # nsizeTe = seeds_dataset.shape[0]*TAM_SET_TEST/100
    # nsizeEv = seeds_dataset.shape[0]*TAM_SET_EVAL/100

    # X = seeds_dataset[ : , :-1]
    # y = seeds_dataset[ : ,-1:]

    #divide o dataset em dois conjuntos: treinamento e teste
    # Xtrain = X[ :nsizeTr, :-1]
    # ytrain = y[ :nsizeTr ,-1:]

    # Xtest = X[nsizeTr : nsizeTr+nsizeTe , :-1]
    # ytest = y[nsizeTr: nsizeTr+nsizeTe, -1:]

    # Xeval = X[nsizeTr+nsizeTe: , :-1]
    # yeval = y[nsizeTr+nsizeTe:, -1:]X

    # print 'train:' , Xtrain.shape, ytrain.shape
    # print Xtrain, '\n chesque \n'
    # print 'test:' , Xtest.shape, ytest.shape
    # print Xtest, '\n dele \n'
    # print 'eval:' , Xeval.shape, yeval.shape
    # print Xeval

    #print target
    """ ========================== ATE AQUI OS DADOS ESTAO PRONTOS ======================= """
    """ ====== NO FORMATO CERTO (NUMPY) E OS SETS (train,test,evaluation) SEPARADOS ====== """

    som = MiniSom(12,12,7,sigma=1.0, learning_rate=0.5)

    #inicializa pesos com valores aleatorios
    som.random_weights_init(seeds_dataset)
    # progress(0, 100, "Training...")
    print "Training..."
    som.train_random(seeds_dataset, N_ITER_TRAIN)
    print "... ready!"
    bone()
    pcolor(som.distance_map().T)
    colorbar()
    markers = ['o','s','D']
    colors = ['r','g','b']
    target[target == 3] = 0
    #print target


    for cnt,xx in enumerate(seeds_dataset):
        w = som.winner(xx) # getting the winner
        # palce a marker on the winning position for the sample xx
        plot(w[0]+.5,w[1]+.5,markers[target[cnt]],markerfacecolor='None', markeredgecolor=colors[target[cnt]],markersize=12,markeredgewidth=2)
        axis([0,som._weights.shape[0],0,som._weights.shape[1]])
    show() # show the figure

    
    
