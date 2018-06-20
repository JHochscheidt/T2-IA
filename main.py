import numpy as np
import sys

np.set_printoptions(threshold=np.inf)
#tamanho conjunto de treinamento (em %)
TAM_SET_TRAIN = 60

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python %s <dataset file name (full path)>' % sys.argv[0])
        exit(0)

    try:
        seeds_dataset = np.loadtxt(sys.argv[1],delimiter=';')  # open the dataset file
    except:
        print('Could not open', sys.argv[1])
        exit(0)

    #embaralha os dados
    np.random.shuffle(seeds_dataset)
    
    #define tamanho do conjunto de treinamento
    nsize = seeds_dataset.shape[0]*TAM_SET_TRAIN/100
    

    #divide o dataset em dois conjuntos: treinamento e teste
    Xtrain = seeds_dataset[ :nsize, :-1]
    ytrain = seeds_dataset[ :nsize ,-1:]

    Xtest = seeds_dataset[nsize :, :-1]
    ytest = seeds_dataset[nsize : ,-1:]


    
    