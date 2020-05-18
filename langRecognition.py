import numpy as np
import os

D = 10000
N = 4

def genRandomHV(D):
    if (D % 2 != 0):
        print('Dimension is odd!!')
    else:
        randomIndex = np.random.permutation(range(D))
        randomHV(randomIndex(1, D / 2)) = 1
        randomHV(randomIndex(D / 2 + 1, D)) = -1
        # mean (randomHV)
        return randomHV


def lookupItemMemeory(itemMemory, key, D):
    if (key in itemMemory):
        randomHV = itemMemory(key)
    else:
        itemMemory(key) = genRandomHV(D)
        randomHV = itemMemory(key)
        return [itemMemory, randomHV]


def cosAng(u, v):
    cosAngle = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return cosAngle


def computeSumHV(buffer, itemMemory, N, D):
    # init
    block = np.zeros(N, D)
    sumHV = np.zeros(1, D)

    for numItems in range(1, len(buffer)):
        # read a key
        key = buffer(numItems)

        # shift read vectors
        block = np.roll(block,[1,1])
        [itemMemory, block[1:]] = lookupItemMemeory(itemMemory,key,D)

        if (numItems >= N):
            nGrams = block[1:]
            for i in range(2,N):
                nGrams = nGrams*block[1:]
            sumHV = sumHV + nGrams

    return[itemMemory, sumHV]

def binarizeHV(v):
    threshold = 0
    for i in range(1, len(v)):
        if (v(i) > threshold):
            v(i) = 1

        else:
            v(i) = -1

    return v


def binarizeLanguageHV(langAM):
    langLabels = ['afr', 'bul', 'ces', 'dan', 'nld', 'deu', 'eng', 'est', 'fin', 'fra', 'ell', 'hun', 'ita', 'lav',
                  'lit', 'pol', 'por', 'ron', 'slk', 'slv', 'spa', 'swe']

    for j in range(1, len(langLabels)):
        v = langAM(str(langLabels(j)))
        langAM(str(langLabels(j))) = binarizeHV(v)

    return langAM


def buildLanguageHV(N, D):
    iM = list(map())
    langAM = list(map())
    langLabels = ['afr', 'bul', 'ces', 'dan', 'nld', 'deu', 'eng', 'est', 'fin', 'fra', 'ell', 'hun', 'ita', 'lav',
                  'lit', 'pol', 'por', 'ron', 'slk', 'slv', 'spa', 'swe']

    for i in range(1,len(langLabels)):
        fileAddress = ('../training_texts/' + str(langLabels(i)))
        fileID = open(fileAddress,'r')
        buffer = open(fileID,'r')
        fileID.close()
        print('Loaded training language file ' + fileAddress)
        
        [iM, langHV] = computerSumHV(buffer, iM, N, D)
        langHV = langAM(str(langLabels(i)))
    return [iM, langAM]

def test(iM, langAM, N, D):
    total = 0
    correct = 0
    langLabels = ['afr', 'bul', 'ces', 'dan', 'nld', 'deu', 'eng', 'est', 'fin', 'fra', 'ell', 'hun', 'ita', 'lav',
                  'lit', 'pol', 'por', 'ron', 'slk', 'slv', 'spa', 'swe']
    langMap = list(map())
    'afr' = langMap('af')
    'bul' = langMap('bg')
    'ces' = langMap('cs')
    'dan' = langMap('da')
    'nld' = langMap('nl')
    'deu' = langMap('de')
    'eng' = langMap('en')
    'est' = langMap('et')
    'fin' = langMap('fi')
    'fra' = langMap('fr')
    'ell' = langMap('el')
    'hun' = langMap('hu')
    'ita' = langMap('it')
    'lav' = langMap('lv')
    'lit' = langMap('lt')
    'pol' = langMap('pl')
    'por' = langMap('pt')
    'ron' = langMap('ro')
    'slk' = langMap('sk')
    'slv' = langMap('sl')
    'spa' = langMap('es')
    'swe' = langMap('sv')

    fileList = os.listdir('../testing_texts/*.txt')
    for i in range(1,len(fileList)):
        actualLabel = str(fileList(i).name)
        actualLabel = actualLabel[1:2]

        fileAddress = ('../testing_texts/' + fileList(i).name)
        fileID = open(fileAddress,'r')
        buffer = open(fileID,'r')
        fileID.close()
        print('Loaded testing text file ' + fileAddress)

        [iMn, textHV] = computeSumHV(buffer, iM, N, D)
        textHV = binarizeHV(textHV)
        if (iM != iMn):
            print('\n>>>> NEW UNSEEN ITEM IN TEST FILE <<<<\n')
            exit()
        else:
            maxAngle = -1
            for l in range(1,len(langLabels)):
                angle = cosAngle(langAM(str(langLabels(l))), textHV)
                if (angle > maxAngle):
                    maxAngle = angle
                    predicLang = str(langLabels(l))

            if (predicLang == langMap(actualLabel)):
                correct = correct + 1
            else:
                print(langMap(actualLabel) + ' --> ' + predicLang)

            total = total+1

    accuracy = correct/total
