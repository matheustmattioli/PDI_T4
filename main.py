# Código para implementação do quato trabalho prático da disciplina
# Processamento Digital de Imagens 2022/1 ofertada pelo DC-UFSCar
# Feito pelos alunos:
#   - Lucas Machado Cid          - RA: 769841
#   - Matheus Teixeira Mattioli  - RA: 769783
import numpy as np
import matplotlib.pyplot as plt
from filters.non_local import nonLocalMeans
from filters.geometric_mean import geometricMean
from filters.median import median
from filters.gaussian import gaussian
from helpers.plot import compareFilters

s = 3
w = np.ones([s, s])

imageDirectory = 'images/'

imagesStrings = [
    # {"name": "barco.tif", "hasColor": False},
    # {"name": "cameraman.jpg", "hasColor": False},
    # {"name": "coruja.jpg", "hasColor": True},
    # {"name": "flower.jpg", "hasColor": True},
    {"name": "pacman.jpg", "hasColor": False},

]

noises = [
    "gauss",
    "s&p"
]

filters = [
    {"name": "Media Geometrica", "function": lambda img: geometricMean(img, w)},
    {"name": "Mediana", "function": lambda img: median(img, w)},
    # {"name": "Gaussiana", "function": lambda img: gaussian(img, w)},
    {"name": "Non Local Mean", "function": lambda img: nonLocalMeans(3, 5, img)}
]

compareFilters(imagesStrings, noises, filters, imageDirectory)
