import numpy as np
import scipy.signal

# Função gaussiana em duas dimensões
# Retirado do notebook "suavização" da aula 
# da semana 3.
# Retorna um filtro gaussiano w 
def do_gaussian_filter_2d(filter_size):
    sigma = filter_size/6.
    x_vals = np.linspace(-3*sigma, 3*sigma, filter_size)
    y_vals = x_vals.copy()
    z = np.zeros((filter_size, filter_size))
    for row in range(filter_size):
        x = x_vals[row]
        for col in range(filter_size):
            y = y_vals[col]
            z[row, col] = np.exp(-(x**2+y**2)/(2*sigma**2))
    z = z/np.sum(z)

    return z

# Função para gerar um filtro gaussiano e aplicá-lo
# em uma imagem através da biblioteca scipy.signal
def gaussian(img, w):
    w = do_gaussian_filter_2d(6)
    img_filtered = scipy.signal.convolve(img, w, mode='same')
    return img_filtered
