import numpy as np

# Função para criar ruídos distintos em imagens
# os ruídos utilizados são o Gaussiano e o Salt and Peper (Impulsivo)
def noisy(noise_typ,image):
    image = image.astype(int)
    if noise_typ == "gauss":
        row, col = image.shape
        mean = 0
        var = 400
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, size=(row,col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        noisy -= noisy.min() 
        return noisy
    elif noise_typ == "s&p":
        rows,cols = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords[0], coords[1]] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        
        out[coords[0], coords[1]] = 0

        return out

# Função passada pelo Professor para retirar os canais de cores de uma imagem colorida,
# transformando em escala de cinza.
def rgb2gray(img):
    img_gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    return img_gray
