# Código para implementação do quato trabalho prático da disciplina
# Processamento Digital de Imagens 2022/1 ofertada pelo DC-UFSCar
# Feito pelos alunos:
#   - Lucas Machado Cid          - RA: 769841
#   - Matheus Teixeira Mattioli  - RA: 769783
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from skimage.restoration import estimate_sigma, denoise_nl_means

def nonLocalMeans(size_neigh, size_window, cut_off, img):    

    # print("img = ", img)
    num_rows, num_cols = img.shape
    img_padded_neigh = np.pad(img, size_neigh // 2, mode='reflect')
    img_padded_window = np.pad(img, size_window // 2, mode='reflect')
    
    filtered_img = np.zeros((num_rows, num_cols))
    for row in range(num_rows):
        for col in range(num_cols):
            image_neigh = img_padded_neigh[row : row + size_neigh, col : col + size_neigh]
            image_window = img_padded_window[row : row + size_window, col : col + size_window]
            diff_matrix = quadraticDifference(image_window, image_neigh)
            diff_matrix = diff_matrix/np.sum(diff_matrix)
            weighted_average = np.exp((-(diff_matrix**2))/(cut_off**2))
            weighted_average = weighted_average/np.sum(weighted_average)
            
            filtered_img[row, col] = float(np.sum(weighted_average * image_window))

    return filtered_img

def quadraticDifference(img, obj):
    '''Calcula a diferença quadrática entre as imagens img e obj. É assumido que img é maior do 
       que obj. Portanto, a diferença é calculada para cada posição do centro da imagem obj ao 
       longo de img. Note que a função é facilmente modificável para processar imagens coloridas.'''
    
    num_rows, num_cols = img.shape
    num_rows_obj, num_cols_obj = obj.shape   

    half_num_rows_obj = num_rows_obj//2        # O operador // retorna a parte inteira da divisão
    half_num_cols_obj = num_cols_obj//2

    # Cria imagem com zeros ao redor da borda. Note que ao invés de adicionarmos 0, seria mais 
    # preciso calcularmos a diferença quadrática somente entre pixels contidos na imagem.
    img_padded = np.pad(img, ((half_num_rows_obj,half_num_rows_obj),
                             (half_num_cols_obj,half_num_cols_obj)), 
                             mode='reflect')
    
    img_diff = np.zeros((num_rows, num_cols))
    for row in range(num_rows):
        for col in range(num_cols):
            # patch é a região de img de mesmo tamanho que obj e centrada em (row, col)
            patch = img_padded[row:row+num_rows_obj, col:col+num_cols_obj]
            # Utilizando numpy, o comando abaixo calcula a diferença entre cada valor
            # dos arrays 2D patch e obj
            diff_region = (patch - obj)**2
            img_diff[row, col] = np.sum(diff_region)
            
    return img_diff

# Função para realizar filtragem dos pixels de uma imagem
# através da média geometrica dos pixels de uma vizinhança.
# O tamanho dessa vizinhança é definido por w
def geometricMean(img, w):
    # Dimensões de linhas e colunas da imagem e 
    # da vizinhança do w passado.
    num_rows, num_cols = img.shape
    num_rows_f, num_cols_f = w.shape

    # Pegamos metade das dimensões de w
    # para fazer o padding
    half_num_rows_f = num_rows_f//2       # O operador // retorna a parte inteira da divisão
    half_num_cols_f = num_cols_f//2

    # Cria imagem com uns ao redor da borda
    # Escolhemos um ao invés de zero, pois ele é o elemento neutro da multiplicação
    # se fosse zeros a imagem ficaria toda preta.
    img_padded = np.ones((num_rows+2*half_num_rows_f, num_cols+2*half_num_cols_f), dtype=img.dtype)
    for row in range(num_rows):
        for col in range(num_cols):   
            img_padded[row+half_num_rows_f, col+half_num_cols_f] = img[row, col]
    
    # Aplicação do filtro de média geométrica nos pixels da imagem
    img_filtered = np.zeros((num_rows, num_cols))
    for row in range(num_rows):
        for col in range(num_cols):
            prod_region = 1
            for s in range(num_rows_f):
                for t in range(num_cols_f):
                    prod_region *= (img_padded[row+s, col+t] * w[s, t])**(1/(num_rows_f*num_cols_f))
            img_filtered[row, col] = int(prod_region)
            
    return img_filtered

# Função para realizar filtragem dos pixels de uma imagem
# através da mediana dos pixels de uma vizinhança.
# O tamanho dessa vizinhança é definido por w
def median(img, w):
    # Dimensões de linhas e colunas da imagem e 
    # da vizinhança do w passado.
    num_rows, num_cols = img.shape
    num_rows_f, num_cols_f = w.shape  

    # Pegamos metade das dimensões de w
    # para fazer o padding
    half_num_rows_f = num_rows_f//2       # O operador // retorna a parte inteira da divisão
    half_num_cols_f = num_cols_f//2

    # Cria imagem com zeros ao redor da borda
    img_padded = np.ones((num_rows+2*half_num_rows_f, num_cols+2*half_num_cols_f), dtype=img.dtype)
    for row in range(num_rows):
        for col in range(num_cols):   
            img_padded[row+half_num_rows_f, col+half_num_cols_f] = img[row, col]
 
    # Aplicação do filtro de mediana nos pixels da imagem
    img_filtered = np.zeros((num_rows, num_cols))
    for row in range(num_rows):
        for col in range(num_cols):
            region = []     # Array que armazena o valor dos pixels da vizinhaça
            for s in range(num_rows_f):
                for t in range(num_cols_f):
                    region.append(img_padded[row+s, col+t])
            region.sort() # Ordenação em ordem crescente
            # Verificamos se o tamanho do filtro é par ou ímpar,
            # se par o meio é calculado através da média entre os valores centrais,
            # se ímpar escolhemos o valor central
            if num_cols_f*num_rows_f % 2 == 0: 
                a = int(region[num_rows_f*num_cols_f//2 - 1])
                b = int(region[num_rows_f*num_cols_f//2])
                
                img_filtered[row, col] = (a + b)//2
            else:
                img_filtered[row, col] = region[num_rows_f*num_cols_f//2]

            
    return img_filtered

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
def gaussian_filter(img, w):
    w = do_gaussian_filter_2d(6)
    img_filtered = scipy.signal.convolve(img, w, mode='same')
    return img_filtered

# Função passada pelo Professor para retirar os canais de cores de uma imagem colorida,
# transformando em escala de cinza.
def rgb2gray(img):
    img_gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    return img_gray

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


s = 3
w = np.ones([s, s])


img = plt.imread("cameraman.jpg")
sigma = np.mean(estimate_sigma(img[:,:,0]))
filtered_img = nonLocalMeans(3, 5, 1.*sigma, img[:,:,0])
# filtered_img = denoise_nl_means(img[:,:,0], h=1.*sigma, fast_mode=True, patch_size=5, patch_distance=3, multichannel=False)

plt.imshow(filtered_img, 'gray')
plt.show()