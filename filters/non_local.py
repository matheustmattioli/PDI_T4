import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma


# Implementação do filtro não linear non-local means
# Espera receber o tamanho da vizinhança, da janela e uma imagem em esacala de cinza.
def nonLocalMeans(size_neigh, size_window, img):    
    sigma = np.mean(estimate_sigma(img))
    cut_off = 0.002*sigma
    # Dimensões da imagem
    num_rows, num_cols = img.shape
    # Padding para evitar acesso indevido nas bordas.
    img_padded_neigh = np.pad(img, size_neigh // 2, mode='reflect')
    img_padded_window = np.pad(img, size_window // 2, mode='reflect')
    
    filtered_img = np.zeros((num_rows, num_cols))
    # Laço principal
    # Para cada pixel da imagem, aplicar o filtro.
    for row in range(num_rows):
        for col in range(num_cols):
            # Recorte da vizinhança
            image_neigh = img_padded_neigh[row : row + size_neigh, col : col + size_neigh]
            # Recorte da janela de visualização
            image_window = img_padded_window[row : row + size_window, col : col + size_window]
            # Matriz de diferenças, obtida através de template matching por diferença quadrática.
            diff_matrix = quadraticDifference(image_window, image_neigh)
            # Normalização dos valores.
            diff_matrix = diff_matrix/np.sum(diff_matrix)
            # Cálculo da imagem w, implementado conforme a fórmula do slide.
            weighted_average = np.exp((-(diff_matrix**2))/(cut_off**2))
            weighted_average = weighted_average/np.sum(weighted_average)
            
            # Finalmente, é realizado a filtragem do pixel.
            filtered_img[row, col] = float(np.sum(weighted_average * image_window))
    # Após a filtragem em todos os pixels, retornamos a imagem filtrada.
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
