import numpy as np
import matplotlib as plt

def nonLocalMeans(size_neigh, size_window, cut_off, img):    

    num_rows, num_cols = img.shape
    img_padded_neigh = np.pad(img, size_neigh // 2)
    img_padded_window = np.pad(img, size_window // 2)

    filtered_img = np.zeros((num_rows, num_cols))
    for row in range(num_rows):
        for col in range(num_cols):
            image_neigh = img_padded_neigh[row: row + size_neigh, col: col + size_neigh]
            image_window = img_padded_window[row: row + size_window, col: col + size_window]
            diff_matrix = quadraticDifference(image_window, image_neigh)
            weighted_average = np.exp((-diff_matrix**2)/(cut_off**2))
            weighted_average = weighted_average/np.sum(weighted_average)

            sum = 0
            for row_window in range(size_window):
                for col_window in range(size_window):
                    sum += weighted_average[row_window, col_window] * image_window[row_window, col_window]
            
            filtered_img[row, col] = sum

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
