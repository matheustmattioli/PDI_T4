import numpy as np

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