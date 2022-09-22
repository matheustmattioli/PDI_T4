import numpy as np

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