from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def histogram_gray(channel):
    # Converte para uint8 para garantir valores entre 0-255
    channel = (channel * 255).astype(np.uint8)
    
    hist = np.zeros(256, dtype=int)
    for pixel_value in channel.flatten():
        hist[pixel_value] += 1
    return hist

# Função para mostrar o histograma RGB
def show_histogram(img):
    histogram_r = histogram_gray(img[:, :, 0])
    histogram_g = histogram_gray(img[:, :, 1])
    histogram_b = histogram_gray(img[:, :, 2])
    
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis("off")
    plt.title("Imagem Original")
    
    plt.subplot(2, 1, 2)
    plt.bar(range(256), histogram_r, color='red', alpha=0.6, label='Red')
    plt.bar(range(256), histogram_g, color='green', alpha=0.6, label='Green')
    plt.bar(range(256), histogram_b, color='blue', alpha=0.6, label='Blue')
    plt.title("Histograma RGB")
    plt.xlabel("Intensidade do Pixel")
    plt.ylabel("Frequência")
    plt.legend()
    
    plt.show()

# Função para abrir a imagem e garantir que ela tenha 3 canais
def load_image(path):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

# Carregar imagens
img1 = load_image(r"C:\Users\marcu\Desktop\JooJs\Projeto 1\P1.png")
img2 = load_image(r"C:\Users\marcu\Desktop\JooJs\Projeto 1\P2.png")
img3 = load_image(r"C:\Users\marcu\Desktop\JooJs\Projeto 1\P3.jpg")
img4 = load_image(r"C:\Users\marcu\Desktop\JooJs\Projeto 1\P4.png")

# Encontrar o maior tamanho entre as imagens
max_width = max(img1.width, img2.width, img3.width, img4.width)
max_height = max(img1.height, img2.height, img3.height, img4.height)

# Redimensionar todas as imagens para o tamanho máximo encontrado
img1 = img1.resize((max_width, max_height))
img2 = img2.resize((max_width, max_height))
img3 = img3.resize((max_width, max_height))
img4 = img4.resize((max_width, max_height))

# Converter para arrays NumPy
ft1 = np.array(img1).astype(np.float64) / 255
ft2 = np.array(img2).astype(np.float64) / 255
ft3 = np.array(img3).astype(np.float64) / 255
ft4 = np.array(img4).astype(np.float64) / 255

#1. Junção das fotos
cima = np.hstack((ft1, ft2))
baixo = np.hstack((ft3, ft4))
juncao = np.vstack((cima, baixo))

# 2. Trocar as cores
# Converter valores de comparação para a escala correta (0 a 1)
branco = np.all(juncao > [177/255, 177/255, 177/255], axis=-1) # Para ignorar fundo branco, barriga e olhos do pinguim
preto = np.all(juncao < [30/255, 30/255, 30/255], axis=-1) # Para ignorar as linhas pretas do traço do pinguim

# Criar uma cópia para modificação
juncao_ciano = juncao.copy()

# Aplicar a cor ciano onde não for preto nem branco
ciano = np.array([0, 255, 255]) / 255  # Normalizar para 0 a 1
juncao_ciano[~branco & ~preto] = ciano

# Converter para uint8 (0 a 255) antes de transformar em imagem
juncao_ciano_uint8 = (juncao_ciano * 255).astype(np.uint8)

# Converter para imagem do Pillow
juncao_ciano_img = Image.fromarray(juncao_ciano_uint8)

# Aplicar o espelhamento corretamente
imagem_espelhada = cv2.flip(juncao_ciano_uint8, 1)

# 3. Aplicar o espelhamento horizontal
imagem_espelhada = cv2.flip(np.array(juncao_ciano), 1)

# 4. Recortar um único pinguim
altura, largura, _ = ft3.shape
x1, y1, x2, y2 = largura // 5, altura // 10, 3 * largura // 5, 3 * altura // 5
pinguim_recortado = ft3[y1:y2, x1:x2]
pinguim_recortado_uint8 = (pinguim_recortado * 255).astype(np.uint8)

# 5. Analisar histograma e escolher threshold
show_histogram(ft3)
cinza = cv2.cvtColor(pinguim_recortado_uint8, cv2.COLOR_RGB2GRAY)
threshold = 230 # Mantendo os olhos e a barriga brancas, alterando o resto do corpo do pinguim

# 6. Aplicar conversão para ciano onde os pixels forem menores que o threshold
pinguim_modificado = pinguim_recortado.copy()
pinguim_modificado[cinza < threshold] = [0,255,255]  # Ciano

# Exibir resultados
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(juncao)
plt.title("Imagem Sobreposta")

plt.subplot(2, 3, 2)
plt.imshow(juncao_ciano)
plt.title("Cores Alteradas")

plt.subplot(2, 3, 3)
plt.imshow(imagem_espelhada)
plt.title("Espelhamento")

plt.subplot(2, 3, 4)
plt.imshow(pinguim_recortado)
plt.title("Pinguim Recortado")

plt.subplot(2, 3, 5)
plt.imshow(pinguim_modificado)
plt.title("Pinguim com Threshold Ciano")

plt.show()