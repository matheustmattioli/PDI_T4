import matplotlib.pyplot as plt
from helpers.utils import rgb2gray, noisy

def compareFilters(imagesStrings, noises, filters, imageDirectory):
    for image in imagesStrings:
        plt.figure(figsize=[15,5])
        img = plt.imread(imageDirectory + image["name"])
        if(len(img.shape) == 3):
            img = img[:,:,0]

        separator = "/"
        file_img_name = "." + separator + "filtered_" + imageDirectory + separator + "filter_" + image["name"].split(".")[0] + ".jpg"
        
        if(image["hasColor"]):
            img = rgb2gray(img)

        for i in range(0, len(noises)):
            rows = len(noises)
            cols = len(filters) + 1
            img_noised = noisy(noises[i], img)
            plt.subplot(rows, cols, (i * cols) + 1)
            plt.axis('off')
            plt.title("Noise: " + str(noises[i]))
            
            plt.imshow(img_noised, cmap='gray')
            for j in range(0, len(filters)):
                plt.subplot(rows, cols, (i * cols) + 2 + j)
                img_smoothed = filters[j]["function"](img_noised)
                plt.axis('off')
                plt.title("Filtro: " + str(filters[j]["name"]))
                plt.imshow(img_smoothed, cmap='gray')
        plt.savefig(file_img_name, bbox_inches='tight')

def plotImageForFilterAndNoise(imagesStrings, noises, filters):
    for image in imagesStrings:
        img = plt.imread(image["name"])

        if(len(img.shape) == 4):
            img = img[:,:,0]

        separator = "/"
        file_img_name = "." + separator + "individual_filtered_images" + separator
        
        if(image["hasColor"]):
            img = rgb2gray(img)

        # print("tipo: ", img.dtype)

        for i in range(0, len(noises)):
            img_noised = noisy(noises[i], img)
            plt.axis('off')
            plt.title("Noise: " + str(noises[i]))
            plt.imshow(img_noised, cmap='gray')
            plt.savefig(file_img_name + image["name"].split(".")[0] + "_" + str(noises[i]) + ".jpg", bbox_inches='tight')
            for j in range(0, len(filters)):
                img_smoothed = filters[j]["function"](img_noised)
                plt.title("Noise: " + str(noises[i]) + "\n" + "Filtro: " + str(filters[j]["name"]))
                plt.imshow(img_smoothed, cmap='gray')
                plt.savefig(file_img_name + image["name"].split(".")[0] + "_" + str(noises[i]) + "_" + str(filters[j]["name"]) + ".jpg", bbox_inches='tight')
