

# import os
# import cv2
# import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras import layers, Model
# from sklearn.model_selection import train_test_split
# import tensorflow
# import numpy as np
# from keras import Model
# from keras.layers import Conv2D, PReLU, BatchNormalization, Flatten
# from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add
# from tqdm import tqdm
#
#
# #########################################################################
#
# # Define blocks to build the generator
# def res_block(ip):
#     res_model = Conv2D(64, (3, 3), padding="same")(ip)
#     res_model = BatchNormalization(momentum=0.5)(res_model)
#     res_model = PReLU(shared_axes=[1, 2])(res_model)
#
#     res_model = Conv2D(64, (3, 3), padding="same")(res_model)
#     res_model = BatchNormalization(momentum=0.5)(res_model)
#
#     return add([ip, res_model])
#
#
# def upscale_block(ip):
#     up_model = Conv2D(256, (3, 3), padding="same")(ip)
#     up_model = UpSampling2D(size=2)(up_model)
#     up_model = PReLU(shared_axes=[1, 2])(up_model)
#
#     return up_model
#
#
# # Generator model
# def create_gen(gen_ip, num_res_block):
#     layers = Conv2D(64, (9, 9), padding="same")(gen_ip)
#     layers = PReLU(shared_axes=[1, 2])(layers)
#
#     temp = layers
#
#     for i in range(num_res_block):
#         layers = res_block(layers)
#
#     layers = Conv2D(64, (3, 3), padding="same")(layers)
#     layers = BatchNormalization(momentum=0.5)(layers)
#     layers = add([layers, temp])
#
#     layers = upscale_block(layers)
#     layers = upscale_block(layers)
#
#     op = Conv2D(3, (9, 9), padding="same")(layers)
#
#     return Model(inputs=gen_ip, outputs=op)
#
#
# # Descriminator block that will be used to construct the discriminator
# def discriminator_block(ip, filters, strides=1, bn=True):
#     disc_model = Conv2D(filters, (3, 3), strides=strides, padding="same")(ip)
#
#     if bn:
#         disc_model = BatchNormalization(momentum=0.8)(disc_model)
#
#     disc_model = LeakyReLU(alpha=0.2)(disc_model)
#
#     return disc_model
#
#
# # Descriminartor
# def create_disc(disc_ip):
#     df = 64
#
#     d1 = discriminator_block(disc_ip, df, bn=False)
#     d2 = discriminator_block(d1, df, strides=2)
#     d3 = discriminator_block(d2, df * 2)
#     d4 = discriminator_block(d3, df * 2, strides=2)
#     d5 = discriminator_block(d4, df * 4)
#     d6 = discriminator_block(d5, df * 4, strides=2)
#     d7 = discriminator_block(d6, df * 8)
#     d8 = discriminator_block(d7, df * 8, strides=2)
#
#     d8_5 = Flatten()(d8)
#     d9 = Dense(df * 16)(d8_5)
#     d10 = LeakyReLU(alpha=0.2)(d9)
#     validity = Dense(1, activation='sigmoid')(d10)
#
#     return Model(disc_ip, validity)
#
#
# # VGG19
# # We need VGG19 for the feature map obtained by the j-th convolution (after activation)
# # before the i-th maxpooling layer within the VGG19 network.(as described in the paper)
# # Let us pick the 3rd block, last conv layer.
# # Build a pre-trained VGG19 model that outputs image features extracted at the
# # third block of the model
# # VGG architecture: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
# from keras.applications import VGG19
#
#
# def build_vgg(hr_shape):
#     vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)
#
#     return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)
#
#
# # Combined model
# def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
#     gen_img = gen_model(lr_ip)
#
#     gen_features = vgg(gen_img)
#
#     disc_model.trainable = False
#     validity = disc_model(gen_img)
#
#     return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])
#
#
# # 2 losses... adversarial loss and content (VGG) loss
# # AdversariaL: is defined based on the probabilities of the discriminator over all training samples
# # use binary_crossentropy
#
# # Content: feature map obtained by the j-th convolution (after activation)
# # before the i-th maxpooling layer within the VGG19 network.
# # MSE between the feature representations of a reconstructed image
# # and the reference image.
#
# ###################################################################################
#
# # Load first n number of images (to train on a subset of all images)
# # For demo purposes, let us use 5000 images
# n = 60000
# lr_list = os.listdir("/Users/eliyahunezri/Desktop/AI_applications/mix data/lr/")[:n]
#
# lr_images = []
# for img in lr_list:
#
#     if img != ".DS_Store":
#         img_lr = cv2.imread("/Users/eliyahunezri/Desktop/AI_applications/mix data/lr/" + img)
#
#         img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
#         lr_images.append(img_lr)
#
# hr_list = os.listdir("/Users/eliyahunezri/Desktop/AI_applications/mix data/hr/")[:n]
#
# hr_images = []
# for img in hr_list:
#
#     if img != ".DS_Store":
#         img_hr = cv2.imread("/Users/eliyahunezri/Desktop/AI_applications/mix data/hr/" + img)
#         img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
#         hr_images.append(img_hr)
#
# lr_images = np.array(lr_images)
# hr_images = np.array(hr_images)
#
# # Sanity check, view few mages
# import random
# import numpy as np
#
# image_number = random.randint(0, len(lr_images) - 1)
# # plt.figure(figsize=(12, 6))
# # plt.subplot(121)
# # plt.imshow(np.reshape(lr_images[image_number], (32, 32, 3)))
# # plt.subplot(122)
# # plt.imshow(np.reshape(hr_images[image_number], (128, 128, 3)))
# # plt.show()
#
# # Scale values
# lr_images = lr_images / 255.
# hr_images = hr_images / 255.
#
# # Split to train and test
# lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images,
#                                                         test_size=0.33, random_state=42)
#
# hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])
# lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])
#
# lr_ip = Input(shape=lr_shape)
# hr_ip = Input(shape=hr_shape)
#
# generator = create_gen(lr_ip, num_res_block=16)
# generator.summary()
#
# discriminator = create_disc(hr_ip)
# discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
# discriminator.summary()
#
# vgg = build_vgg((128, 128, 3))
# print(vgg.summary())
# vgg.trainable = False
#
# gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)
#
# # 2 losses... adversarial loss and content (VGG) loss
# # AdversariaL: is defined based on the probabilities of the discriminator over all training samples
# # use binary_crossentropy
#
# # Content: feature map obtained by the j-th convolution (after activation)
# # before the i-th maxpooling layer within the VGG19 network.
# # MSE between the feature representations of a reconstructed image
# # and the reference image.
# gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="adam")
# gan_model.summary()
#
# # Create a list of images for LR and HR in batches from which a batch of images
# # would be fetched during training.
# batch_size = 1
# train_lr_batches = []
# train_hr_batches = []
# for it in range(int(hr_train.shape[0] / batch_size)):
#     start_idx = it * batch_size
#     end_idx = start_idx + batch_size
#     train_hr_batches.append(hr_train[start_idx:end_idx])
#     train_lr_batches.append(lr_train[start_idx:end_idx])
#
# g_loss_list = []
# d_loss_list = []
#
# epochs = 10
# # Enumerate training over epochs
# for e in range(epochs):
#
#     fake_label = np.zeros((batch_size, 1))  # Assign a label of 0 to all fake (generated images)
#     real_label = np.ones((batch_size, 1))  # Assign a label of 1 to all real images.
#
#     # Create empty lists to populate gen and disc losses.
#     g_losses = []
#     d_losses = []
#
#     # Enumerate training over batches.
#     for b in tqdm(range(len(train_hr_batches))):
#         lr_imgs = train_lr_batches[b]  # Fetch a batch of LR images for training
#         hr_imgs = train_hr_batches[b]  # Fetch a batch of HR images for training
#
#         fake_imgs = generator.predict_on_batch(lr_imgs)  # Fake images
#
#         # First, train the discriminator on fake and real HR images.
#         discriminator.trainable = True
#         d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label)
#         d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)
#
#         # Now, train the generator by fixing discriminator as non-trainable
#         discriminator.trainable = False
#
#         # Average the discriminator loss, just for reporting purposes.
#         d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)
#
#         # Extract VGG features, to be used towards calculating loss
#         image_features = vgg.predict(hr_imgs)
#
#         # Train the generator via GAN.
#         # Remember that we have 2 losses, adversarial loss and content (VGG) loss
#         g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])
#
#         # Save losses to a list so we can average and report.
#         d_losses.append(d_loss)
#         g_losses.append(g_loss)
#
#     # Convert the list of losses to an array to make it easy to average
#     g_losses = np.array(g_losses)
#     d_losses = np.array(d_losses)
#
#     # Calculate the average losses for generator and discriminator
#     g_loss = np.sum(g_losses, axis=0) / len(g_losses)
#     d_loss = np.sum(d_losses, axis=0) / len(d_losses)
#
#     g_loss_list.append(g_loss)
#     d_loss_list.append(d_loss)
#
#     # Report the progress during training.
#     print("epoch:", e + 1, "g_loss:", g_loss, "d_loss:", d_loss)
#
#     if (e + 1) % 5 == 0:  # Change the frequency for model saving, if needed
#         # Save the generator after every n epochs (Usually 10 epochs)
#         generator.save("gen_e_" + str(e + 1) + ".h5")
#
# plt.plot(np.arange(epochs), g_loss_list, label="g_loss")
# plt.plot(np.arange(epochs), d_loss_list, label="d_loss")
# plt.legend()
# plt.xlabel("epochs")
# plt.ylabel('loss')
# plt.title('training')
# plt.savefig('graph.png')
# plt.show()



##################################################################################
import skimage

def info_detection(image):

    # pytesseract.pytesseract.tesseract_cmd = "/Users/eliyahunezri/Desktop/AI_applications/tesseract.exe"

    img = cv2.resize(image, (252, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    info = pytesseract.image_to_string(img, lang='eng', config='--psm 6')


    return  info

def find_largest_objects(binary_image, num_objects=7):
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    # Extract areas of connected components
    areas = stats[:, cv2.CC_STAT_AREA]
    # Find indices of largest objects
    largest_indices = np.argsort(areas)[::-1][1:num_objects + 1]  # Exclude background
    # Create a blank image
    result_image = np.zeros_like(binary_image)
    # Draw the largest objects
    for index in largest_indices:
        result_image[labels == index] = 255
    return result_image

def find_objects(binary_image, min_area=100, max_area=10000):
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a blank image to store each object
    object_images = []
    # Iterate through each contour
    for i, contour in enumerate(contours):
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        # Check if the area of the contour satisfies the condition
        if min_area < area < max_area:
            # Create a mask for the current object
            mask = np.zeros_like(binary_image)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            # Apply the mask to the original image
            object_image = cv2.bitwise_and(binary_image, mask)
            # Add the object to the list
            object_images.append(object_image)
    return object_images

def edge_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Canny edge detection to detect edges
    edges = np.uint8(cv2.GaussianBlur(gray, (5, 5), 0).astype(int))
    # Dilate the edges to make them more prominent
    dilated = cv2.dilate(edges, None, iterations=2)
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on the original image
    cv2.drawContours(gray, contours, -1, (0, 0, 0), 2)
    # Display the original image with edges
    cv2.imshow("Original with Edges", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert the image to grayscale
    # gray = cv2.cvtColor(image_255, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = np.uint8(cv2.GaussianBlur(gray, (5, 5), 0).astype(int))
    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 80, 150)
    return edges

# Test - perform super resolution using saved generator model
#
from keras.models import load_model
from numpy.random import randint
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')
import imutils
from PIL import Image
import PIL.ImageOps
import pytesseract
from matplotlib.backends.backend_agg import FigureCanvasAgg


def project(path):
    i = 1

    generator = load_model('gen_e_10_old.h5', compile=False)

    # path = "/Users/eliyahunezri/Desktop/AI_applications/"
    # img = cv2.imread(path + "test" + str(i) + ".jpeg")
    img = cv2.imread(path)

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # ret, bin_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)

    hr = cv2.resize(gray_img, (128, 128))
    lr = cv2.resize(gray_img, (32, 32))

    # # Change images from BGR to RGB for plotting.
    # # Remember that we used cv2 to load images which loads as BGR.
    lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

    lr = lr / 255.
    hr = hr / 255.

    lr = np.expand_dims(lr, axis=0)
    hr = np.expand_dims(hr, axis=0)

    generated_hr = generator.predict(lr)

    image_255_hr = np.clip(np.array(generated_hr[0, :, :, :]) * 255, 0, 255).astype(int)
    image_255_hr = np.float32(image_255_hr)

    image_255_lr = np.clip(np.array(lr[0, :, :, :]) * 255, 0, 255).astype(int)
    image_255_lr = np.float32(image_255_lr)


    edge = detect_edges(image_255_hr)

    test = np.uint8((cv2.cvtColor(image_255_hr, cv2.COLOR_RGB2GRAY) + edge))
    # test = cv2.cvtColor(test, cv2.COLOR_GRAY2RGB)
    ret, test = cv2.threshold(test, 80, 255, cv2.THRESH_BINARY_INV)
    test = np.uint8(cv2.GaussianBlur(test, (3, 3), 0).astype(int))
    ret, test = cv2.threshold(test, 80, 255, cv2.THRESH_BINARY)
    test = test + edge
    test = np.uint8(cv2.GaussianBlur(test, (3, 3), 0).astype(int))
    ret, test = cv2.threshold(test, 80, 255, cv2.THRESH_BINARY)
    test = find_largest_objects(test, num_objects=8)

    info = info_detection(test)


    ####
    # Find objects in the binary image
    # object_images = find_objects(test)

    # Display each object image
    # for i, object_image in enumerate(object_images):
    #     text = pytesseract.image_to_string(object_image)
    #     print("text:", text)
    #     plt.imshow(object_image, 'gray')
    #     plt.show()
    #####

    # text = pytesseract.image_to_string(test)
    # print("text:", text)

    # # Example usage
    # edge_detection(image_255_hr)

    # img_pil = Image.fromarray(test)
    # img_pil.save("12.png")

    # plt.imshow(image_255_lr.astype(int))
    # plt.savefig('x1_lr.jpeg')
    # plt.imshow(edge)
    # plt.savefig('x1_edge.jpeg')

    # plot all three images
    px = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(layout="constrained", figsize=(720*px, 960*px))
    ax = fig.subplot_mosaic([['left', 'center', 'right']])

    # plt.figure(figsize=(16, 8))
    # plt.subplot(232)

    ax['center'].imshow(image_255_lr.astype(int))
    ax['center'].set_title('LR Image')

    # plt.subplot(233)

    ax['right'].imshow(image_255_hr.astype(int))
    ax['right'].set_title('Super resolution')
    # plt.subplot(233)
    # plt.title('edge')
    # plt.imshow(edge, 'gray')
    # plt.subplot(231)


    ax['left'].imshow(hr[0, :, :, :])
    ax['left'].set_title('Orig. HR image')
    # plt.subplot(235)
    # plt.title('test')
    # plt.imshow(test, 'gray')

    fig.suptitle("\n" + " \n" + "result: " + info, fontsize='40', color='blue')

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    im = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    return im

    # path = "/Users/eliyahunezri/Desktop/AI_applications/res/"
    # plt.savefig(path + name_save)
    # plt.show()







