import sys
import os
import matplotlib.pyplot as plt
import idx2numpy
import numpy as np


x = [1, 10, 20, 30, 40, 50, 100, 200, 1000]
y = [31.92, 16.05, 14.2, 13.17, 12.5, 12.15, 10.74, 9.85, 8.51]
plt.plot(x, y)
plt.show()
sys.path.append(os.getcwd())

from lib.logistic_regression import model

TRN_IMG_PATH = os.path.join(os.getcwd(), "res", "train_num_imgs")
TRN_LABELS_PATH = os.path.join(os.getcwd(), "res", "train_num_labels")
TEST_IMG_PATH = os.path.join(os.getcwd(), "res", "test_num_imgs")
TEST_LABELS_PATH = os.path.join(os.getcwd(), "res", "test_num_labels")

imgs = idx2numpy.convert_from_file(TRN_IMG_PATH)
lbls = idx2numpy.convert_from_file(TRN_LABELS_PATH)

m, px, py = imgs.shape

train_x_0 = imgs.reshape(imgs.shape[0], -1).T
train_y_0 = lbls.reshape(1, m)

imgs = idx2numpy.convert_from_file(TEST_IMG_PATH)
lbls = idx2numpy.convert_from_file(TEST_LABELS_PATH)

test_x_0 = imgs.reshape(imgs.shape[0], -1).T
test_y_0 = lbls.reshape(1, lbls.shape[0])

train_x_0 = train_x_0/255
test_x_0 = test_x_0/255

mods_by_number = {}
for i in range(10):
	if i == 0:
		train_y = np.where(train_y_0 != i, 2, train_y_0)
		train_y = np.where(train_y == i, 1, train_y)
		train_y = np.where(train_y == 2, 0, train_y)
		test_y = np.where(test_y_0 != i, 2, test_y_0)
		test_y = np.where(test_y == i, 1, test_y)
		test_y = np.where(test_y == 2, 0, test_y)
	else:
		train_y = np.where(train_y_0 != i, 0, train_y_0)
		train_y = np.where(train_y == i, 1, train_y)
		test_y = np.where(test_y_0 != i, 0, test_y_0)
		test_y = np.where(test_y == i, 1, test_y)
	mods_by_number[i] = model(train_x_0, train_y, test_x_0, test_y, 2000, print_cost = True)
	
final_class = np.zeros((test_y_0.shape[1], 2))
for j in range(10):
	i = 0
	while i < test_y_0.shape[1]:
		#print (mods_by_number[j]["Y_prediction_test"].shape, final_class[i].shape)
		#print (j, mods_by_number[j]["Y_prediction_test"][0][i], final_class[i][1])
		if 	mods_by_number[j]["Y_prediction_test"][0][i] > final_class[i][1]:
			final_class[i][1] = mods_by_number[j]["Y_prediction_test"][0][i]
			final_class[i][0] = j
		i+=1

i = 0
wrongs = 0
while i < test_y_0.shape[1]:	
	if test_y_0[0][i] != final_class[i][0]:
		wrongs+=1
		print(test_y_0[0][i], final_class[i][0])
		plt.imshow(imgs[i])
		plt.show()
	i+=1
	
print(wrongs/test_y_0.shape[1]*100)
