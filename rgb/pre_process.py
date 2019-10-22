# import required libraries
from my_imports import *


# create csv file for the images and masks (this is just for reference)

def load_image(im_height=64, im_width=64):

	if not os.path.exists('image.csv'):
		base_dir = os.path.join(os.getcwd(),'data')

		# generate csv file for images
		image_folder = base_dir + '\\images\\'
		images_in_folder = sorted(os.listdir(image_folder))

		# constructing data frame and saving csv file
		df = pd.DataFrame()
		df['images'] = [image_folder + str(x) for x in images_in_folder]
		df.to_csv('image.csv', header=None)

		# generate csv file for masks
		mask_folder = base_dir + '\\masks\\'
		masks_in_folder = sorted(os.listdir(mask_folder))

		# constructing data frame and saving csv file
		df = pd.DataFrame()
		df['masks'] = [mask_folder + str(x) for x in masks_in_folder]
		df.to_csv('mask.csv', header=None)
		

	# import list of image and mask paths from the csv file
	# NOTE: you can directly load image from the folder without any need for creating and loading from
	#       a CSV file. However, its recommended to have a cvs file becasue it makes it easier to make
	#       changes in the data set or loading only part of the dataset

	import csv
	def get_path_csv(folder, csv_file):
		with open(os.path.join(folder, csv_file), 'r') as f:
			reader = csv.reader(f)
			meta_data = list(reader)
			path_list = [item[1] for item in meta_data]
		return path_list
	try:
		csvFile.close()
	except:
		print("")
		
	folder = os.getcwd()
	img_path = get_path_csv(folder, 'image.csv')
	mask_path = get_path_csv(folder, 'mask.csv')


	imgs = []
	for i in img_path:
		img_array = cv2.imread(i, 1)/255
		new_array = cv2.resize(img_array, (im_height, im_width))
		imgs.append(new_array)
		
	masks = []
	for i in mask_path:
		mask_array = cv2.imread(i, 1)/255
		new_mask = cv2.resize(mask_array, (im_height, im_width))
		masks.append(new_mask)
		
		
	# convert to numpy array
	imgs_np = np.array(imgs, dtype=np.float32)
	masks_np = np.array(masks, dtype=np.float32)
	
	return imgs_np, masks_np
