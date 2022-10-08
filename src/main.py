from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img
   
# parameters for ImageDataGenerator

# featurewise_center=False,
# samplewise_center=False,
# featurewise_std_normalization=False,
# samplewise_std_normalization=False,
# zca_whitening=False,
# zca_epsilon=1e-06,
# rotation_range=0,
# width_shift_range=0.0,
# height_shift_range=0.0,
# brightness_range=None,
# shear_range=0.0,
# zoom_range=0.0,
# channel_shift_range=0.0,
# fill_mode='nearest',
# cval=0.0,
# horizontal_flip=False,
# vertical_flip=False,
# rescale=None,
# preprocessing_function=None,
# data_format=None,
# validation_split=0.0,
# interpolation_order=1,
# dtype=None

datagen = ImageDataGenerator(
    height_shift_range=100,
    shear_range = 0.2,
)
    
# Loading a sample image 
img = load_img('image path goes here') 

# Converting the input sample image to an array
x = img_to_array(img)
x = x.reshape((1, ) + x.shape) 
    
count = 0
for batch in datagen.flow(x, batch_size = 1,
                          save_to_dir ='assets', 
                          save_prefix ='image', save_format ='jpeg'):
    count += 1
    # set the count of image to be generated
    if count > 5:
        break