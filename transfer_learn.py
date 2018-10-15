import glob
import matplotlib.pyplot as plt

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Count of files in this path and it's subfolders
def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])

# Count of subfolders directly below the path (aka our categories)
def get_num_subfolders(path):
    if not os.path.exists(path):
        return 0
    return  sum([len(d) for r, d, files in os.walk(path)])

# Image generater function
def create_img_generator():
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )


# Main code
Image_width, Image_height = 299, 299
Training_Epochs = 2
Batch_Size = 32
Number_FC_Neurons = 1024

train_dir = './data/train'
validate_dir = './data/validate'

num_train_samples = get_num_files(train_dir)
num_classes = get_num_subfolders(train_dir)
num_validate_samples = get_num_files(validate_dir)

num_epoch = Training_Epochs
batch_size = Batch_Size

# define data pre-processing
train_image_gen = create_img_generator()
test_image_gen = create_img_generator()


# Connect the image generator to a folder which contains the source image that the image generator alters
# Training image generator:
train_generator = train_image_gen.flow_from_directory(
    train_dir,
    target_size=(Image_width, Image_height),
    batch_size=batch_size,
    seed=420
)


# Training image generator:
validation_generator = train_image_gen.flow_from_directory(
    validate_dir,
    target_size=(Image_width, Image_height),
    batch_size=batch_size,
    seed=420
)


# Fully connected layer
InceptionV3_base_model = InceptionV3(
    weights='imagenet',
    include_top=False,      # excludes the final FC layer
)
print('[+] Inception v3 base model without last FC loaded.')

# Define the layers
L0 = InceptionV3_base_model.output
L1 = GlobalAveragePooling2D()(L0)
L2 = Dense(Number_FC_Neurons, activation='relu')(L1)
predictions = Dense(num_classes, activation='softmax')(L2)


# New model
model = Model(inputs=InceptionV3_base_model.input, outputs=predictions)
print(model.summary())


print('[+] Performing basic transfer Learning')

# Freeze all layers in the Inception V3 base model
for layer in InceptionV3_base_model.layers:
    layer.trainable = False

# Define model copile for basaic Transfer Learning
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# By using generators we can ask continute to request sample images and the generators will pull images from
# the training or validation folders and alter them slightly
history_transfer_learning = model.fit_generator(
    train_generator,
    epochs=num_epoch,
    steps_per_epoch=num_train_samples // batch_size,
    validation_data=validation_generator,
    validation_steps=num_validate_samples // batch_size,
    class_weight='auto'
)

# Save the model
model.save('inceptionv3-transfer-learning.model')


# Option 2 specific to Inception
print('\n[+] Fine tuning existing model')
Layers_To_Freeze = 172
for layer in model.layers[:Layers_To_Freeze]:
    layer.trainable = False
for layer in model.layers[Layers_To_Freeze:]:
    layer.trainable = True

model.compile(
    optimizer=SGD(lr=0.0001, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_transfer_learning = model.fit_generator(
    train_generator,
    epochs=num_epoch,
    steps_per_epoch=num_train_samples // batch_size,
    validation_data=validation_generator,
    validation_steps=num_validate_samples // batch_size,
    class_weight='auto'
)

model.save('inceptionv3-fine-tune.model')