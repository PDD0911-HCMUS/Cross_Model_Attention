from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import glob
import cv2
import os
from create_model_kg import *
import tensorflow as tf
import tensorflow_addons as tfa
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
'''CREATE DATASET'''

# dataAll = pd.read_csv('capLabel_All.csv')
# dataAll.head()
# dataJson = dataAll.to_json('capLabel_All.json', orient = 'records')
# =====> run this if dont have the annotaions json file #

root_dir = "datasets_incidents"
annotations_dir = os.path.join(root_dir, "annotations")
images_dir = os.path.join(root_dir, "incidents_cleaned")
tfrecords_dir = os.path.join(root_dir, "tfrecords")
annotation_file = os.path.join(annotations_dir, "capLabel_All.json")
print(annotation_file)
with open(annotation_file, "r") as f:
    annotations = json.load(f)

image_path_to_caption = collections.defaultdict(list)
for element in annotations:
    caption = f"{element['caption'].lower().rstrip('.')}"
    image_path = images_dir + '/' + element["image_id"]
    image_path_to_caption[image_path].append(caption)

image_paths = list(image_path_to_caption.keys())
print(f"Number of images: {len(image_paths)}")

train_size = 4047
valid_size = 677
captions_per_image = 1
images_per_file = 2000

train_image_paths = image_paths[:train_size]
num_train_files = int(np.ceil(train_size / images_per_file))
train_files_prefix = os.path.join(tfrecords_dir, "train")

valid_image_paths = image_paths[-valid_size:]
num_valid_files = int(np.ceil(valid_size / images_per_file))
valid_files_prefix = os.path.join(tfrecords_dir, "valid")

tf.io.gfile.makedirs(tfrecords_dir)


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_example(image_path, caption):
    feature = {
        "caption": bytes_feature(caption.encode()),
        "raw_image": bytes_feature(tf.io.read_file(image_path).numpy()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(file_name, image_paths):
    caption_list = []
    image_path_list = []
    for image_path in image_paths:
        captions = image_path_to_caption[image_path][:captions_per_image]
        caption_list.extend(captions)
        image_path_list.extend([image_path] * len(captions))

    with tf.io.TFRecordWriter(file_name) as writer:
        for example_idx in range(len(image_path_list)):
            example = create_example(
                image_path_list[example_idx], caption_list[example_idx]
            )
            writer.write(example.SerializeToString())
    return example_idx + 1


def write_data(image_paths, num_files, files_prefix):
    example_counter = 0
    for file_idx in tqdm(range(num_files)):
        file_name = files_prefix + "-%02d.tfrecord" % (file_idx)
        start_idx = images_per_file * file_idx
        end_idx = start_idx + images_per_file
        example_counter += write_tfrecords(file_name, image_paths[start_idx:end_idx])
    return example_counter


train_example_count = write_data(train_image_paths, num_train_files, train_files_prefix)
print(f"{train_example_count} training examples were written to tfrecord files.")

valid_example_count = write_data(valid_image_paths, num_valid_files, valid_files_prefix)
print(f"{valid_example_count} evaluation examples were written to tfrecord files.")

feature_description = {
    "caption": tf.io.FixedLenFeature([], tf.string),
    "raw_image": tf.io.FixedLenFeature([], tf.string),
}


def read_example(example):
    features = tf.io.parse_single_example(example, feature_description)
    print(features)
    raw_image = features.pop("raw_image")
    features["image"] = tf.image.resize(
        tf.image.decode_jpeg(raw_image, channels=3), size=(299, 299)
    )
    print(features)
    return features


def get_dataset(file_pattern, batch_size):

    return (
        tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
        .map(
            read_example,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        .shuffle(batch_size * 10)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        .batch(batch_size)
    )
'''DONE CREATE DATASET'''

'''CREATE MODEL'''
class DualEncoder(keras.Model):
    def __init__(self, text_encoder, image_encoder, modelAtt, temperature=1.0, **kwargs):
        super(DualEncoder, self).__init__(**kwargs)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.modelAtt = modelAtt
        self.temperature = temperature
        
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, training=False):
        # Place each encoder on a separate GPU (if available).
        # TF will fallback on available devices if there are fewer than 2 GPUs.
        with tf.device("/gpu:0"):
            # Get the embeddings for the captions.
            caption_embeddings = text_encoder(features["caption"], training=training)
        with tf.device("/gpu:1"):
            # Get the embeddings for the images.
            image_embeddings = vision_encoder(features["image"], training=training)
        return caption_embeddings, image_embeddings

    def compute_loss(self, caption_embeddings, image_embeddings):
        # logits[i][j] is the dot_similarity(caption_i, image_j).
        print("temperature: ", self.temperature)
        print("caption_embeddings: ", caption_embeddings)
        print("image_embeddings: ", image_embeddings)
        logits = (
            tf.matmul(caption_embeddings, image_embeddings, transpose_b=True)
            / self.temperature
        )
        print("logits: ", logits)
        # images_similarity[i][j] is the dot_similarity(image_i, image_j).
        images_similarity = tf.matmul(
            image_embeddings, image_embeddings, transpose_b=True
        )
        print("images_similarity: ", images_similarity)
        # captions_similarity[i][j] is the dot_similarity(caption_i, caption_j).
        captions_similarity = tf.matmul(
            caption_embeddings, caption_embeddings, transpose_b=True
        )
        print("captions_similarity: ", captions_similarity)
        # targets[i][j] = avarage dot_similarity(caption_i, caption_j) and dot_similarity(image_i, image_j).
        targets = keras.activations.softmax(
            (captions_similarity + images_similarity) / (2 * self.temperature)
        )
        print("targets: ", targets)
        # Compute the loss for the captions using crossentropy
        captions_loss = keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )
        print("captions_loss: ", captions_loss)
        # Compute the loss for the images using crossentropy
        images_loss = keras.losses.categorical_crossentropy(
            y_true=tf.transpose(targets), y_pred=tf.transpose(logits), from_logits=True
        )
        print("images_loss: ", images_loss)
        # Return the mean of the loss over the batch.
        print("loss: ", (captions_loss + images_loss) / 2)
        return (captions_loss + images_loss) / 2

    def train_step(self, features):
        with tf.GradientTape() as tape:
            # Forward pass
            caption_embeddings, image_embeddings = self(features, training=True)
            loss = self.compute_loss(caption_embeddings, image_embeddings)
        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, features):
        caption_embeddings, image_embeddings = self(features, training=False)
        loss = self.compute_loss(caption_embeddings, image_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


# df = pd.read_csv("list_cap_detail_KG_incidents.csv")
# trainDf, testDf = train_test_split(df, test_size=0.2)
# capsTrain = load_cap(trainDf)
# kgTrain = load_kg(trainDf)
# imagesTrain = load_image(trainDf, "datasets_incidents/")

# capsValid = load_cap(testDf)
# kgValid = load_kg(testDf)
# imagesValid = load_image(testDf, "datasets_incidents/")

# print(f"shape Train example: {capsTrain.shape}")
# print(f"shape Valid example: {capsTrain.shape}")

num_epochs = 21  # In practice, train for at least 30 epochs
batch_size = 32

vision_encoder = create_vision_encoder(
    num_projection_layers=1, projection_dims=256, dropout_rate=0.1
)
text_encoder = create_text_encoder(
    num_projection_layers=1, projection_dims=256, dropout_rate=0.1
)
modelAtt = cross_model_attention(text_encoder, vision_encoder)
# kg_encoder = create_kg_encoder(
#     num_projection_layers=1, projection_dims=256, dropout_rate=0.1
# )

dual_encoder = DualEncoder(text_encoder, vision_encoder, modelAtt, temperature=0.05)
# modelAtt.compile(
#     optimizer=tfa.optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-3 / 200), loss="mean_absolute_percentage_error", metrics=['accuracy']
#     )
dual_encoder.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.001), loss="mean_absolute_percentage_error"
)

'''DONE CREATE MODEL'''


'''GET DATA FOR TRAINING'''
print(f"Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")
print(f"Number of examples (caption-image pairs): {train_example_count}")
print(f"Batch size: {batch_size}")
print(f"Steps per epoch: {int(np.ceil(train_example_count / batch_size))}")
train_dataset = get_dataset(os.path.join(tfrecords_dir, "train-*.tfrecord"), batch_size)
valid_dataset = get_dataset(os.path.join(tfrecords_dir, "valid-*.tfrecord"), batch_size)
# Create a learning rate scheduler callback.
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3
)
# Create an early stopping callback.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
history = dual_encoder.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=valid_dataset,
    callbacks=[reduce_lr],
)
print("Training completed. Saving vision and text encoders...")
vision_encoder.save("vision_encoder")
text_encoder.save("text_encoder")
print("Models are saved.")

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["train", "valid"], loc="upper right")
#plt.savefig('Epoch21.png')
plt.show()

# print(dual_encoder.summary())
# plot_model(dual_encoder.modelAtt, to_file='model_modelAtt.png', show_shapes=True, show_layer_names=True)
# '''DONE'''





# mha = tfa.layers.MultiHeadAttention(head_size=256, num_heads=12)
# att_layer = mha([text_encoder.output, kg_encoder.output, vision_encoder.output])
# # att_layer = layers.Attention()([text_encoder.output, kg_encoder.output, vision_encoder.output])
# # att_layer2 = layers.Attention()([att_layer, att_layer, att_layer])
# # att_layer3 = layers.Attention()([att_layer2, att_layer2, att_layer2])
# # att_layer4 = layers.Attention()([att_layer3, att_layer3, att_layer3])
# # att_layer5 = layers.Attention()([att_layer4, att_layer4, att_layer4])
# Dense_1 = layers.Dense(256, activation="relu")(att_layer)
# Dense_2 = layers.Dense(128, activation="relu")(Dense_1)
# Dense_3 = layers.Dense(2, activation="relu")(Dense_2)
# Dense_4 = layers.Dense(1, activation="linear")(Dense_3)
# out_layer = layers.LayerNormalization()(Dense_4)
# modelAtt = keras.Model(inputs=[text_encoder.input, kg_encoder.input, vision_encoder.input], outputs=out_layer)
# num_epochs = 30  # In practice, train for at least 30 epochs
# batch_size = 3
# print(f"Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")
# #print(f"Number of examples (caption-image pairs): {train_example_count}")
# print(f"Batch size: {batch_size}")
# #print(f"Steps per epoch: {int(np.ceil(train_example_count / batch_size))}")
# # Create a learning rate scheduler callback.
# reduce_lr = keras.callbacks.ReduceLROnPlateau(
#     monitor="val_loss", factor=0.2, patience=3
# )
# # Create an early stopping callback.
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor="val_loss", patience=5, restore_best_weights=True)

# history = dual_encoder.fit([capsTrain, kgTrain, imagesTrain],imagesTrain, 
# 						validation_data=([capsValid, kgValid, imagesValid], imagesValid),
# 						batch_size=1, 
# 						epochs=70,
# 						callbacks=[reduce_lr])
# print("Training completed. Saving vision and text encoders...")
# vision_encoder.save("vision_encoder_7")
# text_encoder.save("text_encoder_7")
# print("Models are saved.")