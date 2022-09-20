import os
import collections
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import cv2

# Suppressing tf.hub warnings
tf.get_logger().setLevel("ERROR")

def project_embeddings(
    embeddings, num_projection_layers, projection_dims, dropout_rate
):
    projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = layers.Dense(projection_dims)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Add()([projected_embeddings, x])
        projected_embeddings = layers.LayerNormalization()(x)
    return projected_embeddings

def create_vision_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained Xception model to be used as the base encoder.
    xception = keras.applications.Xception(
        include_top=False, weights="imagenet", pooling="avg"
    )
    # Set the trainability of the base encoder.
    for layer in xception.layers:
        layer.trainable = trainable
    # Receive the images as inputs.
    inputs = layers.Input(shape=(299, 299, 3), name="image")
    # Preprocess the input image.
    xception_input = tf.keras.applications.xception.preprocess_input(inputs)
    # Generate the embeddings for the images using the xception model.
    embeddings = xception(xception_input)
    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the vision encoder model.
    return keras.Model(inputs, outputs, name="vision_encoder")

def create_text_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the BERT preprocessing module.
    preprocess = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2",
        name="text_preprocessing",
    )
    # Load the pre-trained BERT model to be used as the base encoder.
    print(trainable)
    bert = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
    )
    # Set the trainability of the base encoder.
    bert.trainable = trainable
    # Receive the text as inputs.
    inputs = layers.Input(shape=(), dtype=tf.string, name="caption")
    # Preprocess the text.
    bert_inputs = preprocess(inputs)
    # Generate embeddings for the preprocessed text using the BERT model.
    embeddings = bert(bert_inputs)["pooled_output"]
    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the text encoder model.
    return keras.Model(inputs, outputs, name="text_encoder")

def create_kg_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the BERT preprocessing module.
    preprocess = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2",
        name="kg_preprocessing",
    )
    # Load the pre-trained BERT model to be used as the base encoder.
    print(trainable)
    bert = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
    )
    # Set the trainability of the base encoder.
    bert.trainable = trainable
    # Receive the text as inputs.
    inputs = layers.Input(shape=(), dtype=tf.string, name="kg")
    # Preprocess the text.
    bert_inputs = preprocess(inputs)
    # Generate embeddings for the preprocessed text using the BERT model.
    embeddings = bert(bert_inputs)["pooled_output"]
    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the text encoder model.
    return keras.Model(inputs, outputs, name="kg_encoder")

# vision_encoder = create_vision_encoder(
#     num_projection_layers=1, projection_dims=256, dropout_rate=0.1
# )
# text_encoder = create_text_encoder(
#     num_projection_layers=1, projection_dims=256, dropout_rate=0.1
# )


def cross_model_attention(tx_encoder, vs_encoder):
    mha = tfa.layers.MultiHeadAttention(head_size=256, num_heads=12)
    att_layer = mha([tx_encoder.output, vs_encoder.output])
    # att_layer = layers.Attention()([text_encoder.output, kg_encoder.output, vision_encoder.output])
    # att_layer2 = layers.Attention()([att_layer, att_layer, att_layer])
    # att_layer3 = layers.Attention()([att_layer2, att_layer2, att_layer2])
    # att_layer4 = layers.Attention()([att_layer3, att_layer3, att_layer3])
    # att_layer5 = layers.Attention()([att_layer4, att_layer4, att_layer4])
    Dense_1 = layers.Dense(256, activation="relu")(att_layer)
    Dense_2 = layers.Dense(128, activation="relu")(Dense_1)
    Dense_3 = layers.Dense(2, activation="relu")(Dense_2)
    Dense_4 = layers.Dense(1, activation="linear")(Dense_3)
    out_layer = layers.LayerNormalization()(Dense_4)
    return keras.Model(inputs=[tx_encoder.input, vs_encoder.input], outputs=out_layer,name="modelAtt")

def load_cap(df):
	caps = []
	for i in df["caption"]:
		caps.append(i)
	return np.array(caps)
def load_image(df, inputPath):
	images = []
	for i in df["filename"]:
		imagePath = inputPath + i
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (299, 299))
		images.append(image)
		
	return np.array(images)
def load_kg(df):
	kgs = []
	for i in df["concat"]:
		kgs.append(i)
	return np.array(kgs)