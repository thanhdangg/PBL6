import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization, Activation, Dropout, Conv2D, Flatten, Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet101
import streamlit as st

# Define Soft Attention Layer
class SoftAttention(tf.keras.layers.Layer):
    def __init__(self, ch, m, concat_with_x=False, aggregate=False, **kwargs):
        self.channels = int(ch)
        self.multiheads = m
        self.aggregate_channels = aggregate
        self.concat_input_with_scaled = concat_with_x
        super(SoftAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.i_shape = input_shape
        
        # Fix the kernel shape to match the number of channels (input_shape[-1])
        kernel_shape_conv2d = (3, 3, input_shape[-1], self.multiheads)  # (H, W, C, multiheads)
        self.out_attention_maps_shape = input_shape[0:1] + (self.multiheads,) + input_shape[1:-1]

        # Handle output shape based on aggregation and concatenation options
        if not self.aggregate_channels:
            self.out_features_shape = input_shape[:-1] + (input_shape[-1] + input_shape[-1] * self.multiheads,)
        else:
            if self.concat_input_with_scaled:
                self.out_features_shape = input_shape[:-1] + (input_shape[-1] * 2,)
            else:
                self.out_features_shape = input_shape

        # Correct weight shape for 2D convolution
        self.kernel_conv2d = self.add_weight(
            shape=kernel_shape_conv2d, initializer='he_uniform', name='kernel_conv2d')
        self.bias_conv2d = self.add_weight(
            shape=(self.multiheads,), initializer='zeros', name='bias_conv2d')

    def call(self, x):
        # Perform 2D convolution with the input shape
        conv2d = tf.nn.conv2d(x, filters=self.kernel_conv2d, strides=[1, 1, 1, 1], padding='SAME')
        conv2d = tf.nn.bias_add(conv2d, self.bias_conv2d)
        conv2d = Activation('relu')(conv2d)  # Shape: [batch_size, height, width, multiheads]

        # Transpose dimensions for further processing
        conv2d = tf.transpose(conv2d, perm=(0, 3, 1, 2))  # Shape: [batch_size, multiheads, height, width]

        # Reshape for softmax application
        conv2d = tf.reshape(conv2d, shape=(-1, self.multiheads, self.i_shape[1] * self.i_shape[2]))  # Shape: [batch_size, multiheads, height * width]

        softmax_alpha = tf.nn.softmax(conv2d, axis=-1)  # Apply softmax
        softmax_alpha = tf.reshape(softmax_alpha, shape=(-1, self.multiheads, self.i_shape[1], self.i_shape[2]))  # Correct the reshape call

        if not self.aggregate_channels:
            exp_softmax_alpha = tf.expand_dims(softmax_alpha, axis=-1)
            exp_softmax_alpha = tf.transpose(exp_softmax_alpha, perm=(0, 2, 3, 1, 4))  # Shape: [batch_size, H, W, multiheads, 1]

            x_exp = tf.expand_dims(x, axis=-2)  # Shape: [batch_size, H, W, 1, C]
            u = tf.multiply(exp_softmax_alpha, x_exp)  # Element-wise multiplication
            u = tf.reshape(u, shape=(-1, self.i_shape[1], self.i_shape[2], u.shape[-1] * u.shape[-2]))  # Shape: [batch_size, H, W, multiheads * C]
        else:
            exp_softmax_alpha = tf.transpose(softmax_alpha, perm=(0, 2, 3, 1))  # Shape: [batch_size, H, W, multiheads]
            exp_softmax_alpha = tf.reduce_sum(exp_softmax_alpha, axis=-1)  # Sum along multihead axis
            exp_softmax_alpha = tf.expand_dims(exp_softmax_alpha, axis=-1)  # Shape: [batch_size, H, W, 1]
            u = tf.multiply(exp_softmax_alpha, x)  # Element-wise multiplication with original input

        if self.concat_input_with_scaled:
            o = tf.concat([u, x], axis=-1)  # Concatenate the input and scaled output
        else:
            o = u

        return [o, softmax_alpha]


    def compute_output_shape(self, input_shape):
        # Output the feature map shape and attention map shape
        return [self.out_features_shape, self.out_attention_maps_shape]


# Build the Model
inputs = tf.keras.layers.Input(shape=(224, 224, 3))
model = ResNet101(include_top=False, input_tensor=inputs, weights="imagenet")
model.trainable = False

conv = MaxPooling2D(pool_size=(2, 2), padding="same")(model.output)
conv = BatchNormalization()(conv)

# Soft Attention Layer
attention_layer, map2 = SoftAttention(aggregate=True, m=16, concat_with_x=False, ch=int(conv.shape[-1]), name='soft_attention')(conv)
attention_layer = MaxPooling2D(pool_size=(2, 2), padding="same")(attention_layer)

# Combine the layers
conv = MaxPooling2D(pool_size=(2, 2), padding="same")(conv)
conv = concatenate([conv, attention_layer])
conv = Activation("relu")(conv)
conv = Dropout(0.5)(conv)

# Further convolution and pooling layers
conv = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv)
conv = BatchNormalization()(conv)
conv = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv)
conv = BatchNormalization()(conv)
conv = Conv2D(filters=512, kernel_size=(1, 1), activation="relu", padding="same", kernel_initializer='he_normal')(conv)
conv = BatchNormalization()(conv)

# Pooling and fully connected layers
conv = MaxPooling2D(pool_size=(4, 4), padding="same")(conv)
conv = Flatten()(conv)
conv = Dense(4096, activation="relu")(conv)
conv = Dense(4096, activation="relu")(conv)
conv = Dense(7, activation="softmax")(conv)

# Define the model
model = Model(inputs=inputs, outputs=conv, name="ResNet101_with_SoftAttention")

classes = {
    4: ('nv', 'melanocytic nevi'), 
    6: ('mel', 'melanoma'), 
    2: ('bkl', 'benign keratosis-like lesions'), 
    1: ('bcc', 'basal cell carcinoma'), 
    5: ('vasc', 'pyogenic granulomas and hemorrhage'), 
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),  
    3: ('df', 'dermatofibroma')
}

# Prediction Function
def predict_image(image):
    img = cv2.resize(image, (224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_idx = np.argmax(pred, axis=1)[0]
    return classes[class_idx], pred


# Streamlit interface
st.title("Image Classification with Soft Attention")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    # Predict the image
    label, pred = predict_image(image)
    
    # Display the result
    st.write(f"Predicted Class: {label[1]} ({label[0]})")
    st.write("Prediction Probabilities:")
    st.write(pred)