from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, Conv2DTranspose, BatchNormalization, Activation, Reshape, ConvLSTM2D
from keras.optimizers import Adam
import numpy as np
import cv2

class ConvBlock:
    def __init__(self, filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer

    def __call__(self, inputs):
        conv = Conv2D(self.filters, self.kernel_size, activation=self.activation, padding=self.padding, kernel_initializer=self.kernel_initializer)(inputs)
        conv = Conv2D(self.filters, self.kernel_size, activation=self.activation, padding=self.padding, kernel_initializer=self.kernel_initializer)(conv)
        return conv

class UpConvBlock:
    def __init__(self, filters, kernel_size=(2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer

    def __call__(self, inputs):
        up = Conv2DTranspose(self.filters, self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer)(inputs)
        up = BatchNormalization(axis=3)(up)
        up = Activation('relu')(up)
        return up

class ConvLSTMBlock:
    def __init__(self, filters, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.kernel_initializer = kernel_initializer

    def __call__(self, inputs):
        conv_lstm = ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding, return_sequences=self.return_sequences, go_backwards=self.go_backwards, kernel_initializer=self.kernel_initializer)(inputs)
        return conv_lstm

class BCDU_net_D3:
    def __init__(self, input_size=(256, 256, 3), out_channels=5):
        self.input_size = input_size
        self.out_channels = out_channels

    def build_model(self):
        N = self.input_size[0]
        inputs = Input(self.input_size)

        conv1 = ConvBlock(64)(inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = ConvBlock(128)(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = ConvBlock(256)(pool2)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = ConvBlock(512)(pool3)
        drop4_1 = Dropout(0.5)(conv4)

        conv4_2 = ConvBlock(512)(drop4_1)
        conv4_2 = Dropout(0.5)(conv4_2)

        merge_dense = concatenate([conv4_2, drop4_1], axis=3)
        conv4_3 = ConvBlock(512)(merge_dense)
        drop4_3 = Dropout(0.5)(conv4_3)

        up6 = UpConvBlock(256)(drop4_3)
        up6 = concatenate([up6, conv3], axis=3)
        conv6 = ConvBlock(256)(up6)

        up7 = UpConvBlock(128)(conv6)
        x1 = Reshape(target_shape=(1, N//2, N//2, 128))(conv2)
        x2 = Reshape(target_shape=(1, N//2, N//2, 128))(up7)
        merge7 = concatenate([x1, x2], axis=1)
        merge7 = ConvLSTMBlock(64)(merge7)

        conv7 = ConvBlock(128)(merge7)

        up8 = UpConvBlock(64)(conv7)
        x1 = Reshape(target_shape=(1, N, N, 64))(conv1)
        x2 = Reshape(target_shape=(1, N, N, 64))(up8)
        merge8 = concatenate([x1, x2], axis=1)
        merge8 = ConvLSTMBlock(32)(merge8)

        conv8 = ConvBlock(64)(merge8)
        conv8 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv9 = Conv2D(self.out_channels, 1, activation='sigmoid')(conv8)

        model = Model(inputs=inputs, outputs=conv9)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def save_predictions(self, predictions, image_ids, output_dir):
        for i, image_id in enumerate(image_ids):
            for j in range(self.out_channels):
                mask = (predictions[i, :, :, j] > 0.5).astype(np.uint8) * 255
                cv2.imwrite(f'{output_dir}/ISIC_{image_id}_attribute_{j}.png', mask)