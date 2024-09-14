import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras import layers, models # type: ignore

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()
        self.conv1 = layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv2 = layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same', activation='relu')
        
    def call(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        return conv2

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(num_filters)
        self.pool = layers.MaxPooling2D((2, 2))
        
    def call(self, inputs):
        conv = self.conv_block(inputs)
        pool = self.pool(conv)
        return conv, pool

class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        super(Bottleneck, self).__init__()
        self.conv_block = ConvBlock(num_filters)
        
    def call(self, inputs):
        return self.conv_block(inputs)

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        super(DecoderBlock, self).__init__()
        self.conv_transpose = layers.Conv2DTranspose(num_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.conv_block = ConvBlock(num_filters)
        
    def call(self, inputs, skip_features):
        conv_transpose = self.conv_transpose(inputs)
        concatenate = layers.concatenate([conv_transpose, skip_features])
        conv = self.conv_block(concatenate)
        return conv

class MultiTaskUNet(tf.keras.Model):
    def __init__(self, input_shape):
        super(MultiTaskUNet, self).__init__()
        
        # Define input shape
        self.inputs = layers.Input(shape=input_shape)

        # Encoder blocks
        self.enc1 = EncoderBlock(64)
        self.enc2 = EncoderBlock(128)
        self.enc3 = EncoderBlock(256)
        self.enc4 = EncoderBlock(512)
        
        # Bottleneck
        self.bottleneck = Bottleneck(1024)
                
        # Decoder blocks
        self.dec1 = DecoderBlock(512)
        self.dec2 = DecoderBlock(256)
        self.dec3 = DecoderBlock(128)
        self.dec4 = DecoderBlock(64)
        
        # output
        self.outputs = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')

    def call(self, inputs):
        # Encoder
        s1, p1 = self.enc1(inputs)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        d1 = self.dec1(b, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)
        
        # Output
        outputs = self.outputs(d4)
            
        model = Model(inputs=[inputs], outputs=[outputs])

        return model
