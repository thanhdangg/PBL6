import tensorflow as tf
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
        
        # Shared output block
        self.shared_conv = layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')
        
        # Multi-task outputs
        self.output1 = layers.Conv2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid', name='pigment_network')
        self.output2 = layers.Conv2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid', name='negative_network')
        self.output3 = layers.Conv2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid', name='streaks')
        self.output4 = layers.Conv2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid', name='milia_like_cysts')
        self.output5 = layers.Conv2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid', name='globules')
    
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
        
        # Shared output
        shared_output = self.shared_conv(d4)
        
        # Multi-task outputs
        out1 = self.output1(shared_output)
        out2 = self.output2(shared_output)
        out3 = self.output3(shared_output)
        out4 = self.output4(shared_output)
        out5 = self.output5(shared_output)
        
        return [out1, out2, out3, out4, out5]

# # Define the input shape (e.g., 256x256x3)
# input_shape = (256, 256, 3)
# inputs = layers.Input(shape=input_shape)

# # Instantiate the multi-task U-Net model
# model = MultiTaskUNet(input_shape)

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Print the model summary
# model.build(input_shape=(None, *input_shape))
# model.summary()
