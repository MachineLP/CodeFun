
BN_AXIS = 3

def make_basic_block_base(inputs, filter_num, stride=1):
    x = tf.keras.layers.Conv2D(filters=filter_num,
                                        kernel_size=(3, 3),
                                        strides=stride,
                                        kernel_initializer='he_normal',
                                        padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization(axis=BN_AXIS)(x)
    x = tf.keras.layers.Conv2D(filters=filter_num,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        kernel_initializer='he_normal',
                                        padding="same")(x)
    x = tf.keras.layers.BatchNormalization(axis=BN_AXIS)(x)

    shortcut = inputs
    if stride != 1:
        shortcut = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=stride,
                                            kernel_initializer='he_normal')(inputs)
        shortcut = tf.keras.layers.BatchNormalization(axis=BN_AXIS)(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x

def make_basic_block_layer(inputs, filter_num, blocks, stride=1):
    x = make_basic_block_base(inputs, filter_num, stride=stride)

    for _ in range(1, blocks):
        x = make_basic_block_base(x, filter_num, stride=1)

    return x

class DeepHomoModel_:
    """Class for Keras Models to predict the corners displacement from an image. These corners can then get used 
    to compute the homography.

    Arguments:
        pretrained: Boolean, if the model is loaded pretrained on ImageNet or not
        input_shape: Tuple, shape of the model's input 
    Call arguments:
        input_img: a np.array of shape input_shape
    """

    def __init__(self, pretrained=False, input_shape=(256, 256), pooling="avg", layer_params=[2, 2, 2, 2]):

        self.input_shape = input_shape
        img_input = tf.keras.layers.Input(shape=(self.input_shape[0], self.input_shape[1], 3))

        x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
        x = tf.keras.layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
        x = tf.keras.layers.BatchNormalization(axis=BN_AXIS, name='bn_conv1')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)


        x = make_basic_block_layer(x, filter_num=64,
                                    blocks=layer_params[0])
        x = make_basic_block_layer(x, filter_num=128,
                                         blocks=layer_params[1],
                                         stride=2)
        x = make_basic_block_layer(x, filter_num=256,
                                         blocks=layer_params[2],
                                         stride=2)
        x = make_basic_block_layer(x, filter_num=512,
                                         blocks=layer_params[3],
                                         stride=2)
        
        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)

        
        dense_name_base = "full_" + str(2)
        for indx, neuron in enumerate([512, 512, 256, 128]):
            x = tf.keras.layers.Dense(
                neuron, name=dense_name_base + str(neuron) + "_" + str(indx))(x)


        x = tf.keras.layers.Dense(8, name="full_2" + "output")(x)
        outputs = tf.keras.layers.Activation("tanh")(x)

        self.model = tf.keras.models.Model(
            inputs=[img_input], outputs=outputs, name="DeepHomoPyramidalFull"
        )

        self.preprocessing = _build_homo_preprocessing(input_shape)

    def __call__(self, input_img):

        img = self.preprocessing(input_img)
        corners = self.model.predict(np.array([img]))

        return corners
    
    def load_weights(self, weights_path):
        try:
            self.model.load_weights(weights_path)
            print("Succesfully loaded weights from {}".format(weights_path))
        except:
            orig_weights = "Randomly"
            print(
                "Could not load weights from {}, weights will be loaded {}".format(
                    weights_path, orig_weights
                )
            )




def _build_resnet18():
    """Builds a resnet18 model in keras from a .h5 file.

    Arguments:

    Returns:
        a tf.keras.models.Model
    Raises:
    """
    resnet18_path_to_file = tf.keras.utils.get_file(
        RESNET_ARCHI_TF_KERAS_NAME,
        RESNET_ARCHI_TF_KERAS_PATH,
        RESNET_ARCHI_TF_KERAS_TOTAR,
    )

    resnet18 = tf.keras.models.load_model(resnet18_path_to_file)
    resnet18.compile()

    inputs = resnet18.input
    outputs = resnet18.layers[-2].output

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name="custom_resnet18")

  
  
  
