import argparse
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, 
    Concatenate, 
    Convolution2D, 
    MaxPooling2D, 
    UpSampling2D
)


def parse_args(args):
    parser = argparse.ArgumentParser(description='convert model')
    parser.add_argument(
        '--unet',
        help='path to unet weights',
        type=str,
        required=False,
        default='dann_zurich2mai_u_net.h5'
    )
    parser.add_argument(
        '--encoder',
        help='path to encoder weights',
        type=str,
        required=False,
        default='dann_zurich2mai_encoder.h5'
    )
    return parser.parse_args(args)


def get_encoder(input_shape=(None, None, 4)) -> Model: 
    inputs = Input(shape=input_shape, batch_size=1)

    conv1 = Convolution2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv2)

    return Model(inputs=inputs, outputs=conv3)


def get_unet(input_shape=(None, None, 32)): 
    inputs = Input(shape=input_shape, batch_size=1)

    conv1 = Convolution2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(512, (3, 3), activation='relu', padding='same')(pool3)
    feature = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv4)
    
    up5 = Concatenate()([Convolution2D(256, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(feature)), conv3])
    conv5 = Convolution2D(256, (3, 3), activation='relu', padding='same')(up5)
    conv5 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv5)

    up6 = Concatenate()([Convolution2D(128, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5)), conv2])
    conv6 = Convolution2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate()([Convolution2D(64, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv1])
    conv7 = Convolution2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv7)
    
    up8 = Convolution2D(16, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    conv8 = Convolution2D(16, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv8)

    x_out = Convolution2D(3, (1, 1), activation='sigmoid')(conv8)
    return Model(inputs=inputs, outputs=[x_out, feature])


def get_inference_model(encoder: Model, u_net: Model) -> Model:
    return Model(
        inputs=encoder.inputs,
        outputs=[
            u_net(
                encoder(
                    encoder.inputs
                )
            )[0]
        ]
    )


def convert_model(model: Model, save_path: str) -> None:
    # Export your model to the TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()
    open(save_path, "wb").write(tflite_model)

    # -----------------------------------------------------------------------------
    # That's it! Your model is now saved as model.tflite file
    # You can now try to run it using the PRO mode of the AI Benchmark application:
    # https://play.google.com/store/apps/details?id=org.benchmark.demo
    # More details can be found here (RUNTIME VALIDATION):
    # https://ai-benchmark.com/workshops/mai/2021/#runtime
    # -----------------------------------------------------------------------------


def main(args=None) -> None:
    args=parse_args(args)
    u_net_weights_path = args.unet
    encoder_weights_path = args.encoder

    print('loading none model')
    encoder = get_encoder(input_shape=(None, None, 4))
    encoder.load_weights('dann_zurich2mai_encoder.h5')
    u_net = get_unet()
    u_net.load_weights('dann_zurich2mai_u_net.h5')
    model = get_inference_model(encoder=encoder, u_net=u_net)
    print('export and save model')
    convert_model(model=model, save_path='model_none.tflite')

    print('loading none model')
    encoder = get_encoder(input_shape=(128, 128, 4))
    encoder.load_weights('dann_zurich2mai_encoder.h5')
    u_net = get_unet()
    u_net.load_weights('dann_zurich2mai_u_net.h5')
    model = get_inference_model(encoder=encoder, u_net=u_net)
    print('export and save model')
    convert_model(model=model, save_path='model_128x128.tflite')

    print('loading model')
    encoder = get_encoder(input_shape=(544, 960, 4))
    encoder.load_weights('dann_zurich2mai_encoder.h5')
    u_net = get_unet()
    u_net.load_weights('dann_zurich2mai_u_net.h5')
    model = get_inference_model(encoder=encoder, u_net=u_net)
    print('export and save model')
    convert_model(model=model, save_path='model.tflite')


if __name__ == '__main__':
    main()
