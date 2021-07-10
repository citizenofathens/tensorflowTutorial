import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if __name__ == '__main__':
    # Sequential model은
    # 모델에 다중 입력 또는 다중 출력이 있는 경우 적합하지 않다
    # 모든 레이어에 여러 입력 또는 여러 출력 적합하지 않다
    # 레이어 공유를 해야할 때
    # 비선형 토폴로지를 원할 때때적합하지 않다

    # Define Sequential model with 3 layers
    model = keras.Sequential(
        [
            layers.Dense(2, activation="relu", name="layer1"),
            layers.Dense(3, activation="relu", name="layer2"),
            layers.Dense(4, name="layer3"),
        ]
    )
    # Call model on a test input
    x = tf.ones((3, 3))
    y = model(x)
    """"
    # 위와 동일
    # Create 3 layers
    layer1 = layers.Dense(2, activation="relu", name="layer1")
    layer2 = layers.Dense(3, activation="relu", name="layer2")
    layer3 = layers.Dense(4, name="layer3")

    # Call layers on a test input
    x = tf.ones((3, 3))
    y = layer3(layer2(layer1(x)))
    """

    # 순차적 모델 생성

    #model = keras.Sequential()
    #model.add(layers.Dense(2, activation="relu"))
    #model.add(layers.Dense(3, activation="relu"))
    #model.add(layers.Dense(4))


    # 레이어를 만들고 처음에는 가중치가 없다
    layer = layers.Dense(3)
    layer.weights  # Empty



    # Call layer on a test input
    # 레이어를 만들고 입력을 넣었을 때 가중치 생성
    x = tf.ones((1, 4))
    y = layer(x)
    layer.weights  # Now it has weights, of shape (4, 3) and (3,)


    model = keras.Sequential(
        [
            layers.Dense(2, activation="relu"),
            layers.Dense(3, activation="relu"),
            layers.Dense(4),
        ]
    )  # No weights at this stage!

    # At this point, you can't do this:
    # model.weights

    # You also can't do this:
    # model.summary()

    # Call the model on a test input
    x = tf.ones((1, 4))
    y = model(x)
    print("Number of weights after calling the model:", len(model.weights))  # 6

    model.summary()


    # 현재 출력 포함하여 지금까지 모델 요약을 표시 할 때
    #
    model = keras.Sequential()
    model.add(keras.Input(shape=(4,)))
    model.add(layers.Dense(2, activation="relu"))

    model.summary()


    # input 객체는 레이어가 아니므로 model.layers의 일부로 표시 되지 않는다
    model.layers

    # 입력 형상을 미리 입력하면 가중치가 존재하기 때문에 model 정보를 볼 수 있다

    model = keras.Sequential()
    model.add(layers.Dense(2, activation="relu", input_shape=(4,)))

    model.summary()

    # symetric matrix?

    # tensordot

    model = keras.Sequential()
    model.add(keras.Input(shape=(250, 250, 3))) # RGB images
    model.add(layers.Conv2D(32, 5 , strides=2, activation="relu"))
    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.MaxPooling2D(3))

    # Can you guess what the current output shape is at this point? Probably not.
    # Let's just print it:
    model.summary()
    # maybe fail

    # The answer was: (40, 40, 32), so we can keep downsampling...


    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.MaxPooling2D(3))
    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.MaxPooling2D(2))

    # And now?
    model.summary()

    # Now that we have 4x4 feature maps, time to apply global max pooling.
    model.add(layers.GlobalMaxPooling2D())

    # Finally, we add a classification layer.
    model.add(layers.Dense(10))

    initial_model = keras.Sequential(
        [
            keras.Input(shape=(250, 250, 3)),
            layers.Conv2D(32, 5, strides=2, activation="relu"),
            layers.Conv2D(32, 3, activation="relu"),
            layers.Conv2D(32, 3, activation="relu"),
        ]
    )
    # 모든 중간 레이어의 출력을 추출하는 모델 생성
    feature_extractor = keras.Model(

        inputs=initial_model.inputs,
        outputs=[layer.output for layer in initial_model.layers],


    )

    # Call feature extractor on test input.
    x = tf.ones((1, 250, 250, 3))
    features = feature_extractor(x)



    # 하나의 레이어에서 특성만 추출하는 것과 유사
    initial_model = keras.Sequential(
        [
            keras.Input(shape=(250, 250, 3)),
            layers.Conv2D(32, 5, strides=2, activation="relu"),
            layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
            layers.Conv2D(32, 3, activation="relu"),
        ]
    )

    feature_extractor = keras.Model(
        inputs=initial_model.inputs,
        outputs=initial_model.get_layer(name="my_intermediate_layer").output,
    )
    # Call feature extractor on test input.
    x = tf.ones((1, 250, 250, 3))
    features = feature_extractor(x)


    """ 전이 학습 """
    # 가장 마지막 output layer를 동결하고 맨 위 레이어만 훈련

    # model = keras.Sequential(
    #     [ < br > keras.Input(shape=(784)) < br > layers.Dense(32, activation='relu'), < br > layers.Dense(32,
    #                                                                                                       activation='relu'), < br > layers.Dense(
    #     32, activation='relu'), < br > layers.Dense(
    #     10), < br >]) < br > < br >  # Presumably you would want to first load pre-trained weights.<br>model.load_weights(...)<br> <br># Freeze all layers except the last one.<br>for layer in model.layers[:-1]:<br>  layer.trainable = False<br> <br># Recompile and train (this will only update the weights of the last layer).<br>model.compile(...)<br>model.fit(...)