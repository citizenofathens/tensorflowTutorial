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