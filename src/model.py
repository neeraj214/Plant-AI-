import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetV2B0

def build_model(num_classes=38, base="mobilenetv2", input_size=224, dropout=0.2, train_base=False):
    if base == "mobilenetv2":
        base_model = MobileNetV2(input_shape=(input_size, input_size, 3), include_top=False, weights="imagenet")
        last_conv_name = "Conv_1"
    else:
        base_model = EfficientNetV2B0(input_shape=(input_size, input_size, 3), include_top=False, weights="imagenet")
        last_conv_name = "top_conv"
    base_model.trainable = train_base
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model, last_conv_name

def unfreeze_for_finetune(model, fraction=0.3):
    total = len(model.layers)
    cutoff = int(total * (1 - fraction))
    for i, layer in enumerate(model.layers):
        layer.trainable = i >= cutoff
    return model
