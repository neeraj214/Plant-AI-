import tensorflow as tf
import numpy as np

def gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        idx = tf.argmax(predictions[0])
        loss = predictions[:, idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, denom)
    return heatmap.numpy()

def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = tf.image.resize(heatmap[..., None], image.shape[:2]).numpy().squeeze()
    colored = tf.keras.preprocessing.image.array_to_img(tf.keras.preprocessing.image.apply_affine_transform(np.dstack([heatmap, heatmap, heatmap]), theta=0))
    colored = np.array(colored)
    colored = tf.image.grayscale_to_rgb(tf.convert_to_tensor(colored[..., :1])).numpy()
    out = np.clip(alpha * colored + (1 - alpha) * image, 0, 255).astype(np.uint8)
    return out
