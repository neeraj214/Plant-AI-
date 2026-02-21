import tensorflow as tf

def representative_dataset_gen(dataset, max_samples=200):
    count = 0
    for x, _ in dataset:
        for i in range(x.shape[0]):
            yield [x[i:i+1]]
            count += 1
            if count >= max_samples:
                return

def convert_to_tflite_int8(model, rep_dataset, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(rep_dataset)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    return out_path
