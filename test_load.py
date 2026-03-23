import tensorflow as tf

class CustomDense(tf.keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        if 'quantization_config' in config:
            del config['quantization_config']
        return super().from_config(config)

class CustomDropout(tf.keras.layers.Dropout):
    @classmethod
    def from_config(cls, config):
        if 'quantization_config' in config:
            del config['quantization_config']
        return super().from_config(config)

class CustomGlobalAveragePooling2D(tf.keras.layers.GlobalAveragePooling2D):
    @classmethod
    def from_config(cls, config):
        if 'quantization_config' in config:
            del config['quantization_config']
        return super().from_config(config)

custom_objects = {
    'Dense': CustomDense,
    'Dropout': CustomDropout,
    'GlobalAveragePooling2D': CustomGlobalAveragePooling2D
}

try:
    print("Loading model...")
    model = tf.keras.models.load_model('model.keras', compile=False, custom_objects=custom_objects)
    print("Success! Model loaded.")
except Exception as e:
    import traceback
    traceback.print_exc()
