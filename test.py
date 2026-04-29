import tensorflow as tf
from keras.models import load_model

#  Define custom activation FIRST
@tf.keras.utils.register_keras_serializable()
def hard_swish(x):
    return x * tf.nn.relu6(x + 3) / 6

# Now load model with custom_objects
model = load_model(
    "litefdnet_tdf.keras",
    custom_objects={"hard_swish": hard_swish},
    compile=False
)

print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

model.summary()