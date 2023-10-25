import tensorflow as tf

def compute_gradients(model, loss, x, y):
    with tf.GradientTape() as tape:
        # Forward pass.
        logits = model(x)
        # Loss value for this batch.
        loss_value = loss(y, logits)

        # Get gradients of loss wrt the weights.
        gradients = tape.gradient(loss_value, model.trainable_weights)
    return gradients # [gradients[0], gradients[1]]  # we need grad & bias of the first layer
    # return [tf.cast(gradients[0], tf.float64), tf.cast(gradients[1], tf.float64)]
