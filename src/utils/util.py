
def grad_sum(clients):
    gradsum, biassum = 0, 0
    for client in clients:
    	gradsum += client.gradients[0]
    	biassum += client.gradients[1]
    return gradsum, biassum


def inverted_sentence(inv, red, closest_sample, vectorize_layer, sequence_length):
    # Print the inverted sentence. ??? means that the inversion dict was not precise enough
    # We are working with floating points. Therefore, translation back over the hash values might not always
    # be perfect.

    # However, looking two cells above, we see the recall is 1, ie. extracting the embeddings works perfectly
    # we are able to perfectly have the embeddings and just need a better hash function for inversion
    # back to text
    _dit = vectorize_layer.get_vocabulary()

    resh_resc = closest_samples[0].reshape((-1, sequence_length))
    sentence = []
    for i in range(sequence_length):

        tok = inv[red(resh_resc[i])]
        if tok < 0:
            sentence.append(f"???")
        else:
            sentence.append(_dit[tok])

    sentence = " ".join(sentence)
    # inv[red(resh_resc[i])]
    return sentence

