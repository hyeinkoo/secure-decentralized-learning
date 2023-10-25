import numpy as np
from copy import deepcopy
import os
import pickle


def rescale_gradients(gradient, bias):
    gradient = gradient.numpy()
    bias = bias.numpy()
    inverse_bias = 1 / bias
    rescaled_gradients = gradient * inverse_bias
    # remove the NaNs (for more compact representation in figure)
    rescaled_gradients = rescaled_gradients[:, ~np.isnan(rescaled_gradients).any(axis=0)].T

    return rescaled_gradients



def calculate_all_l2_distances(reconstructed_samples, original_samples):
    """

    """
    num_orig_samples = original_samples.shape[0]
    features = original_samples.shape[1]
    # features = np.prod(original_samples.shape[1:])

    l2_dists = np.zeros(
        (len(reconstructed_samples), len(original_samples)))  # how many reconstructed and how many original
    for i in range(len(original_samples)):
        l2_dist = np.linalg.norm((reconstructed_samples - original_samples[i].reshape(1, features)), axis=1)
        l2_dists[:, i] = l2_dist

    # replace the nan values
    l2_dists = np.nan_to_num(l2_dists, nan=3072)
    return l2_dists


def get_per_data_point_closest_sample_indices(distances_array):
    """Distances array should be in shape num_reconstructed samples, num_original samples
    Each cell gives the l2 dist between the reconstructed and the original sample"""
    closest_samples_indices = np.argmin(distances_array, axis=0)

    return closest_samples_indices


def calculate_l2distance_of_closest_sample(closest_samples, original_samples):
    """Given the closest (according to l2_norm, or VGG embedding) samples and the corresponding original samples, return an array of their l2 distances"""

    distances = np.linalg.norm(closest_samples - original_samples, axis=1)

    return distances


def calculate_attack_recall(distances, l2_dist_eps):
    """Looks for every training example, how many reconstructed samples lie in the l2_dist_eps ball around it.
    Returns per sample numbers of close reconstructions, and the percentage of images that got reconstructions"""

    # find all the entries that are smaller than the l2_dist_eps somewhere in the array
    smaller_mask = distances < l2_dist_eps

    per_example_sum = np.sum(smaller_mask, axis=0)

    # for what percentage of samples did we get a reconstruction
    percentage = np.sum(per_example_sum != 0) / per_example_sum.shape[0]

    return per_example_sum, percentage


def calculate_attack_precision(distances, l2_dist_eps):
    """Of all the reconstructed samples, what percentage is in the l2_dist_eps ball around any training sample?
    Measures hardness of reconstruction. """

    # find all the entries that are smaller than the l2_dist_eps somewhere in the array
    smaller_mask = distances < l2_dist_eps

    # now look row-wise if there is any
    per_element_true = np.any(smaller_mask, axis=1)

    percentage = np.sum(per_element_true) / per_element_true.shape[0]

    return percentage


def process_gradients(gradients, bias):
    """Given the gradients perform the following operations:
    1. Calculate Inverse Bias
    2. Multiply Gradients with Inverse Bias to Rescale Them
    3. Remove NaN Values
    4. Transpose, such that number of examples is first axis"""

    inverse_bias = 1 / bias
    rescaled_c = gradients * inverse_bias

    rescaled_c_no_nan = rescaled_c[:, ~np.isnan(rescaled_c).any(axis=0)]

    rescaled_c_no_nan = rescaled_c_no_nan.T  # transpose such that first axis is number of samples

    return rescaled_c_no_nan





##############

def evaluate_attack(client, batch_idx,
                    l2_distance=[0.0, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 10],
                    print_result=True, emb_layer=None):
    # Using trap weights
    data = client.data[batch_idx][0]
    if emb_layer:
        data = emb_layer(data)
        gradient = client.gradients[1].numpy()
        bias = client.gradients[2].numpy()
    else:
        gradient = client.gradients[0].numpy()
        bias = client.gradients[1].numpy()
    data = data.numpy().reshape(data.shape[0], -1)

    # Calculate and Display Success Metrics
    rescaled_gradients = process_gradients(gradient, bias)

    distances = calculate_all_l2_distances(rescaled_gradients, data)

    closest_samples_indices = get_per_data_point_closest_sample_indices(distances)
    closest_samples = rescaled_gradients[closest_samples_indices]
    l2_dists = calculate_l2distance_of_closest_sample(closest_samples, data)

    keys = ["shape", "minmax_l2", "recall", "precision"]
    values = [rescaled_gradients.shape, [np.min(l2_dists), np.max(l2_dists)]] + [
        dict([(l2, None) for l2 in l2_distance]) for _ in range(2)]

    eval_dic = dict([(key, value) for key, value in zip(keys, values)])

    for l2 in l2_distance:
        _, attack_recall = calculate_attack_recall(distances, l2)
        attack_precision = calculate_attack_precision(distances, l2)
        eval_dic["recall"][l2] = attack_recall
        eval_dic["precision"][l2] = attack_precision

    if print_result:
        print("shape:\t\t", eval_dic["shape"])
        print("l2 dist:\t", eval_dic["minmax_l2"], "\n")
        for key, dic in sorted(eval_dic.items()):
            if key in ["shape", "minmax_l2"]:
                continue
            print(f"-- {key} -- ")
            for l2, value in sorted(dic.items()):
                print("%.8f:\t" % l2, value)
            print("")

    return eval_dic, closest_samples



def sentence_reconstruction(closest_samples, vectorize_layer, inv, red, sequence_length=250):
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

    return " ".join(sentence)

    # inv[red(resh_resc[i])]


def text_reconstruction(org_sentence, recons_text, closest_samples_list, reconstruction_data, save=False, run=None,
                        print_result=False):

    reconst, reconst_rgm = recons_text
    closest_sample, closest_sample_rgm = closest_samples_list

    sentence = sentence_reconstruction(closest_sample, **reconstruction_data)
    sentence_rgm = sentence_reconstruction(closest_sample_rgm, **reconstruction_data)

    if print_result:
        print(f"ORG:\n{org_sentence}\n")
        print(f"REC:\n{sentence}\n")
        print(f"RGM:\n{sentence_rgm}\n")

    if save:
        text = '\n\n'.join([org_sentence, sentence, sentence_rgm])
        with open(f"{run}.txt", 'w') as f:
            f.write(text)


def eval_dic_update(global_dic, eval_dic):
    for key, value in eval_dic.items():
        if key in ["recall", "precision"]:
            for l2, l2_value in eval_dic[key].items():
                global_dic[key][l2].append(l2_value)
        else:
            global_dic[key].append(value)
    return global_dic


def summarize(rec_dic, rgm_dic, l2_distance, run, save=True):
    keys = ["shape", "minmax_l2", "recall", "precision"]
    values = [None, None] + [dict([(l2, None) for l2 in l2_distance]) for _ in range(2)]

    summary_rec = dict([(key, value) for key, value in zip(keys, deepcopy(values))])
    summary_rgm = dict([(key, value) for key, value in zip(keys, deepcopy(values))])

    for dic, summary_dic in zip([rec_dic, rgm_dic], [summary_rec, summary_rgm]):
        for key, value in dic.items():
            if key in ["recall", "precision"]:
                for l2, l2_values in dic[key].items():
                    summary_dic[key][l2] = np.average(l2_values)
            else:
                summary_dic[key] = np.average(dic[key], axis=0)
    if save:
        with open(os.path.join(run, "summary_rec.pkl"), 'wb') as f:
            pickle.dump(summary_rec, f)
        with open(os.path.join(run, "summary_rgm.pkl"), 'wb') as f:
            pickle.dump(summary_rgm, f)




# def write_evaluation(eval_dics, num_mixing, run, save=True):
#     rec_dic, rgm_dic = eval_dics
#
#     keys = ["shape", "minmax_l2", "recall", "precision"]
#     values = [None, None] + [dict([(l2, []) for l2 in l2_distance]) for _ in range(2)]
#
#     summary_rec = dict([(key, value) for key, value in zip(keys, deepcopy(values))])
#     summary_rgm = dict([(key, value) for key, value in zip(keys, deepcopy(values))])
#
#     text = "\tCol 1: before RGM\tCol 2: after RGM\n\n"
#     text += str(num_mixing) + '\n\n'
#
#     for c, (rec_e, rgm_e) in enumerate(zip(rec_eval, rgm_eval)):
#         text += f"--- Client {c} ---------------------\n"
#         text += f"shape:\t\t{rec_e[2]}\t\t{rgm_e[2]}\n"
#         text += f"L2 dist:\t{rec_e[1]}\t{rgm_e[1]}\n\n"
#
#         for l2 in l2_distance:
#             text += "recall    %.8f:\t%f\t%f\n" % (l2, rec_e[0][l2]["recall"], rgm_e[0][l2]["recall"])
#             text += "precision %.8f:\t%f\t%f\n\n" % (l2, rec_e[0][l2]["precision"], rgm_e[0][l2]["precision"])
#
#     if save:
#         with open(f"{run}.txt", 'w') as f:
#             f.write(text)