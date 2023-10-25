import numpy as np
import copy
import random
from tqdm import tqdm




def random_gradient_mixing(clients, n_mix, scale, frac, dh_params=[14,29]):

    ids = [c.id for c in clients]
    for c in clients:
        peer = random.sample(ids, len(ids))
        peer.remove(c.id)
        c.peer_ids = peer

    n_mix_min, n_mix_max = n_mix

    for _ in range(n_mix_max):
        for i, c1 in enumerate(clients):
            if not c1.mix or c1.n_mixed >= n_mix_min:
                continue
            while c1.peer_ids != []:
                c2_id = c1.peer_ids.pop(0)
                c2 = clients[ids.index(c2_id)]
                if c2.mix and c2_id not in c1.peer_dic.keys():
                    break
            exchange_gradients(c1, c2, dh_params, scale, frac)



def exchange_gradients(c1, c2, dh_params, scale, frac):

    g, p = dh_params

    r1 = c2.receive_proposal(*c1.send_proposal(g, p))
    r2 = c1.receive_proposal(*c2.send_proposal(g, p))

    if not (r1 and r2):  # already exchanged
        return

    c1.generate_mask(c2.id, p, frac)
    c2.generate_mask(c1.id, p, frac)

    c2.receive_gradients(c1.id, c1.send_gradients(c2.id, scale))
    c1.receive_gradients(c2.id, c2.send_gradients(c1.id, scale))

    c1.update_gradients(c2.id)
    c2.update_gradients(c1.id)

    c1.n_mixed += 1
    c2.n_mixed += 1


