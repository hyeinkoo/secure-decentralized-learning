import yaml


def load_parameters(file="config.yaml", key=None):
    with open(file, 'r') as f:
        params = yaml.safe_load(f)
    if key:
        return params[key]
    else:
        return params

def update_parameters(params, args):
    params["model"] = args.model
    params['dropout_rate'] = args.dropout
    params['rgm_params']['n_mix'] = args.n_mix
    params['rgm_params']['scale'] = args.scale
    params['rgm_params']['frac'] = args.frac
    return params


def print_args(args):
    print()
    print(f"data:\t\t{args.dataset}")
    print(f"model:\t\t{args.model}")
    print(f"adv_init:\t{args.adversarial_init}")
    print(f"RGM:\t\t{args.rgm}\n")
    print(f"n_clients:\t{args.num_clients}")
    print(f"dropout:\t{args.dropout}")
    print(f"n_epochs:\t{args.num_epochs}")
    print(f"batch_size:\t{args.batch_size}\n")
    print(f"n_mix:\t\t{args.n_mix}")
    print(f"scale:\t\t{args.scale}")
    print(f"frac:\t\t{args.frac}\n")