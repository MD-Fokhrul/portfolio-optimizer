import argparse
import json


# load config file
def load_config(config_filename='config.json'):
    with open(config_filename, 'r') as f:
        return json.load(f)


# helper method for argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def avg_results(results, key, lookback=None):
    consider = results[key]
    if lookback:
        consider = consider[-lookback:]

    return sum(consider) / len(consider)
