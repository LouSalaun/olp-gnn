'''
© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import os
import sys
import numpy as np
from channel_generation import create_channel_nlos
from olp_precoding import data_generation_olp
#from pypapi import events


N = int(sys.argv[1])  # number of samples/graphs to generate
M = int(sys.argv[2])  # number of APs
K = int(sys.argv[3])  # number of UEs
mor = sys.argv[4]
precoder = sys.argv[5]  # precoding algorithm: olp, mr, zf
min_ap_per_ue = int(sys.argv[6])  # minimum number of APs serving each UE
# probability of a channel being connected
connection_proba = float(sys.argv[7])

# If FLOPS count is not required, then set papi_events to None
papi_events = None#[events.PAPI_DP_OPS]  # None

assert precoder in ['olp', 'mr', 'zf'], ("precoder must be one of \'olp\', "
                                         "\'mr\' and \'zf\'.")

def my_channel_generation(M, K):
    return create_channel_nlos(mor, M, K)

def my_mask_generation(M, K):
    '''
    Generate a mask with random connections between UEs and APs. Each channel
    is connected with probability 'connection_proba'. Each UE is connected to
    at least 'min_ap_per_ue' APs.
    '''
    mask = np.zeros((M, K), dtype=bool)
    for k in range(K):
        ue_random_connections = \
            np.random.choice([True, False], size=M,
                             p=[connection_proba, 1-connection_proba])
        n_connected_ap = np.sum(ue_random_connections)
        if n_connected_ap < min_ap_per_ue:
            indices = np.random.choice(np.nonzero(~ue_random_connections)[0],
                                        min_ap_per_ue-n_connected_ap,
                                        replace=False)
            ue_random_connections[indices] = True
        mask[:, k] = ue_random_connections

    return mask


print(('Generating {} samples using {} for {} NLoS model with {} APs and '
       '{} UEs:').format(N, precoder, mor, M, K))

if precoder == 'olp':
    data = data_generation_olp(N, my_channel_generation, M, K, papi_events,
                               mask_gen=my_mask_generation)
elif precoder == 'zf':
    assert False, "ZF precoding with mask not implemented"
else:
    assert False, "MR precoding with mask not implemented"

# Create 'data' folder if it does not already exists
os.makedirs('data', exist_ok=True)

basefilename = 'data/data_{}_{}_{}_{}'.format(precoder, mor, M, K)
if os.path.exists(basefilename+'.npz'):
    i = 1
    filename = "{}({})".format(basefilename, i)
    while os.path.exists(filename+'.npz'):
        i += 1
        filename = "{}({})".format(basefilename, i)
    np.savez(filename, **data)
else:
    np.savez(basefilename, **data)
