"""============================================================================
Constants.
============================================================================"""


LOCAL_DIR  = './'
REMOTE_DIR = './'


# Argparse available options (`choices`).
# -----------------------------------------------------------------------------

BOPT_METHOD = [
    'ess',
    'map',
    'nuts'
]

CHECK_PARAMS = [
    'B',
    'K',
    'R',
    'W',
    'X',
    'XBR'
]

DATASETS = [
    'blobs',
    'bridges',
    'circles',
    'cifar',
    'congress',
    'covid-geo',
    'covid-time',
    'cmu',
    'fiji',
    'highschool',
    'hippo',
    'lorenz',
    'mnist',
    'mnistb',
    'montreal',
    'moons',
    'newsgroups',
    's-curve',
    'simdata1',
    'simdata2',
    'spam',
    'spikes',
    'yale'
]

EMISSIONS = [
    'bernoulli',
    'binomial',
    'gaussian',
    'multinomial',
    'negbinom',
    'poisson'
]

INIT_W = [
    'normal',
    'proj',
    'prior'
]

INIT_X = [
    'pca',
    'random'
]

KERNELS = [
    'rbf',
    'rbf_linear',
    'cauchy',
    'OU'
]

MODELS = EMISSIONS.copy()

PRIOR_X = [
    'normal',
    'ou',
    'rbf'
]

PRIOR_X_WU_2017 = [
    'ar1',
    'normal',
    'rbf'
]

XOPT_METHOD = [
    'map',
    'nuts'
]

PRIOR_X_KERNELS = [
    'ou',
    'rbf'
]
