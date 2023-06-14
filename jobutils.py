"""============================================================================
Utility functions for running jobs on Della.
============================================================================"""

from   copy import copy
import datetime
import itertools
import os

# This allows me to test locally without runtime errors.
IS_LOCAL = 'gwg3' not in os.getcwd()


# -----------------------------------------------------------------------------

def run_jobs(script, args, root_dir):
    """Run jobs for entire input parameter space.
    """
    if IS_LOCAL:
        args.directory = f'/Users/gwg/projects/dynamic-rflvm/{root_dir}/' \
                         f'{get_now_str()}_{args.directory}'
    else:
        args.directory = f'/scratch/gpfs/gwg3/dynamic-rflvm/{root_dir}/' \
                         f'{get_now_str()}_{args.directory}'
    mkdir(args.directory)

    iterables   = []
    iter_fields = []
    for f in vars(args):
        a = getattr(args, f)
        if type(a) is list and len(a) > 1:
            iterables.append(a)
            iter_fields.append(f)

    need_multi_dirs = any([len(x) > 1 for x in iterables])

    for a in itertools.product(*iterables):
        for f in iter_fields:
            setattr(args, f, None)
        _args = copy(args)
        for i, f in enumerate(iter_fields):
            setattr(_args, f, a[i])

        if need_multi_dirs:
            _args.directory = gen_subdir_str(_args, iter_fields)

        mkdir(_args.directory)
        submit_job(script, _args)


# -----------------------------------------------------------------------------

def submit_job(script, args):
    """Run sbatch command based on script inputs.
    """
    mkdir(args.directory)
    sbatch_fname = f'{args.directory}/sbatch.sh'
    contents = gen_sbatch_file(script, args)
    with open(sbatch_fname, 'w+') as f:
        f.write(contents)
    if not IS_LOCAL:
        os.system(f'sbatch {sbatch_fname}')


# -----------------------------------------------------------------------------

def gen_subdir_str(args, iter_fields):
    """Return subdirectory name based on experimental setup.
    """
    subdir = []
    for k in iter_fields:
        v = getattr(args, k)
        k = k.replace('_', '')
        subdir.append(f'{k}{v}')
    subdir = '_'.join(subdir)
    return os.path.join(args.directory, subdir)


# -----------------------------------------------------------------------------

def mkdir(directory):
    """Make directory if necessary.
    """
    if not os.path.isdir(directory):
        os.system('mkdir %s' % directory)


# -----------------------------------------------------------------------------

def gen_sbatch_file(script, args):
    """Return contents of sbatch file based on experimental setup. For
    configuring SLURM for multiprocessing, see:

    https://askrc.princeton.edu/question/322/
    """
    cmds = []
    for key, val in vars(args).items():
        if isinstance(val, list):
            assert(len(val) == 1)
            val = val[0]
        cmds.append(f'--{key}={val}')
    cmds_str = ' '.join(cmds)
    mem = f'{args.mem}G'
    return f"""#!/bin/bash

#SBATCH --nodes=1                             # No. of nodes.
#SBATCH --ntasks=1                            # No. of tasks across all nodes.
#SBATCH --mem-per-cpu={mem}                   # No. per cpu-core
#SBATCH --time={args.wall_time}:00:00         # total run time limit (HH:MM:SS)

#SBATCH --output={args.directory}/out.txt

module load matlab/R2019a
module load anaconda3
module load rh/devtoolset/7
export PYTHONPATH=$HOME/software/lib/python3.7/site-packages:$PYTHONPATH

conda activate rflvm
cd /scratch/gpfs/gwg3/dynamic-rflvm
python -O {script} {cmds_str} --is_local=0
"""


# -----------------------------------------------------------------------------

def get_now_str():
    """Return date string for experiments folders, e.g.: '20180621'.
    """
    now     = datetime.datetime.now()
    day     = '0%s' % now.day if now.day < 10 else now.day
    month   = '0%s' % now.month if now.month < 10 else now.month
    now_str = '%s%s%s' % (now.year, month, day)
    return now_str
