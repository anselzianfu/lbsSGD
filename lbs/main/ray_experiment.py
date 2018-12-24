"""
Connects to or directly creates a ray cluster and executes runs
from the specified grid search.

Each run trains a network under some configuration, and then
runs evaluation on all of its checkpoints.

Assumes all runs require 0 CPUs and 1 GPU. Syncronizes all
output logs into a local ./raydata directory (on the driver), and then moves
all these logs to the logroot.
"""

import os
import sys
import shutil
import yaml

from absl import app
from absl import flags
import ray
# monkey patch rllib dep to avoid bringing in gym and TF
ray.rllib = None
import ray.tune
from ray.tune import register_trainable, run_experiments

from lbs.main import train
from lbs import log
from lbs import evaluate

flags.DEFINE_integer(
    'self_host', 0, 'if >0, create a local ray '
    'cluster with the specified number '
    'of virtual GPUs')
flags.DEFINE_boolean(
    'cpu', False, 'use cpu only (for testing, only applies'
    ' when self_host is active)')
flags.DEFINE_string('port', '6379', 'ray port to connect to')
flags.DEFINE_string('server_port', '10000', 'tune server port')
flags.DEFINE_multi_string(
    'experiment_name', None, 'experiment name, should be a valid, empty s3 '
    'directory in the parameter s3 bucket. There should '
    'be as many experiment names as configs specified.')
flags.mark_flag_as_required('experiment_name')
flags.DEFINE_string(
    's3', None, 's3 bucket to upload runs to; just the bucket'
    ' name, not the full URI.')
flags.DEFINE_multi_string(
    'config', None, 'yaml filename of variants to be '
    'grid-searched over; can run multiple experiments '
    'at once')
flags.mark_flag_as_required('config')


def _verify_s3(bucket_name):
    if bucket_name is None:
        return
    import boto3
    from botocore.exceptions import ClientError
    s3 = boto3.resource('s3')
    try:
        s3.meta.client.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 403:
            raise ValueError(
                'Forbidden access to bucket {}'.format(bucket_name))
        elif error_code == 404:
            raise ValueError('Bucket {} does not exist'.format(bucket_name))
        else:
            raise e


def ray_train(config, status_reporter):
    """
    Entry point called by ray hypers tuning, remotely.

    config should be a dictionary with the usual command-line flags
    for train.py, but already parsed by yaml.
    logroot is assumed to NOT be defined (so that local logfiles are
    stored relative to the invocation cwd).
    """
    assert 'logroot' not in config
    # highly coupled, hacky flag-setting. bite me.
    argv = list(flags.flag_dict_to_args(config))
    argv = ['lbs.main.train'] + argv
    flags.FLAGS(argv)

    status_reporter(timesteps_total=0, done=0)
    train.main(argv)
    status_reporter(timesteps_total=1, done=0)
    # hacks on hacks
    experiment_dir = os.path.dirname(log.logging_directory())
    evaluate.single_experiment_result(experiment_dir, training_module=argv[0])
    status_reporter(timesteps_total=2, done=1)


def _main(_):
    _verify_s3(flags.FLAGS.s3)

    if flags.FLAGS.self_host:
        if flags.FLAGS.cpu:
            ray.init(num_cpus=flags.FLAGS.self_host)
        else:
            ray.init(num_gpus=flags.FLAGS.self_host)
    else:
        ip = ray.services.get_node_ip_address()
        ray.init(redis_address=(ip + ':' + flags.FLAGS.port))

    register_trainable('ray_train', ray_train)

    configs = []
    for c in flags.FLAGS.config:
        with open(c) as f:
            configs.append(yaml.load(f))

    cpu = 1 if flags.FLAGS.self_host and flags.FLAGS.cpu else 0
    resources = {'cpu': cpu, 'gpu': 1 - cpu}

    experiment_setting = {
        exp_name: {
            'run': 'ray_train',
            'trial_resources': resources,
            'stop': {
                'done': 1
            },
            'config': c,
            'local_dir': './raydata'
        }
        for exp_name, c in zip(flags.FLAGS.experiment_name, configs)
    }

    if flags.FLAGS.s3 is not None:
        bucket_path = 's3://' + flags.FLAGS.s3
        experiment_setting['upload_dir'] = bucket_path

    try:
        run_experiments(
            experiment_setting,
            server_port=int(flags.FLAGS.server_port),
            with_server=True)
    except ray.tune.error.TuneError as e:
        print('swallowing tune error {}'.format(e), file=sys.stderr)
    # extract stuff from ray...
    logroot = flags.FLAGS.logroot
    os.makedirs(logroot, exist_ok=True)
    for experiment in os.listdir('raydata'):
        if experiment not in flags.FLAGS.experiment_name:
            continue
        experiment_dir = os.path.join('raydata', experiment)
        for runname in os.listdir(experiment_dir):
            rundir = os.path.join(experiment_dir, runname)
            run_logdir = os.path.join(rundir, 'logs')
            if not os.path.isdir(run_logdir):
                continue
            loglist = os.listdir(run_logdir)
            if not loglist:
                continue
            if len(loglist) > 1:
                print('was not expecting multiple runs, but found {} in {}'
                      .format(loglist, run_logdir))
                print('going with first')
            logdir = os.path.join(run_logdir, loglist[0])
            dst = os.path.join(logroot, loglist[0])
            os.makedirs(dst, exist_ok=True)
            for seed in os.listdir(logdir):
                seeddir = os.path.join(logdir, seed)
                seeddst = os.path.join(dst, seed)
                if os.path.isdir(seeddst):
                    print('did not expect destination dir {} to already exist'
                          ', moving it to *-old'.format(seeddst))
                    ctr = 0
                    olddst = seeddst
                    while os.path.exists(olddst):
                        olddst = dst + '-old-{}'.format(ctr)
                        ctr += 1
                    shutil.move(seeddst, olddst)
                shutil.move(seeddir, seeddst)


if __name__ == "__main__":
    app.run(_main)
