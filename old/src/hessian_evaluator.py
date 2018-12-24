#!/bin/python
import json
import logging
import numpy as np
from optimizers import SGD, NoisySGD, ReservoirSGD, HessianVecSGD
import os
import glob
import plot_logs
import sys
import pickle

# Add third-party libraries to path
sys.path.append("./lib/")
sys.path.extend([os.path.join("./lib", directory) for directory in os.listdir("./lib/")])

from settings import CONFIG
from shutil import copyfile
import time
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from wide_resnet import cast, create_dataset, data_parallel
from wide_resnet import wide_resnet
from resnet import resnet34, resnet50
from C1 import C1
from lanczos import Operator, lanczos_bidiag
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from main import create_model, create_data_iterators


def load_model(model, checkpoint_str):
    if CONFIG['training'].use_gpu:
        state_dict = torch.load(checkpoint_str)
    else:
        state_dict = torch.load(checkpoint_str, map_location=lambda storage, loc: storage)
    params_tensors = state_dict['params']
    dict(model.named_parameters()).update(params_tensors)
    #for k, v in model.named_parameters():
    #    v.data.copy_(params_tensors[k])

class FDHessianOperator(Operator):
    """
    Use  finite difference for Hessian Vec product calculation
    """
    def __init__(self, input, target, model, negate=False, eps=1e-3):
        self.model = model
        self.input = input
        self.target = target
        self.negate = negate
        self.eps = eps
        if CONFIG['training'].use_gpu:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def apply_adjoint(self, vec):
        return self.apply(vec)

    def set_lambda_max(self, lambda_max):
        """
        Set max eigenvalue if we want to negate things
        """
        self.lambda_max = lambda_max

    def apply(self, vec):
        self.prepare_grad(False)
        orig_grad_vec = self.grad_vec.clone()

        # save out old params and perturb them
        old_params_vec = parameters_to_vector(self.model.parameters()).clone()
        perturbed_params = old_params_vec + self.eps * vec
        vector_to_parameters(perturbed_params, self.model.parameters())

        # calculate the new gradient w.r.t. these perturbed parameters
        self.prepare_grad(False)
        new_grad_vec = self.grad_vec
        hessian_vec_prod = (new_grad_vec - orig_grad_vec) / self.eps

        # restore original parameters to model and finish
        vector_to_parameters(old_params_vec, self.model.parameters())
        self.grad_vec = orig_grad_vec
        if self.negate:
            assert hasattr(self, 'lambda_max'), 'must solve forward version first!'
            hessian_vec_prod -= float(self.lambda_max) * vec

        return hessian_vec_prod

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def prepare_grad(self, retain=True):
        """
        Compute gradient w.r.t loss over all parameters and vectorize
        """
        self.zero_grad()
        output = self.model(self.input)
        loss = F.cross_entropy(output, self.target)
        grad_dict = torch.autograd.grad(
            loss, self.model.parameters(), retain_graph=retain, create_graph=retain
        )
        self.grad_vec = parameters_to_vector(grad_dict)
        self.size = self.grad_vec.size


class HVPOperator(Operator):
    """
    Use *PyTorch* finite difference for Hessian Vec product calculation
    """
    def __init__(self, input, target, model, negate=False):
        self.model = model
        self.input = input
        self.target = target
        self.negate = negate
        if CONFIG['training'].use_gpu:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor


    def apply_adjoint(self, vec):
        return self.apply(vec)

    def apply(self, vec):
        self.prepare_grad(True)
        # HESSIAN VEC COMPUTATION
        grad_vec = self.grad_vec
        # compute the product
        grad_product = torch.sum(grad_vec * vec)
        grad_grad = torch.autograd.grad(
            grad_product, self.model.parameters()
        )
        # h_v_p = hessian_vec_product
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in grad_grad])
        if self.negate:
            assert hasattr(self, 'lambda_max'), 'must solve forward version first!'
            hessian_vec_prod.data.add_(-self.lambda_max, vec.data)
        return hessian_vec_prod

    def set_lambda_max(self, lambda_max):
        """
        Set max eigenvalue if we want to negate things
        """
        self.lambda_max = lambda_max

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def prepare_grad(self, retain=True):
        """
        Compute gradient w.r.t loss over all parameters and vectorize
        """
        self.zero_grad()
        output = self.model(self.input)
        loss = F.cross_entropy(output, self.target)
        grad_dict = torch.autograd.grad(
            loss, self.model.parameters(), retain_graph=retain, create_graph=retain
        )
        self.grad_vec = parameters_to_vector(grad_dict)
        self.size = self.grad_vec.size


def parse_batch_size(file_str):
    bs_ind = file_str.rindex('bs_')
    after_bs = file_str[bs_ind + 3:]
    batch_size = int(after_bs[:after_bs.index('_')])
    return batch_size


def load_model_files_by_bs(log_dir, experiment='baseline_resnet34_cifar10', track_during_training=False):
    """ return dict where dict[bs] = list of log files for bs 'bs' """
    # If we have model data throughout training,
    # We do everything we could have done, except that we first index 
    # by iteration
    if track_during_training:
        log_file_strs = glob.glob(os.path.join(log_dir, experiment + '_*/*.pt7'))
        log_files = {}
        for log_file in log_file_strs:
            start_ind = log_file.rfind('_')
            assert start_ind != -1, "can't find start of iteration str"
            end_ind = log_file.rfind('.')
            assert end_ind != -1, "can't find end of iteration str"
            iteration = int(log_file[start_ind+1:end_ind])
            if iteration not in log_files.keys():
                log_files[iteration] = [log_file]
            else:
                log_files[iteration].append(log_file)
    else:
        log_files = glob.glob(os.path.join(log_dir, experiment + '_*/model.pt7'))

    results = {}
    if track_during_training:
        for iteration, iteration_log_files in log_files.items():
            results[iteration] = {}
            for log_file in iteration_log_files:
                bs = parse_batch_size(log_file)
                if bs not in results[iteration].keys():
                    results[iteration][bs] = [log_file]
                else:
                    results[iteration][bs].append(log_file)
    else:
        for log_file in log_file_strs:
            bs = parse_batch_size(log_file)
            if bs not in results.keys():
                results[bs] = [log_file]
            else:
                results[bs].append(log_file)
    return results

class LambdaOperator(Operator):
    def __init__(self, apply_fn, size):
        self.apply_fn = apply_fn
        self.size = size

    def apply(self, x):
        return self.apply_fn(x)

def power_iteration(operator, steps=20):
    vector_size = operator.size()[:1] + (-1, 1)
    vec = torch.rand(vector_size)
    if CONFIG['training'].use_gpu:
        vec  = Variable(vec.cuda(), volatile=False)
    else:
        vec  = Variable(vec, volatile=False)
    for i in range(steps):
        vec = operator.apply(vec)
        vec = vec / torch.norm(vec)
    # Final calculation!
    lambda_max = torch.norm(operator.apply(vec)) / torch.norm(vec)
    return lambda_max

def min_power_iteration(operator, principal_eigenvalue=None, steps=20):
    if principal_eigenvalue is None:
        principal_eigenvalue = power_iteration(power_iteration, steps=steps)
    new_apply = LambdaOperator(lambda x: operator.apply(x) - principal_eigenvalue * x, operator.size)
    return -power_iteration(new_apply, steps) + principal_eigenvalue

def compute_top_bot_eigenvalues(model_files, input, target, k=2):
    """
    Compute top, bottom k eigenvalues for all models in this batch
    return (top_eigenval_means, bot_eigenval_means, top_eigenval_std, bot_eigenval_std)
    Each array is a length k vector with mean/std of top/bot i-th eigenvalue
    """
    top_eigenvals = []
    bot_eigenvals = []

    power_iteration_top = []
    power_iteration_bot = []

    n = len(model_files)
    model_name = CONFIG['model'].name
    model, _ = create_model(model_name, 10)
    # Hessian-vector product operators for Lanczos
    Op = FDHessianOperator
    top_model_op, bot_model_op = Op(input, target, model, negate=False), \
                                    Op(input, target, model, negate=True)
    for j, model_str in enumerate(model_files):
        logging.info('\tCalculating eigenvalues for model {0} of {1}'.format(j+1, n))

        # Reload model parameters
        #load_model(model, model_str)
        if CONFIG['training'].use_gpu:
            torch.cuda.empty_cache()

        # update models
        load_model(top_model_op.model, model_str)
        load_model(bot_model_op.model, model_str)
        # Compute gradient for these parameters
        top_model_op.prepare_grad()
        bot_model_op.prepare_grad()

        logging.info('\t\tCalculating top eigenvalues...')
        top_model_eigenvals = lanczos_bidiag(top_model_op, k)

        # calculate lambda_max for inverse power iteration
        STEPS = CONFIG['logging'].steps
        lambda_max = power_iteration(top_model_op, steps=STEPS).data[0]

        bot_model_op.set_lambda_max(lambda_max)
        logging.info('\t\tCalculating bottom eigenvalues...')
        bot_model_eigenvals = -lanczos_bidiag(bot_model_op, k) + lambda_max

        lambda_min  = min_power_iteration(top_model_op, lambda_max, steps=STEPS).data[0]

        top_model_eigenvals[::-1].sort()
        bot_model_eigenvals.sort()

        top_eigenvals.append(top_model_eigenvals)
        bot_eigenvals.append(bot_model_eigenvals)

        power_iteration_top.append(lambda_max)
        power_iteration_bot.append(lambda_min)

    top_means = np.mean(top_eigenvals, axis=0)
    bot_means = np.mean(bot_eigenvals, axis=0)
    top_std   = np.std(top_eigenvals, axis=0)
    bot_std   = np.std(bot_eigenvals, axis=0)

    power_top = np.mean(power_iteration_top)
    power_bot = np.mean(power_iteration_bot)

    return top_means, bot_means, top_std, bot_std, power_top, power_bot


if __name__ == '__main__':
    # Set logging level
    log_level = logging.DEBUG if CONFIG['logging'].verbose else logging.WARNING
    logging.basicConfig(level=log_level)

    # Parse available GPUs
    if CONFIG['training'].use_gpu:
        gpu_str = CONFIG['general'].gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    logging.debug("\tCUDA_VISIBLE_DEVICES: %s" % os.environ["CUDA_VISIBLE_DEVICES"])

    log_dir = CONFIG['logging'].log_dir
    logging.info('Loading models from directory: {}'.format(log_dir))

    track_during_training = CONFIG['general'].track_during_training

    experiment = CONFIG['logging'].experiment_name
    model_file_strs = load_model_files_by_bs(log_dir, experiment, track_during_training)
    assert len(model_file_strs) > 0, "Need more than one log!!!"

    logging.info('Loading data...')
    num_gpus = len(CONFIG['general'].gpus.split(','))
    import time
    import numpy as np
    np.random.seed(int(time.time()))
    loader, _ = create_data_iterators('./data/cifar',
                                     128,
                                     10,
                                     num_gpus)
    input, target = next(iter(loader))
    if CONFIG['training'].use_gpu:
        input = Variable(input.cuda(), volatile=False)
        target = Variable(target.cuda())
    else:
        input = Variable(input, volatile=False)
        target = Variable(target)

    results = {}

    k = CONFIG['logging'].k
    if track_during_training:
        for iteration in model_file_strs.keys():
            results[iteration] = {}
            for bs, model_files in model_file_strs[iteration].items():
                logging.info("{0} models for batch size {1}".format(len(model_files), bs))
                top_means, bot_means, top_std, bot_std, power_top_mean, power_bot_mean = compute_top_bot_eigenvalues(model_files, input, target, k=k)
                logging.info('\tTOP EIGENVALS: {}'.format(top_means))
                logging.info('\tBOT EIGENVALS: {}'.format(bot_means))
                results[iteration][bs] = {
                    'top_means': top_means,
                    'bot_means': bot_means,
                    'top_stds': top_std,
                    'bot_stds': bot_std,
                    'power_top_mean': power_top_mean,
                    'power_bot_mean': power_bot_mean
                }
    else:
        for bs, model_files in model_file_strs.items():
            logging.info("{0} models for batch size {1}".format(len(model_files), bs))
            top_means, bot_means, top_std, bot_std, power_top_mean, power_bot_mean = compute_top_bot_eigenvalues(model_files, input, target, k=k)
            logging.info('\tTOP EIGENVALS: {}'.format(top_means))
            logging.info('\tBOT EIGENVALS: {}'.format(bot_means))
            results[bs] = {
                'top_means': top_means,
                'bot_means': bot_means,
                'top_stds': top_std,
                'bot_stds': bot_std,
                'power_top_mean': power_top_mean,
                'power_bot_mean': power_bot_mean
            }

    logging.info('Saving eigenvalue results to eigenval_results.pkl')
    pickle.dump(results, open('{}_eigenval_results.pkl'.format(experiment), 'wb'))
