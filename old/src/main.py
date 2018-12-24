"""
Run large BS experiments!
Use the 'run.sh' script with a specified YAML config and --gpus --batch_sizes flags
"""
# Built-in imports
import json, logging, numpy as np, os, plot_logs, sys, time
from shutil import copyfile
from tqdm import tqdm

# Add third-party libraries to path
sys.path.append("./lib/")
sys.path.extend([os.path.join("./lib", directory) for directory in os.listdir("./lib/")])

# Import models
from C1 import C1
from resnet import resnet34, resnet50
from wide_resnet import cast, create_dataset, data_parallel
from wide_resnet import wide_resnet

# Torch specific imports
import torch, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchnet.engine import Engine
from wide_resnet import create_dataset
from resnet import resnet34, resnet50
from C1 import C1
import torchnet as tnt

# Imports from this module
from lanczos import lanczos_bidiag
from optimizers import SGD, NoisySGD, ReservoirSGD, HessianVecSGD
from settings import CONFIG

TRAINING_SIZE = 50000 # CIFAR 10 SIZE TODO CALCULATE THIS FROM TRAINLOADER
MAX_BS_PER_GPUS = CONFIG['general'].max_bs_per_gpu

def parse_config():
    """
    Parse logging/GPU data from CONFIG
    """
    # Set logging level
    log_level = logging.DEBUG if CONFIG['logging'].verbose else logging.WARNING
    logging.basicConfig(level=log_level)

    # Parse available GPUs
    gpu_str = CONFIG['general'].gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    logging.debug("\tCUDA_VISIBLE_DEVICES: %s" % os.environ["CUDA_VISIBLE_DEVICES"])

    # Convert batch sizes to an actual list
    unparsed_batch_sizes = CONFIG['training'].batch_sizes
    batch_sizes = list(map(int, unparsed_batch_sizes.split(',')))
    batch_sizes.sort()
    CONFIG['training'].batch_sizes = batch_sizes

### DATASET LOADING
def create_wikitext_data_iterators(dataroot, batch_size, seq_len=35, num_gpus=4):
    return torchtext.datasets.WikiText2.iters(batch_size=batch_size, seq_len=35, test=None)

def create_cifar_data_iterators(dataroot, batch_size, num_classes=10, num_gpus=4):
    """
    Create training and test data loaders using CONFIG
    """
    dataset_name = 'CIFAR10' # TODO add this back as an arg maybe
    def create_iterator(is_train):
        return DataLoader(create_dataset(dataset_name, dataroot, is_train),
                          batch_size=batch_size,
                          shuffle=is_train,
                          num_workers=num_gpus,
                          pin_memory=torch.cuda.is_available())
    return create_iterator(True), create_iterator(False)

data_iterator_fns = {
    'CIFAR10': create_cifar_data_iterators,
    'WikiText': create_wikitext_data_iterators
}
allowed_datasets = set(data_iterator_fns.keys())

create_data_iterators = create_cifar_data_iterators #TODO get this for other datasets

### OPTIMIZER CREATION
def create_optimizer_fn(model_params, momentum, weight_decay, name='SGD',
                        mini_batch_size=128, other_params=None):
    # TODO make this a bit cleaner
    def create_optimizer(lr):
        if name == 'SGD':
            return SGD(model_params.values(), lr, \
                       momentum=momentum, weight_decay=weight_decay)
        elif name == 'NoisySGD':
            noise_factor = other_params['noise_factor']
            return NoisySGD(model_params.values(), lr, noise_factor, momentum, weight_decay)
        elif name == 'ReservoirSGD':
            scale = other_params['scale']
            is_distributed = other_params['distributed']
            max_reservoir_size = other_params['max_reservoir_size']
            num_gradients_to_sample = other_params['num_gradients_to_sample']
            # TODO make not distributed version of this? maybe
            if not is_distributed:
                raise ValueError('ReservoirSGD only supports distributed mode right now!')
            return ReservoirSGD(model_params.values(), lr, scale,
                                num_gradients_to_sample,
                                max_reservoir_size,
                                momentum, weight_decay)
        elif name == 'HessianVecSGD':
            noise_factor = other_params['noise_factor']
            return HessianVecSGD(model_params.values(), lr, noise_factor, momentum, weight_decay)

    return create_optimizer
# END OPTIMIZER CREATION

def load_checkpoint(checkpoint_str, model_params, optimizer):
    state_dict = torch.load(checkpoint)
    epoch = state_dict['epoch']
    iteration = state_dict['iteration']
    params_tensors = state_dict['model_params']
    for k, v in model_params.items():
        v.data.copy_(params_tensors[k])
    optimizer.load_state_dict(state_dict['optimizer'])
    return model_params, optimizer


def create_log_dirs(base_log_dir, batch_size, timestamp, other_params=None):
    name = CONFIG['general'].name
    bs = '_bs_{}_'.format(str(batch_size))
    lr = 'lr_{}_'.format(str(CONFIG['training'].learning_rate))
    optimizer = 'opt_{}'.format(str(CONFIG['training'].optimizer))
    log_dir_name = name + bs + lr + optimizer
    # add optimizer-specific attributes to the save directory name
    if other_params is not None:
        for k, v in other_params.items():
            log_dir_name += '_' + str(k) + '_' + str(v)
    # add timestamp to trial
    log_dir_name += '_' + timestamp
    save_dir = os.path.join(base_log_dir, log_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def create_cross_entropy_fn(model, model_name, model_params, num_gpus):
    def h(sample):
        inputs, targets = sample[:2]
        y = model(inputs)
        return F.cross_entropy(y, targets), y
    return h

def create_log_fn(model_params, save_dir):
    def log(t, state):
        iteration = t['iteration']
        save_dict = {
            'params': { k: v.data for k,v in model_params.items()  },
            'optimizer': state['optimizer'].state_dict(),
            'epoch': t['epoch'],
            'iteration': t['iteration']
        }
        if iteration % CONFIG['logging'].save_iters == 0:
            model_save_file = os.path.join(save_dir, 'model_{}.pt7'.format(str(iteration)))
            torch.save(save_dict, open(model_save_file, 'wb'))
        logname = os.path.join(save_dir, 'log.txt')
        with open(logname, 'a') as f:
            f.write(json.dumps(t) + '\n')

        logging.debug('\tLOG DATA:')
        for k, v in t.items():
            logging.debug('\t\t%s : %s' % (k, v))

    return log

## BEGIN TORCHNET ENGINE HOOKS

def create_on_sample_fn():
    def on_sample(state):
        state['sample'].append(state['train'])
    return on_sample

def create_on_forward_fn(classacc, meter_loss):
    def on_forward(state):
        classacc.add(state['output'].data, state['sample'][1].data)
        meter_loss.add(state['loss'].data[0])
    return on_forward

def create_on_start_fn(epoch, iteration=0):
    def on_start(state):
        state['epoch'] = epoch
        state['iteration'] = iteration
    return on_start

def create_on_update_fn(engine, cross_entropy, train_loader, test_loader, batch_size, log,
                        classacc, meter_loss, timer_train, timer_test, save_dir, period=8, iterations=25000):
    meter_loss_state_vars = ['n', 'sum', 'var', 'val', 'mean', 'mean_old', 'm_s', 'std']
    classacc_state_vars= ['sum', 'n']
    def on_update(state):
        if state['iteration'] % period == 0:
            # HACKY WAY TO STORE METER STATE
            meter_state = {}
            classacc_state = {}
            for var in meter_loss_state_vars:
                meter_state[var] = getattr(meter_loss, var)
            for var in classacc_state_vars:
                classacc_state[var] = getattr(classacc, var)
            # END HACKY

            # RESET METERS
            meter_loss.reset()
            classacc.reset()
            timer_test.reset()

            # GET TRAIN LOSS ON WHOLE DATASET
            engine.test(cross_entropy, train_loader)

            train_loss = meter_loss.value()
            train_acc = classacc.value()
            train_time = timer_train.value()

            meter_loss.reset()
            classacc.reset()
            timer_test.reset()

            # GET VALIDATION LOSS ON ALL DATA
            engine.test(cross_entropy, test_loader)
            test_acc = classacc.value()[0]

            # LOG
            vars_to_log = {
                'train_loss': train_loss[0],
                'train_acc': train_acc[0],
                'test_loss': meter_loss.value()[0],
                'test_acc': test_acc,
                'epoch': state['epoch'],
                'iteration': state['iteration'],
                'train_time': train_time,
                'test_time': timer_test.value()
            }
            log(vars_to_log, state)
            logging.info('\t==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % \
                     (save_dir, state['iteration'], iterations, test_acc))

            # HACKY WAY TO RESTORE METER STATE
            for var in meter_loss_state_vars:
                setattr(meter_loss, var, meter_state[var])
            for var in classacc_state_vars:
                setattr(classacc, var, classacc_state[var])
            # END HACKY
        state['iteration'] = state['iteration'] + 1
    return on_update

def create_on_start_epoch_fn(classacc, meter_loss, timer_train,
                             train_loader, epoch_step, lr_decay_ratio,
                             create_optimizer):
    def on_start_epoch(state):
        # classacc.reset()
        # meter_loss.reset()
        # timer_train.reset()
        state['iterator'] = tqdm(train_loader)
        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(lr * lr_decay_ratio)
    return on_start_epoch

## END TORCHNET ENGINE HOOKS

def create_model(model_name, num_classes):
    create_model_fn = {
        'resnet34': resnet34,
        'resnet50': resnet50,
        'C1': C1
    }
    assert model_name in create_model_fn.keys(), "must be one of {}".format(list(create_model_fn.keys()))
    logging.debug('\tCreating model {}'.format(model_name))
    model = DataParallel(create_model_fn[model_name](num_classes=num_classes))
    if CONFIG['general'].use_gpu:
        model = model.cuda()
    return model, dict(model.named_parameters())

def main(timestamp, batch_size, mini_batch_size=128):
    logging.debug('\tLoading %s data iterators' % CONFIG['dataset'].name)

    # Data loading
    dataset_name = CONFIG['dataset'].name
    dataroot = CONFIG['dataset'].datadir

    use_gpu = CONFIG['general'].use_gpu
    num_gpus = len(CONFIG['general'].gpus.split(',')) if use_gpu else 0
    num_classes = 10 if dataset_name == 'CIFAR10' else 100 # TODO make dataset-dependent lol
    train_loader, test_loader = create_data_iterators(
        dataroot, # TODO add dataset_name
        batch_size,
        num_classes,
        num_gpus)

    # Model construction
    model_name = CONFIG['model'].name
    model, model_params = create_model(model_name, num_classes)

    # Create optimizers
    logging.debug('\tCreating optimizers...')
    lr = CONFIG['training'].learning_rate
    momentum = CONFIG['training'].momentum
    weight_decay = CONFIG['training'].weight_decay
    optimizer_type = CONFIG['training'].optimizer

    # TODO clean this up so that other_params are extracted automatically
    if optimizer_type == 'SGD':
        other_params = None
    elif optimizer_type == 'NoisySGD':
        other_params = {"noise_factor": CONFIG['noisysgd'].noise_factor}
    elif optimizer_type == 'ReservoirSGD':
        if num_gpus == 0: raise ValueError('Need at least one GPU for now for ReservoirSGD!')
        other_params = {
            'scale': CONFIG['reservoir'].scale,
            'max_reservoir_size': CONFIG['reservoir'].max_reservoir_size,
            'num_gradients_to_sample': CONFIG['reservoir'].num_gradients_to_sample,
            'distributed': CONFIG['reservoir'].distributed
        }
    elif optimizer_type == 'HessianVecSGD':
        other_params = {"noise_factor": CONFIG['hessian_vec'].noise_factor}
    else:
        raise ValueError('Unsupported optimizer: %s' % optimizer_type)

    create_optimizer = create_optimizer_fn(model_params, momentum, weight_decay,
                                           optimizer_type, mini_batch_size,
                                           other_params)
    optimizer = create_optimizer(lr)

    epoch = 0
    iteration = 0
    # Load previous model checkpoint if it exists
    checkpoint = CONFIG['model'].checkpoint
    if checkpoint is not None:
        logging.info('\tLoading train state from checkpoint: %s' % checkpoint)
        model_params, optimizer = load_checkpoint(checkpoint,
                                                  model_params,
                                                  optimizer)

    # Calculate number of parameters in model
    num_parameters = sum(p.numel() for p in model_params.values())
    logging.debug('\tNumber of parameters in model: %d' % int(num_parameters))

    # Set up telemetry things
    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')

    # Create log directory for checkpoints
    save_dir = create_log_dirs(CONFIG['logging'].save_dir, batch_size, timestamp, other_params)
    logging.info('\tLogging to: {}'.format(save_dir))

    # Create log/test functions that we use in torchnet engine hooks
    cross_entropy = create_cross_entropy_fn(model, model_name, model_params, num_gpus)
    log = create_log_fn(model_params, save_dir)

    # Create torchnet engine hooks
    engine = Engine(create_graph=CONFIG['training'].optimizer == 'HessianVecSGD',
                    mini_batch_size=mini_batch_size)
    epoch_step_orig = CONFIG['training'].epoch_step
    if isinstance(epoch_step_orig, int):
        epoch_step = [epoch_step_orig]
    else:
        epoch_step = list(map(int, CONFIG['training'].epoch_step.split(',')))

    # CALCULATE NUMBER OF EPOCHS BASED ON ITERATIONS
    if hasattr(CONFIG['training'], 'iterations'):
        logging.debug('\tUsing iterations: {}'.format(CONFIG['training'].iterations))
        iters_per_epoch = TRAINING_SIZE // batch_size
        batch_period = 1 # batch_size // mini_batch_size
        epochs = batch_period * CONFIG['training'].iterations // iters_per_epoch
        # run at least this many epochs
        epochs = max(epochs, CONFIG['training'].epochs)
    else:
        epochs = CONFIG['training'].epochs
    lr_decay_ratio = CONFIG['training'].lr_decay_ratio

    logging.info('\tRUNNING FOR {} EPOCHS'.format(epochs))

    # on_sample = create_on_sample_fn()
    on_forward = create_on_forward_fn(classacc, meter_loss)
    on_start = create_on_start_fn(epoch, iteration)
    on_start_epoch = create_on_start_epoch_fn(classacc,
                                              meter_loss,
                                              timer_train,
                                              train_loader,
                                              epoch_step,
                                              lr_decay_ratio,
                                              create_optimizer)
    on_update = create_on_update_fn(engine, cross_entropy,
                                    train_loader,
                                    test_loader, batch_size,
                                    log,
                                    classacc, meter_loss,
                                    timer_train, timer_test,
                                    save_dir,
                                    period=CONFIG['logging'].evaluation_iters,
                                    iterations=CONFIG['training'].iterations)

    # Hook the torchnet engine up
    # engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_start'] = on_start
    engine.hooks['on_update'] = on_update
    # Start the training process!
    engine.train(cross_entropy, train_loader, epochs, optimizer)

if __name__ == '__main__':
    parse_config()

    num_gpus = len(CONFIG['general'].gpus.split(','))
    num_trials = CONFIG['general'].num_trials
    batch_sizes = CONFIG['training'].batch_sizes

    for trial in range(num_trials):
        # Randomly initialize
        timestamp = int(time.time())
        np.random.seed(timestamp)
        timestamp = str(timestamp)

        logging.info('\tBegin trial number {}'.format(trial))
        max_bs = MAX_BS_PER_GPUS * num_gpus
        for batch_size in batch_sizes:
            logging.info('\tBegin experiment with batch size {}'.format(batch_size))
            mini_batch_size = min(max_bs, batch_size)
            main(timestamp, batch_size, mini_batch_size)
