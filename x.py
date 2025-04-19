#! /usr/bin/env python

import argparse
import multiprocessing
import os
import sys
import time

import concurrent.futures

from functools import partial, wraps

from tqdm import tqdm

import numpy as np


def timethis(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        td = te - ts
        if td > 0:
            print('func:%r args:[%r, %r] took: %2.6f sec' % (f.__name__, len(args), len(kw), td), flush=True, file=sys.stderr)
        return result
    return wrap


def q(label, a):
    print(f'{label}: dtype={a.dtype}, size={a.size}, shape={a.shape}, mean={np.mean(a)}', flush=True)


# @timethis
def create_permutations(path):
    a = np.random.randint(low=0, high=30, size=[500000, 30], dtype=np.int8)
    np.save(path, a)


@timethis
def load_permutations(path):
    a = np.load(path)
    return a


# @timethis
def simulation_func(config, permutations):
    return 1


# @timethis
def multi_simulation_func(config, batch, n):
    permutations = load_permutations(config['permutations'])
    # q(f'[{n}] pid={os.getpid()}, config={config}, permutations=', permutations)
    return [simulation_func(config, permutations) for i in range(batch)]


def test_save():
    create_permutations('test.npy')
    a = load_permutations('test.npy')
    assert a.shape == (500000, 30)


def test_one_simulation_0():
    create_permutations('test.npy')
    result = simulation_func(config={'permutations': 'test.npy'}, n=4)
    assert result == 16


def test_one_simulation_1():
    create_permutations('test.npy')
    result = simulation_func(config={'permutations': 'test.npy'}, n=32)
    assert result == 1024


@timethis
def main(args):
    create_permutations(args.permutations)

    # Some config param
    config = {
        'permutations': args.permutations,
        'someconfig': [],
    }

    # Curry the config param into the function to create a new function
    batch = np.ceil(args.num_simulations / args.max_workers).astype(np.int32)
    simulation_func_with_config = partial(multi_simulation_func, config, batch)

    start_time = time.time()

    results = []
    if args.max_workers == 1:
        results = simulation_func_with_config(0)
    else:
        print('#cores', multiprocessing.cpu_count(), '#workers', args.max_workers)

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            for sub_results in executor.map(simulation_func_with_config, range(args.max_workers)):
                results.extend(sub_results)

                end_time = time.time()
                delta_time = end_time - start_time
                mean_time = delta_time / len(results)
                print(f'#completed={len(results)}, total_time={delta_time}, mean_time={mean_time}')

    print('#results', len(results), 'sum', sum(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation')
    parser.add_argument('--permutations', help='Path to permutations numpy array', default='config.npy')
    parser.add_argument('-n', '--num_simulations', help='Number of simulations to run', type=int, default=1)
    parser.add_argument('--max_workers', help='Number of workers', type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    main(args)
