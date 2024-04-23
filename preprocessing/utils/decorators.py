from functools import wraps, partial
import multiprocessing
from tqdm import tqdm
import dill     # use dill to prevent pickle error


def _run_dill_encoded(payload):
    func, args, kwds = dill.loads(payload)
    return func(*args, **kwds)


def _apply_async(pool, func, args, kwds, callback):
    payload = dill.dumps((func, args, kwds))
    return pool.apply_async(_run_dill_encoded, (payload,), callback=callback)


def multi_processing(func=None, n_jobs=1, divide="auto", gather="dict"):
    """
    Multi Processing decorator
    :param func: target function in one progress
    :param n_jobs: the processing threads
    :param divide: the method to divide input parameters
    :param gather: the method to gather output results
    :return:
    """
    assert divide in ["auto"]
    assert gather in ["dict", "list", "sum"]
    if func is None:
        return partial(multi_processing, n_jobs=n_jobs, divide=divide, gather=gather)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # divide the data
        args = args[0]
        if type(args) is dict:
            key = list(args.keys())
            data = list(args.values())
            divide_args = [(key[i], data[i]) for i in range(len(data))]
        elif type(args) is list:
            divide_args = [(a,) for a in args]
        else:
            raise ValueError("Unknown input type {}".format(type(args)))

        # set gather type
        if gather == "dict":
            output = dict()
        elif gather == "list":
            output = list()
        elif gather == "sum":
            output = 0
        else:
            raise ValueError("Unknown gather method {}".format(gather))

        # set up multi pool
        pool = multiprocessing.Pool(n_jobs)
        pbar = tqdm(desc="N-JOBs={}".format(n_jobs), total=len(divide_args))
        # multi processing running
        # results = []
        for i in range(len(divide_args)):
            # TODO: call back update
            result = _apply_async(pool, func=func, args=divide_args[i], kwds=kwargs, callback=lambda x: pbar.update())
            if type(output) is dict:
                output.update(result.get())
            elif type(output) is list:
                output.append(result.get())
            elif type(output) is int:
                output += result.get()
            # pbar.update()

        pool.close()
        pool.join()
        pbar.close()

        return output
    return wrapper


if __name__ == "__main__":
    # @multi_processing(n_jobs=4, gather="sum")
    # @multi_processing(n_jobs=10, gather="list")
    def plus_A(num, A=2):
        return num + A

    a = [1,25,7,2,7,2,89,2,5,9]

    mfunc = multi_processing(func=plus_A, n_jobs=4, gather="sum")
    output = mfunc(a, A=2)

    plus_A(a, A=3)
