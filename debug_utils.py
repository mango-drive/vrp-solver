import os
import cProfile
import time
from pstats import SortKey
import pstats
import io
import matplotlib.pyplot as plt
import inspect
import csv


LOG_METHODS = "log", "init_population_2d", "init_population_1d"


def update_plot(plot):
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if plot:
                ax
                fname = func.__name__
                eval(f'plot_{fname}(result, ax, *args, **kwargs)')
                plt.pause(0.05)
            return result
        return wrapper
    return decorator

def save(should_save=False):
    def decorated_save(func):
        def save_wrapper(*args, **kwargs):
            fname = func.__name__
            if should_save:
                algo = None
                if 'algo' in kwargs.keys():
                    algo = kwargs["algo"]
                    if algo:
                        result_file = os.path.join(algo.results_dir, fname+'.csv')
                        ret, results = func(*args, **kwargs)
                        fieldnames = results[0].keys()
                        writer = csv.DictWriter(open(result_file, 'a'), fieldnames)
                        writer.writerows(results)
                        return ret, results
            else:
                return func(*args, **kwargs)
        return save_wrapper
    return decorated_save


def timeit(should_time=False):
    def decorated(func):
        def timeit_wrapper(*args, **kwargs):
            if should_time:
                fname = func.__name__
                start = time.time()
                ret = func(*args, **kwargs)
                elapsed = time.time() - start
                algo = None
                if "algo" in kwargs.keys():
                    algo = kwargs["algo"]
                if algo:
                    if fname not in algo.performance:
                        algo.performance[fname] = {
                            'ncalls': 0, 'avgt': 0, 'tot': 0}
                    perf = algo.performance[fname]
                    n = perf["ncalls"]
                    avg = perf["avgt"]
                    avg = (elapsed + n * avg) / (n + 1)
                    perf["avgt"] = avg
                    perf["ncalls"] += 1
                    perf["tot"] += elapsed
                return ret
            else:
                return ret
        return timeit_wrapper
    return decorated


def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE  
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return wrapper


def xprint(func, *args, **kwargs):
    caller = ""
    stack = inspect.stack()
    for i, f in enumerate(stack):
        fn = f.function
        if fn == func.__name__:
            caller = stack[i - 1].function
            break

    if caller in LOG_METHODS:
        print(*args, **kwargs)
