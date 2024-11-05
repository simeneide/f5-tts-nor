import os
import pickle
import re
import hashlib
from functools import wraps

def cache_output_to_pickle(cache_dir='.tmp_cache'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Ensure the cache directory exists
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            # Filter out the "cache" argument from kwargs
            kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'cache'}

            # Process arguments to create a unique cache key
            def process_arg(arg):
                if isinstance(arg, str) | isinstance(arg, float) | isinstance(arg, int):
                    return arg
                else:
                    return hashlib.md5(str(arg).encode()).hexdigest()

            args_processed = [process_arg(arg) for arg in args]
            kwargs_processed = {k: process_arg(v) for k, v in kwargs_filtered.items()}

            # Create a unique cache file name based on the function's input parameters
            cache_key = f"{func.__name__}_{args_processed}_{kwargs_processed}"
            cache_key = hashlib.md5(str(cache_key).encode()).hexdigest()
            # Replace invalid characters in the file name
            cache_key = re.sub(r'[^\w\s-]', '', cache_key)
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")

            use_cache = kwargs.get('cache', True)
            if use_cache and os.path.exists(cache_file):
                #print(f"Loading data from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                return result
            else:
                #print(f"Running function and caching output to: {cache_file}")
                result = func(*args, **kwargs)
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                return result
        return wrapper
    return decorator

if __name__ == "__main__":
    @cache_output_to_pickle()
    def foo(a, b, cache=True):
        return a + b

    # Example usage
    print(foo(a=1, b=2))  # Uses cache by default
    print(foo(1, 2, cache=False))  # Forces recalculation and updates cache
    print(foo(3, 4))  # Caches a different configuration