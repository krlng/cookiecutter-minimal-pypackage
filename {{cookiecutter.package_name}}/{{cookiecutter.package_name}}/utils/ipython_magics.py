from IPython.core.magic import register_line_magic, needs_local_scope

import os
import pandas as pd
@register_line_magic
@needs_local_scope
def cache(line, local_ns):
    lhs, func_call = line.split('=', 1)
    lhs = lhs.strip()
    func_call = func_call.strip()
    func_name = func_call.split('(', 1)[0]

    # Construct the file path
    file_path = os.path.join(DATA_DIR, f"{func_name}.parquet")

    # Check if the cache file exists
    if os.path.isfile(file_path):
        # Read from cache
        df = pd.read_parquet(file_path)
    else:
        # Execute the function and cache the result
        df = eval(func_call, globals(), local_ns)
        df.to_parquet(file_path)

    # Assign the result to the variable in the user's namespace
    local_ns[lhs] = df
