#%%
import torch
from pathlib import Path
import pandas as pd
import numpy as np
# %%
dp = Path().resolve()
exact = pd.read_csv(dp/'exact.csv')
causal = pd.read_csv(dp/'predictions_causal.csv')
baseline = pd.read_csv(dp/'predictions_standard.csv')

# %%
print('rel L2 causal:',np.linalg.norm(exact['value']-causal['value'])/np.linalg.norm(exact['value']))
print('rel L2 baseline:',np.linalg.norm(exact['value']-baseline['value'])/np.linalg.norm(exact['value']))
# %%
print('L2 causal:',np.linalg.norm(exact['value']-causal['value']))
print('L2 baseline:',np.linalg.norm(exact['value']-baseline['value']))
# %%
