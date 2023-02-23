import pandas as pd
import numpy as np
from scipy.stats import norm, laplace

# Step 0: Load the dataset
df = pd.read_stata('private_data_by_cells.dta')

# Step 1: Calculate Local Sensitivity

# Generate variables to fill in with key parameters in each cell g:
df['theta_g'] = np.nan     # estimate of interest
df['SE_theta_g'] = np.nan  # SE of estimate of interest
df['theta_g_d'] = np.nan  # estimate obtained when adding one observation
df['LS_g'] = np.nan       # local sensitivity

cells = df['cell'].unique()  # save list of cells in a variable

# Compute true statistic, SE, and local sensitivity for each cell
for g in cells:
    # Compute true statistic
    reg = df.query('cell == @g')[['kid_rank', 'parent_rank']].values
    reg_x = reg[:, 1]
    reg_y = reg[:, 0]
    reg_x = sm.add_constant(reg_x)
    model = sm.OLS(reg_y, reg_x)
    results = model.fit()

    # Save statistic and SE
    df.loc[df['cell'] == g, 'theta_g'] = results.params[1]*0.25 + results.params[0]
    df.loc[df['cell'] == g, 'SE_theta_g'] = results.bse[1]

    # Compute Local Sensitivity (LS) for each cell
    additional_obs = df[df['cell'] == g].shape[0] + 1
    df.loc[additional_obs] = np.nan
    df.loc[additional_obs, 'cell'] = g
    for i in range(4):
        rank_x = i // 2
        rank_y = i % 2
        df.loc[additional_obs, 'parent_rank'] = rank_x
        df.loc[additional_obs, 'kid_rank'] = rank_y

        reg = df.query('cell == @g')[['kid_rank', 'parent_rank']].values
        reg_x = reg[:, 1]
        reg_y = reg[:, 0]
        reg_x = sm.add_constant(reg_x)
        model = sm.OLS(reg_y, reg_x)
        results = model.fit()

        df.loc[df['cell'] == g, 'theta_g_d'] = results.params[1]*0.25 + results.params[0]

        # Compute LS as the max absolute difference between theta_g_d and theta_g
        abs_diff = np.abs(df.loc[df['cell'] == g, 'theta_g_d'] - df.loc[df['cell'] == g, 'theta_g'])
        df.loc[df['cell'] == g, 'LS_g'] = np.max([abs_diff, df.loc[df['cell'] == g, 'LS_g']])

    df.drop(df.tail(1).index, inplace=True)

# Step 2: Compute Maximum Observed Sensitivity (chi)
df['N_g'] = df.groupby('cell').transform('count')['kid_rank']
df['N_g_LS_g'] = df['N_g'] * df['LS_g']
chi = df['N_g_LS_g'].max()

# Step 3: Determine Privacy Parameter (epsilon)
df_collapsed = df.groupby('cell').agg({
    'theta_g': 'mean', 
    'SE_theta_g': 'mean',
    'N_g': 'first',
    'LS_g': 'first'
})
epsilons = np.arange(1, 11)
mse = []
for epsilon in epsilons:
    draws = 500
    noise