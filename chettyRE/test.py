#%% imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

#%% Load the dataset 
df = pd.read_stata("private_data_by_cells.dta")

# Define a local macro "omega_dist"
omega_dist = "normal"

df["theta_g"] = np.nan
df["SE_theta_g"] = np.nan
df["theta_g_d"] = np.nan
df["LS_g"] = np.nan

cells = np.unique(df["cell"])

#%% Loop through each cell in "cells"
for cell in cells:
    # Subset the data for the current cell
    cell_data = df[df["cell"] == cell]
    
    # Run a regression of "kid_rank" on "parent_rank"
    X = sm.add_constant(cell_data["parent_rank"])
    y = cell_data["kid_rank"]
    model = sm.OLS(y, X)
    results = model.fit()
    
    # Replace "theta_g" and "SE_theta_g" with the coefficient and standard error of "parent_rank", respectively
    df.loc[df["cell"] == cell, "theta_g"] = results.params[1]
    df.loc[df["cell"] == cell, "SE_theta_g"] = results.bse[1]
    
    # Add one observation to the current cell and update "theta_g_d" with the new coefficient obtained from running the regression on the expanded cell
    cell_data_expanded = cell_data.append(cell_data.iloc[0])
    X_expanded = sm.add_constant(cell_data_expanded["parent_rank"])
    y_expanded = cell_data_expanded["kid_rank"]
    model_expanded = sm.OLS(y_expanded, X_expanded)
    results_expanded = model_expanded.fit()
    df.loc[df["cell"] == cell, "theta_g_d"] = results_expanded.params[1]
    
    # Calculate the local sensitivity of the estimate
    perturbations = [("kid_rank", 1, "parent_rank", -1), ("kid_rank", -1, "parent_rank", 1), ("kid_rank", 1, "parent_rank", 1), ("kid_rank", -1, "parent_rank", -1)]
    max_diff = 0
    for perturbation in perturbations:
        X_perturb = X.copy()
        y_perturb = y.copy()
        X_perturb[perturbation[0]] += perturbation[1]
        y_perturb[perturbation[0]] += perturbation[1]
        X_perturb[perturbation[2]] += perturbation[3]
        y_perturb[perturbation[2]] += perturbation[3]
        model_perturb = sm.OLS(y_perturb, X_perturb)
        results_perturb = model_perturb.fit()
        diff = abs(results_perturb.params[1] - results.params[1])
        if diff > max_diff:
            max_diff = diff
    df.loc[df["cell"] == cell, "LS_g"] = max_diff
    
    # Drop the additional observation added in step c
    df.drop(df.tail(1).index, inplace=True)

# Group the dataset by cell and create two new variables
df_grouped = df.groupby('cell').agg(N_g=('kid_rank', 'count'))
df_grouped['N_g_LS_g'] = df_grouped['N_g'] * df_grouped['LS_g']

# Calculate chi
chi = df_grouped['N_g_LS_g'].max()

#%%
df.columns
# %%
