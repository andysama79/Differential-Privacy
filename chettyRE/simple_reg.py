#%% imports
import numpy as np
import pandas as pd
from scipy.stats import norm, laplace

#%% load data
data = pd.read_stata("private_data_by_cells.dta")

data["theta_g"] = np.nan    #*   estimate of interest
data["SE_theta_g"] = np.nan #*   SE of estimate of interest
data["theta_g_d"] = np.nan  #*   estimate obtained when adding one observation
data["LS_g"] = np.nan       #*   local sensitivity

cells = data["cell"].unique()

data.head()
# %%    Compute true statistic, SE, and local sensitivity for each cell
for g in cells:
    #*  run regression to estimate theta_g and SE_that_g
    sub = data[data["cell"] == g]
    X = np.column_stack((sub["parent_rank"], np.ones_like(sub["kid_rank"])))
    y = sub["kid_rank"]
    b = np.linalg.inv(X.T @ X) @ X.T @ y
    data.loc[data["cell"] == g, "theta_g"] = b[0] * 0.25 + b[1]
    data.loc[data["cell"] == g, "SE_theta_g"] = np.sqrt(np.sum((y - X @ b) ** 2) / (sub.shape[0] - 2))

    #*  Compute Local Sensitivity (LS) for each cell
    #*  Add one additional observation in the cell:
    additional_obs = sub.shape[0] + 1
    sub.loc[additional_obs] = [g, 0, 0, 0, 0, 0, 0]
    sub["LS_g"] = 0

    #*  loop over 4 corners of the rank-rank space (0,0), (0,1), (1,0), (1,1)
    for i in range(4):
        sub.loc[additional_obs-1, "parent_rank"] = i // 2
        sub.loc[additional_obs, "kid_rank"] = i % 2
        X = np.column_stack((sub["parent_rank"], np.ones_like(sub["kid_rank"])))
        y = sub["kid_rank"]
        b_d = np.linalg.inv(X.T @ X) @ X.T @ y
        sub.loc[additional_obs-1, "theta_g_d"] = b_d[0] * 0.25 + b_d[1]

        #*  compute LS as the max absolute difference between theta_g_d and theta_g
        LS_g = np.max(np.abs(data.loc[data["cell"] == g, "theta_g_d"] - data.loc[data["cell"] == g, "theta_g"]), sub)
        sub.loc[sub["cell"] == g, "LS_g"] = LS_g
    
    sub.drop(sub.index[-1], inplace=True)
    data.loc[data["cell"] == g, "N_g"] = sub.shape[0]
    data.loc[data["cell"] == g, "chi"] = np.max(sub["N_g"] * sub["LS_g"])

data.head()
# %%
