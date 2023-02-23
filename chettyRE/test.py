import pandas as pd
import numpy as np
from scipy.stats import norm, laplace

# 1. Load data
data = pd.read_stata("private_data_by_cells.dta")

# 2. Set distribution of omega (see Step 4 of algorithm) from laplace or normal
omega_dist = "normal"

# 3. Calculate Local Sensitivity
# Generate variables to fill in with key parameters in each cell g:
data["theta_g"] = np.nan  # estimate of interest
data["SE_theta_g"] = np.nan  # SE of estimate of interest
data["theta_g_d"] = np.nan  # estimate obtained when adding one observation
data["LS_g"] = np.nan  # local sensitivity

cells = data["cell"].unique()  # save list of cells

# Compute true statistic, SE, and local sensitivity for each cell
for g in cells:
    # Compute true statistic
    reg = data.query(f"cell == {g}").loc[:, ["kid_rank", "parent_rank"]].values
    X = np.column_stack((reg[:, 1], np.ones_like(reg[:, 0])))
    y = reg[:, 0]
    b = np.linalg.inv(X.T @ X) @ X.T @ y

    # Save statistic and SE
    data.loc[data["cell"] == g, "theta_g"] = b[0] * 0.25 + b[1]
    data.loc[data["cell"] == g, "SE_theta_g"] = np.sqrt(np.sum((y - X @ b) ** 2) / (reg.shape[0] - 2))

    # Compute Local Sensitivity (LS) for each cell
    # Add one additional observation in the cell:
    additional_obs = reg.shape[0] + 1
    data.loc[additional_obs] = [g, 0, 0, 0, 0]

    # loop over 4 corners of the rank-rank space (0,0), (0,1), (1,0), (1,1)
    for i in range(4):
        data.loc[additional_obs, "parent_rank"] = i // 2
        data.loc[additional_obs, "kid_rank"] = i % 2
        reg = data.query(f"cell == {g}").loc[:, ["kid_rank", "parent_rank"]].values
        X = np.column_stack((reg[:, 1], np.ones_like(reg[:, 0])))
        y = reg[:, 0]
        b_d = np.linalg.inv(X.T @ X) @ X.T @ y
        data.loc[data["cell"] == g, "theta_g_d"] = b_d[0] * 0.25 + b_d[1]

        # compute LS as the max absolute difference between theta_g_d and theta_g
        ls = np.max(np.abs(data.loc[data["cell"] == g, "theta_g_d"] - data.loc[data["cell"] == g, "theta_g"]))
        data.loc[data["cell"] == g, "LS_g"] = ls

    data.drop(additional_obs, inplace=True)

# 4. Compute Maximum Observed Sensitivity (chi)
# compute number of observations per cell
n_g = data.groupby("cell").size().reset_index(name="N_g")
data = pd.merge(data, n_g, on="cell")

# compute N_g * LS_g
data["N_g_LS_g"] = data["N_g"] * data["LS_g"]

# Find its max
chi = data["N_g_LS_g"].max()
