#%% import
import math
import random
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# %% Load the dataset
data = pd.read_stata("private_data_by_cells.dta")
omega_dist = "normal"

data.info()
# %%    Calculate Local Sensitivity
data["theta_g"] = np.nan    #*  estimate of interest
data["SE_theta_g"] = np.nan #*  SE of estimate of interest
data["theta_g_d"] = np.nan  #*  estimate obtained when adding one observation
data["LS_g"] = np.nan       #*  local sensitivity

cells = np.unique(data["cell"])

#TODO: Compute true statistic, SE, and local sensitivity for each cell
for cell in cells:
    #todo: subset the data for current cell
    sub = data[data["cell"] == cell]

    #todo: run regression of kid_rank on parent_rank
    X = sm.add_constant(sub["parent_rank"])
    y = sub["kid_rank"]
    model = sm.OLS(y, X).fit()
    theta_g = model.params[0] + model.params[1] * 0.25
    SE_theta_g = model.bse[1]

    #todo: save statistic and SE
    data.loc[data["cell"] == cell, "theta_g"] = theta_g
    data.loc[data["cell"] == cell, "SE_theta_g"] = SE_theta_g

    #todo: add an additional observation
    n_obs = len(sub) + 1
    additional_obs = pd.DataFrame({"cell": [cell], "parent_rank": [0], "kid_rank": [0]}, index=[n_obs])
    sub = pd.concat([sub, additional_obs], ignore_index=False)
    sub["LS_g"] = 0
    #todo: loop over 4 corners of the rank-rank space (0,0), (0,1), (1,0), (1,1)
    for i in range(4):
        sub.loc[n_obs, "parent_rank"] = i // 2
        sub.loc[n_obs, "kid_rank"] = i % 2
        X = sm.add_constant(sub["parent_rank"])
        y = sub["kid_rank"]
        model = sm.OLS(y, X).fit()
        theta_g_d = model.params[0]
        data.loc[data["cell"] == cell, "theta_g_d"] = theta_g_d
        LS_g = np.max([np.abs(theta_g_d - data.loc[data["cell"] == cell, "theta_g"].values[0]), sub["LS_g"].values[0]])
        data.loc[data["cell"] == cell, "LS_g"] = LS_g

    data = data.loc[data.index != n_obs]

# %%    Compute Maximum Observed Sensitivity (chi)
data["N_g"] = data.groupby("cell")["cell"].transform('count')
data["N_g_LS_g"] = data["N_g"] * data["LS_g"]
chi = data.groupby("cell")["N_g_LS_g"].max().max()
# %%    Determine Privacy Parameter (epsilon)
#todo: collapse at the cell level
data_collapsed = data.groupby("cell")[[ "theta_g", "SE_theta_g", "N_g"]]

#todo: set seed
random.seed(5711)

#todo: compute MSE for multiple epsilons
for epsilon in range(1, 11):
    draws = 500

    #todo: generate 500 draws of noise from the distribution specified by "omega_dist"
    omega = np.random.normal(0, 1, draws)

    #todo: calculate the perturbed estimate of "theta_g" for each draw
    diff_true_noise = []
    for i in range(draws):
        noise_infused_theta_g = data["theta_g"] + (epsilon * chi * omega[i-1])
        diff_true_noise.append((noise_infused_theta_g - data["theta_g"]) ** 2)

    #todo: calculate the mean of the squared differences across all draws
    MSE_eps = np.mean(diff_true_noise)

    #todo: print the MSE for the current epsilon
    print(f"MSE_eps_{epsilon} = {MSE_eps}")

#todo: save the current dataset
data.to_csv("private_data_by_celss_processed.csv", index=False)
# %%
#todo: Collapse the dataset by cell
df_collapsed = data.groupby('cell').agg(
    N_g=('kid_rank', 'count'),
    theta_g=('theta_g', 'mean'),
    SE_theta_g=('SE_theta_g', 'mean'),
    LS_g=('LS_g', 'max')
).reset_index()

#todo: Reshape the dataset to long format
df_long = pd.melt(df_collapsed, id_vars=['cell', 'N_g', 'LS_g'], var_name='epsilon', value_name='MSE_eps')

#todo: Generate a line graph of "MSE_eps" against "epsilon"
sns.lineplot(data=df_long, x='epsilon', y='MSE_eps')
plt.title('Mean Squared Error vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Mean Squared Error')
plt.show()
# %%