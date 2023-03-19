#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm, laplace
import matplotlib.pyplot as plt


# Load data
data = pd.read_stata("private_data_by_cells.dta")

# Define omega distribution
omega_dist = "normal"

# Initialize variables
data["theta_g"] = np.nan
data["SE_theta_g"] = np.nan
data["theta_g_d"] = np.nan
data["LS_g"] = np.nan

# Loop over cells
cells = data["cell"].unique()
for g in cells:
    # Estimate theta_g
    sub_data = data[data["cell"] == g]
    X = sm.add_constant(sub_data["parent_rank"])
    y = sub_data["kid_rank"]
    model = sm.OLS(y, X).fit()
    data.loc[data["cell"] == g, "theta_g"] = model.params[1]*0.25 + model.params[0]
    # Calculate SE_theta_g
    data.loc[data["cell"] == g, "SE_theta_g"] = model.bse[1]
    # Calculate LS_g
    n = sub_data.shape[0]
    additional_obs = n + 1
    noise_infused_theta_g_d = []
    for i in range(4):
        parent_rank = i // 2
        kid_rank = i % 2
        new_obs = pd.DataFrame({"kid_rank": [kid_rank], "parent_rank": [parent_rank], "cell": [g]})
        sub_data_new = sub_data.append(new_obs, ignore_index=True)
        X_new = sm.add_constant(sub_data_new["parent_rank"])
        y_new = sub_data_new["kid_rank"]
        model_new = sm.OLS(y_new, X_new).fit()
        theta_g_d = model_new.params[1]*0.25 + model_new.params[0]
        noise_infused_theta_g_d.append(theta_g_d)
    LS_g = np.max([np.abs(theta_g_d - data.loc[data["cell"] == g, "theta_g"].values[0]) for theta_g_d in noise_infused_theta_g_d])
    data.loc[data["cell"] == g, "LS_g"] = LS_g
    
# Calculate chi and collapse data
data["N_g"] = data.groupby("cell")["cell"].transform("count")
data["N_g_LS_g"] = data["N_g"] * data["LS_g"]
data_grouped = data.groupby("cell").agg({"theta_g": "first", "SE_theta_g": "first", "N_g": "first", "N_g_LS_g": "first"})
data_grouped["chi"] = data_grouped["N_g_LS_g"].max()
data_grouped = data_grouped.drop(columns=["N_g_LS_g"])

#%%
data.head()
#%%

# Loop over epsilon
np.random.seed(419)
epsilon_range = np.arange(1, 11)
draws = 500

MSE_eps = []

for epsilon in epsilon_range:
    for d in range(1, draws+1):
        if omega_dist == "normal":
            omega = np.random.normal(0, 1)
        elif omega_dist == "laplace":
            omega = np.random.laplace(0, 1/np.sqrt(2))
        else:
            raise ValueError("Invalid omega distribution.")
        
        data["noise_infused_theta_g"] = (
            data["theta_g"] + np.sqrt(2)*(data_grouped["chi"] / (epsilon * data["N_g"])) * omega
        )
        data["diff_true_noise"] = (data["noise_infused_theta_g"] - data["theta_g"]) ** 2
        
        MSE_eps.append({"epsilon": epsilon, "MSE": np.mean(data["diff_true_noise"].to_numpy())})
    
MSE_eps_df = pd.DataFrame(MSE_eps)

MSE_eps_mean = (
    MSE_eps_df.groupby("epsilon")
    .agg({"MSE": np.mean})
    .reset_index()
)

plt.plot(MSE_eps_mean["epsilon"], MSE_eps_mean["MSE"])
plt.xlabel("Epsilon")
plt.ylabel("Mean Squared Error (Average)")
plt.show()

epsilon = 4

np.random.seed(5711)

if omega_dist == "normal":
    omega = np.random.normal(0, 1)
elif omega_dist == "laplace":
    omega = np.random.laplace(0, 1/np.sqrt(2))
else:
    raise ValueError("Invalid omega distribution.")

noise_infused_theta_g = (
    data["theta_g"] + np.sqrt(2)*(data_grouped["chi"] / (epsilon * data["N_g"])) * omega
)
SE_noise_infused_theta_g = np.sqrt(data["SE_theta_g"]**2 + 2*((data_grouped["chi"] / (epsilon * data["N_g"]))**2))
noise_infused_N_g = data["N_g"] + np.sqrt(2)*(omega / epsilon)
SD_noise_g = np.sqrt(2) * (data_grouped["chi"] / (epsilon * data["N_g"]))

Var_noise_g = SD_noise_g ** 2 
noise_var = np.mean(Var_noise_g)
total_variance = np.var([data["theta_g"], data["N_g"]])

share_noise_variance = noise_var / total_variance
print(f"Share of noise variance: {share_noise_variance:.3f}")

estimates_df = pd.DataFrame({
    "theta_g": data["theta_g"],
    "SE_theta_g": data["SE_theta_g"],
    "N_g": data["N_g"],
    "noise_infused_theta_g": noise_infused_theta_g,
    "SE_noise_infused_theta_g": SE_noise_infused_theta_g,
    "noise_infused_N_g": noise_infused_N_g
})

estimates_df.to_excel("example_cell_public_estimates_p25.xlsx", index=False)
