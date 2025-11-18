# importing libraries 
import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------
# 1. READ INPUT DATA
# ---------------------------------------------------------
df1 = pd.read_excel('Input.xlsx', sheet_name='Input')
df2 = pd.read_excel('Input.xlsx', sheet_name='Cap_table', index_col=0)

# ---------------------------------------------------------
# 2. COMPUTE EXIT AMOUNT FOR EACH SHARE CLASS
# ---------------------------------------------------------
# Exit Amount (converted to Million) = Conversion Price × #Shares 
df2['Exit Amount (in Million)'] = df2['Conversion Price Per Share (INR)'] * df2['# Shares'] / 10**6

# ---------------------------------------------------------
# 3. PREPARE UNIQUE CONVERSION PRICES
# ---------------------------------------------------------
conv_price = np.unique(df2['Conversion Price Per Share (INR)'])
conv_price.sort()

# Difference between adjacent conversion prices
diff_conv_price = np.diff(conv_price)

# ---------------------------------------------------------
# 4. FIND TOTAL SHARES AT EACH CONVERSION PRICE
# ---------------------------------------------------------
n_shares = []
for i in conv_price:
    n = df2[df2['Conversion Price Per Share (INR)'] == i]['# Shares'].sum()
    n_shares.append(n)

# ---------------------------------------------------------
# 5. CUMULATIVE SHARE COUNT FOR BREAKPOINTS
# ---------------------------------------------------------
sum_shares = 0.0
sum_share_arr = []

for i in n_shares:
    sum_shares += i
    sum_share_arr.append(sum_shares)

# ---------------------------------------------------------
# 6. LIQUIDATION PREFERENCE BREAKPOINTS
# ---------------------------------------------------------
n_lp = np.unique(df2['Liquidation Preference'])
n_lp.sort()

# Remove the lowest value (usually 0x or common)
n_lp_new = np.delete(n_lp, 0)

# Start breakpoints list with a tiny number to avoid log(0)
breakpoints = [1e-17]

# Add liquidation preference breakpoints
for i in n_lp_new:
    breakpoints.append(
        df2[df2['Liquidation Preference'] == i]['Exit Amount (in Million)'].sum()
    )

# ---------------------------------------------------------
# 7. CONVERSION PRICE BREAKPOINTS
# ---------------------------------------------------------
for j in range(len(diff_conv_price)):
    # Tranche width = cumulative shares × price difference
    d = sum_share_arr[j] * diff_conv_price[j] / 10**6
    breakpoints.append(d)

# ---------------------------------------------------------
# 8. CUMULATIVE BREAKPOINT SUM
# ---------------------------------------------------------
breakpoint_sum = 0
breakpoint_cum = []

for i in breakpoints:
    breakpoint_sum += i
    breakpoint_cum.append(breakpoint_sum)

# ---------------------------------------------------------
# 9. BLACK–SCHOLES CALL OPTION
# ---------------------------------------------------------
def black_scholes(S, X, T, r, sigma):
    """
    Standard Black-Scholes Call Option Formula.
    S = current stock price
    X = strike price
    T = time to maturity (years)
    r = risk-free rate
    sigma = volatility
    """
    d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)

# ---------------------------------------------------------
# 10. OPTION VALUES AT BREAKPOINTS
# ---------------------------------------------------------
opt_val = []
for i in breakpoint_cum:
    opt_val.append(
        black_scholes(
            df1.iloc[0, 1], # Stock price
            i,              # Strike
            df1.iloc[3, 1], # Time to maturity
            df1.iloc[1, 1], # Risk-free rate
            df1.iloc[2, 1]  # Volatility
        )
    )

# ---------------------------------------------------------
# 11. COMPUTE TRANCHE VALUES FROM OPTION DIFFERENCES
# ---------------------------------------------------------
# Each tranche value = -(difference in option values)
tranche_val = np.array(np.diff(opt_val) * -1)
tranche_val = np.append(tranche_val, opt_val[-1])

# ---------------------------------------------------------
# 12. DISTRIBUTION MATRIX BY TRANCHE
# ---------------------------------------------------------
df_dist_val = pd.DataFrame(index=df2.index)
df_dist = pd.DataFrame(index=df2.index)

# 12A. First allocate based on liquidation preferences
for i in n_lp_new:
    df_dist_val['Tranche ' + i.astype(str)] = df2[df2['Liquidation Preference'] == i]['Exit Amount (in Million)']
    df_dist_val.fillna(0, inplace=True)
    df_dist['Tranche ' + i.astype(str)] = df2[df2['Liquidation Preference'] == i]['Exit Amount (in Million)']

# 12B. Then allocate conversion price-based tranches
for i in range(len(conv_price)):
    n = i + np.max(n_lp_new) + 1  # tranche numbering continues
    df_dist_val['Tranche ' + n.astype(str)] = df2[df2['Conversion Price Per Share (INR)'] == conv_price[i]]['# Shares']

    df_dist_val.fillna(0, inplace=True)

    # cumulative for full conversion tranche
    df_dist['Tranche ' + n.astype(str)] = df_dist_val.iloc[:, (np.max(n_lp_new)): n].sum(axis=1)

    df_dist.fillna(0, inplace=True)

# ---------------------------------------------------------
# 13. NORMALIZE EACH ROW INTO PERCENTAGE ALLOCATION
# ---------------------------------------------------------
norm_df_dist = df_dist.div(df_dist.sum())

# Convert to numpy array for matrix multiplication
arr = norm_df_dist.to_numpy()

# ---------------------------------------------------------
# 14. VALUE OF SHARES = Weighted sum of tranche values
# ---------------------------------------------------------
val_shares = np.dot(arr, opt_val)

df_val = pd.DataFrame(index=norm_df_dist.index)
df_val['Total Value'] = val_shares

df_val['# Shares'] = df2['# Shares']

# Value per share, scaling back from million
df_val['Value Per Share'] = df_val['Total Value'] / df2['# Shares'] * 10**6

#recording breakpoints in a dataframe
df_breakpoints = pd.DataFrame(index = norm_df_dist.columns)

df_breakpoints['Beginning Tranche Breakpoint'] = breakpoint_cum
df_breakpoints['Ending Tranche Breakpoint'] = df_breakpoints['Beginning Tranche Breakpoint'].shift(-1)
df_breakpoints = df_breakpoints.replace(to_replace = df_breakpoints.iloc[-1,-1], value = "Infinity")

# saving data to excel file
excel_file_path = 'Output.xlsx'
with pd.ExcelWriter(excel_file_path) as writer:
    norm_df_dist.transpose().to_excel(writer, sheet_name='Distribution', index = True)
    df_val.transpose().to_excel(writer, sheet_name='Value', index=True)
    df_breakpoints.to_excel(writer, sheet_name='Breakpoints', index=True)
