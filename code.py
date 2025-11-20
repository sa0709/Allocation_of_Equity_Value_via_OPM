# =====================================================================
# IMPORTING LIBRARIES
# =====================================================================
import numpy as np
import pandas as pd
from scipy.stats import norm

# =====================================================================
# 1. READ INPUT DATA
# =====================================================================
# df1 contains parameters like S, r, sigma, T etc.
# df2 contains details of the cap table (shares, liquidation prefs, etc.)
df1 = pd.read_excel('Input.xlsx', sheet_name='Input')
df2 = pd.read_excel('Input.xlsx', sheet_name='Cap_table', index_col=0)

# =====================================================================
# 2. COMPUTE EXIT AMOUNT FOR EACH SHARE CLASS
# =====================================================================

# Conversion Price formula:
#   Issue Price × Liquidation Multiplier × (1 + Dividend Rate)^(years outstanding)
df2['Conversion Price Per Share (INR)'] = (
    df2['Issue Price Per Share (INR)']
    * df2['Liquidation Multiplier']
    * ((1 + df2['Dividend Rate']) ** (
        (df1.iloc[-1, -1] - df2['Investment Date']).dt.days / 365   # time from issue to exit
    ))
)

# Total Exit Amount for each class (in million INR)
df2['Exit Amount (in Million)'] = df2['Conversion Price Per Share (INR)'] * df2['# Shares'] / 10**6

# =====================================================================
# 3. PREPARE LIST OF UNIQUE CONVERSION PRICES
# =====================================================================

# Unique conversion prices sorted in ascending order
conv_price = np.unique(df2['Conversion Price Per Share (INR)'])
conv_price.sort()

# Difference between subsequent conversion prices (used for tranches)
diff_conv_price = np.diff(conv_price)

# =====================================================================
# 4. FIND TOTAL SHARES AT EACH CONVERSION PRICE
# =====================================================================

# For each conversion price, count total shares issued at that price
n_shares = []
for i in conv_price:
    n = df2[df2['Conversion Price Per Share (INR)'] == i]['# Shares'].sum()
    n_shares.append(n)

# =====================================================================
# 5. CUMULATIVE SHARE COUNT (NEEDED TO CREATE TRANCHE WIDTHS)
# =====================================================================

sum_shares = 0.0
sum_share_arr = []

# Running cumulative sum of shares
for i in n_shares:
    sum_shares += i
    sum_share_arr.append(sum_shares)

# =====================================================================
# 6. LIQUIDATION PREFERENCE BREAKPOINTS
# =====================================================================

# Unique liquidation preference multipliers
n_lp = np.unique(df2['Liquidation Preference'])
n_lp.sort()

# Remove the smallest value (often 0 or 1× for commons)
n_lp_new = np.delete(n_lp, 0)

# Start breakpoint list with tiny value to avoid log(0) in Black-Scholes
breakpoints = [1e-17]

# Add breakpoints for liquidation preferences:
# Each LP class breakpoint = total exit amount for that LP tier
for i in n_lp_new:
    breakpoints.append(
        df2[df2['Liquidation Preference'] == i]['Exit Amount (in Million)'].sum()
    )

# =====================================================================
# 7. CONVERSION PRICE BREAKPOINTS
# =====================================================================

# For each difference in conversion price:
#   Tranche width = cumulative shares × price diff
for j in range(len(diff_conv_price)):
    d = sum_share_arr[j] * diff_conv_price[j] / 10**6  # convert to millions
    breakpoints.append(d)

# =====================================================================
# 8. CUMULATIVE BREAKPOINT SUM (i.e., strike prices for options)
# =====================================================================

breakpoint_sum = 0
breakpoint_cum = []

# Convert tranche widths into cumulative breakpoints
for i in breakpoints:
    breakpoint_sum += i
    breakpoint_cum.append(breakpoint_sum)

# =====================================================================
# 9. BLACK–SCHOLES CALL OPTION FORMULA
# =====================================================================

def black_scholes(S, X, T, r, sigma):
    """
    Computes the Black-Scholes price of a Call option.
    S     = current enterprise value
    X     = strike (breakpoint)
    T     = time to maturity
    r     = risk-free rate
    sigma = volatility
    """
    d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)

# =====================================================================
# 10. OPTION VALUE FOR EACH BREAKPOINT (TRANCHE STRIKE)
# =====================================================================

opt_val = []
for i in breakpoint_cum:
    opt_val.append(
        black_scholes(
            df1.iloc[0, 1], # Enterprise Value
            i,              # Strike = breakpoint
            df1.iloc[3, 1], # Time to maturity
            df1.iloc[1, 1], # Risk-free rate
            df1.iloc[2, 1]  # Volatility
        )
    )

# =====================================================================
# 11. COMPUTE TRANCHE VALUES FROM OPTION DIFFERENCES
# =====================================================================

# Tranche value = -(difference of option values)
tranche_val = np.array(np.diff(opt_val) * -1)

# Last tranche receives the residual option value
tranche_val = np.append(tranche_val, opt_val[-1])

# =====================================================================
# 12. DISTRIBUTION MATRIX SETUP
# =====================================================================

df_dist_val = pd.DataFrame(index=df2.index)   # raw values allocation
df_dist = pd.DataFrame(index=df2.index)       # cumulative allocation

# ------------------ 12A. Allocate liquidation preference tranches -------------------
for i in n_lp_new:
    df_dist_val['Tranche ' + i.astype(str)] = (
        df2[df2['Liquidation Preference'] == i]['Exit Amount (in Million)']
    )
    df_dist_val.fillna(0, inplace=True)
    df_dist['Tranche ' + i.astype(str)] = (
        df2[df2['Liquidation Preference'] == i]['Exit Amount (in Million)']
    )

# ------------------ 12B. Allocate conversion price tranches -------------------------
for i in range(len(conv_price)):
    # Tranche index continues after LP tranches
    n = i + np.max(n_lp_new) + 1

    # Raw allocation: number of shares in this conversion price bucket
    df_dist_val['Tranche ' + n.astype(str)] = (
        df2[df2['Conversion Price Per Share (INR)'] == conv_price[i]]['# Shares']
    )
    df_dist_val.fillna(0, inplace=True)

    # Cumulative contribution for this conversion tranche
    df_dist['Tranche ' + n.astype(str)] = (
        df_dist_val.iloc[:, (np.max(n_lp_new)): n].sum(axis=1)
    )
    df_dist.fillna(0, inplace=True)

# =====================================================================
# 13. NORMALIZE ROWS INTO PERCENTAGE SHARE OF EACH TRANCHE
# =====================================================================

# Divide each row by its row sum → percent allocation matrix
norm_df_dist = df_dist.div(df_dist.sum())

# Convert to numpy for faster matrix multiplication
arr = norm_df_dist.to_numpy()

# =====================================================================
# 14. VALUE OF SHARES (ALLOCATION × TRANCHE VALUES)
# =====================================================================

# Dot product to allocate tranche values to share classes
val_shares = np.dot(arr, tranche_val)

df_val = pd.DataFrame(index=norm_df_dist.index)
df_val['Total Value'] = val_shares
df_val['# Shares'] = df2['# Shares']

# Convert back to INR from millions
df_val['Value Per Share'] = df_val['Total Value'] / df2['# Shares'] * 10**6

# =====================================================================
# RECORD BREAKPOINTS IN A DATAFRAME
# =====================================================================

df_breakpoints = pd.DataFrame(index=norm_df_dist.columns)
df_breakpoints['Beginning Tranche Breakpoint'] = breakpoint_cum
df_breakpoints['Ending Tranche Breakpoint'] = df_breakpoints['Beginning Tranche Breakpoint'].shift(-1)

# Last tranche ends at infinity
df_breakpoints = df_breakpoints.replace(to_replace=df_breakpoints.iloc[-1, -1], value="Infinity")

# =====================================================================
# CREATE VALUE MATRIX FOR OUTPUT
# =====================================================================

df_val_1 = norm_df_dist.transpose()

# Add tranche option values
df_val_1.insert(loc=0, column="Option Value", value=tranche_val)

# Multiply each row by its tranche option value
for col in df_val_1.columns:
    if col != 'Option Value':
        df_val_1[col] = df_val_1[col] * df_val_1['Option Value']

# Combine summary table with per-share values
df_val_1_comb = pd.concat([df_val_1, df_val.transpose()])

# =====================================================================
# SAVE TO EXCEL
# =====================================================================

excel_file_path = 'Output.xlsx'
with pd.ExcelWriter(excel_file_path) as writer:
    norm_df_dist.transpose().to_excel(writer, sheet_name='Distribution', index=True)
    df_val_1_comb.to_excel(writer, sheet_name='Value', index=True)
    df_breakpoints.to_excel(writer, sheet_name='Breakpoints', index=True)
