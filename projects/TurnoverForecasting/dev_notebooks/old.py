# %%
import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math 
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import boxcox

# %%
# Load the dataset
file_path = "../data/nberces5818v1_n2012.csv"
df = pd.read_csv(file_path)

# Display basic info about the dataset
df.info()

# %%
# Check for missing values
missing_values = df.isnull().sum()

print(missing_values)

# %%
# Summary statistics
summary_stats = df.describe()

print(summary_stats)

# %%
# Visualizing missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values in Dataset")
plt.show()

# %%
# # Define Manufacturing Industry Groups (Example: Modify as Needed)
# heavy_mfg = [3311, 3312, 3321, 3322]  # Metal Production, Heavy Equipment
# light_mfg = [3231, 3251, 3261]  # Printing, Chemicals, Plastics
# electronics_mfg = [3341, 3351]  # Semiconductor, Electrical Equipment
# auto_mfg = [3361, 3362]  # Automotive Industry

# # Create a column for manufacturing sector groups
# def categorize_manufacturing(naics):
#     if naics in heavy_mfg:
#         return "Heavy Manufacturing"
#     elif naics in light_mfg:
#         return "Light Manufacturing"
#     elif naics in electronics_mfg:
#         return "Electronics Manufacturing"
#     elif naics in auto_mfg:
#         return "Automotive Manufacturing"
#     else:
#         return "Other Manufacturing"

# df['mfg_sector'] = df['naics'].apply(categorize_manufacturing)

# %%
# # Data Cleaning
# # Continuous Trends - Use Linear Interpolation
# df[['emp', 'pay', 'prode', 'prodh', 'prodw', 'vship', 'matcost', 'vadd', 'energy', 'invent']] = df[['emp', 'pay', 'prode', 'prodh', 'prodw', 'vship', 'matcost', 'vadd', 'energy','invent']].interpolate(method='linear')

# # Moderate Missingness (1-4%) - Use Rolling Median or Global Median
# moderate_missing_cols = ['invest', 'cap', 'equip', 'plant', 'piinv', 'tfp5', 'tfp4']
# df[moderate_missing_cols] = df[moderate_missing_cols].rolling(5, min_periods=1).median()
# df[moderate_missing_cols] = df[moderate_missing_cols].fillna(df[moderate_missing_cols].median())

# # Macroeconomic Indicators - Use Spline Interpolation
# macro_cols = ['piship', 'pimat', 'piinv', 'pien']
# df[macro_cols] = df[macro_cols].interpolate(method='spline', order=3)

# # Define features and targets for predictive imputation
# features = ['emp', 'pay', 'prode', 'invest', 'cap', 'matcost', 'vadd']
# targets = ['dtfp5', 'dtfp4']

# # Process each NAICS group separately
# unique_naics = df['naics'].unique()

# for naics_code in unique_naics:
#     subset_df = df[df['naics'] == naics_code]  # Filter data for current NAICS code
#     train_data = subset_df.dropna(subset=targets + features)  # Ensure enough data exists

#     if not train_data.empty:
#         for target in targets:
#             rf = RandomForestRegressor(n_estimators=100, random_state=42)
#             rf.fit(train_data[features], train_data[target])
            
#             # Identify rows with missing values but complete features
#             missing_rows = subset_df[subset_df[target].isna() & subset_df[features].notnull().all(axis=1)]
            
#             if not missing_rows.empty:
#                 df.loc[missing_rows.index, target] = rf.predict(missing_rows[features])

#     # Apply KNN Imputation only to remaining missing values within the NAICS group
#     knn_imputer = KNNImputer(n_neighbors=5)
#     df.loc[df['naics'] == naics_code, targets] = knn_imputer.fit_transform(subset_df[targets])

# # Verify missing values are fixed
# print(df[targets].isnull().sum())


# %%
# df["naics"] = df["naics"].astype(str)
# df = df[df["naics"].str.startswith(("31", "32", "33"))].copy()

# #df_naics_mfg = df[df["naics"].str.startswith(("31", "32", "33"))].copy()
# #.astype(str)
# # Step 2: Define Manufacturing Subcategories
# manufacturing_subcategories = {
#     "311": "Food Manufacturing",
#     "312": "Beverage & Tobacco Product Manufacturing",
#     "313": "Textile Mills",
#     "314": "Textile Product Mills",
#     "315": "Apparel Manufacturing",
#     "316": "Leather & Allied Product Manufacturing",
#     "321": "Wood Product Manufacturing",
#     "322": "Paper Manufacturing",
#     "323": "Printing & Related Support Activities",
#     "324": "Petroleum & Coal Products Manufacturing",
#     "325": "Chemical Manufacturing",
#     "326": "Plastics & Rubber Products Manufacturing",
#     "327": "Nonmetallic Mineral Product Manufacturing",
#     "331": "Primary Metal Manufacturing",
#     "332": "Fabricated Metal Product Manufacturing",
#     "333": "Machinery Manufacturing",
#     "334": "Computer & Electronic Product Manufacturing",
#     "335": "Electrical Equipment, Appliance & Component Manufacturing",
#     "336": "Transportation Equipment Manufacturing",
#     "337": "Furniture & Related Product Manufacturing",
#     "339": "Miscellaneous Manufacturing"
# }

# # Step 3: Assign Subcategories Based on NAICS Prefix
# df["Subcategory"] = df["naics"].str[:3].map(manufacturing_subcategories)


# %%
# Step 1: Convert NAICS Codes to String and Filter for Manufacturing (31-33)
df["naics"] = df["naics"].astype(str)
df = df[df["naics"].str.startswith(("31", "32", "33"))].copy()

# Step 2: Define Core Manufacturing Subcategories (Excluding 323 & 324)
core_manufacturing_subcategories = {
    "311": "Food Manufacturing",
    "312": "Beverage & Tobacco Product Manufacturing",
    "313": "Textile Mills",
    "314": "Textile Product Mills",
    "315": "Apparel Manufacturing",
    "316": "Leather & Allied Product Manufacturing",
    "321": "Wood Product Manufacturing",
    "322": "Paper Manufacturing",
    "325": "Chemical Manufacturing",
    "326": "Plastics & Rubber Products Manufacturing",
    "327": "Nonmetallic Mineral Product Manufacturing",
    "331": "Primary Metal Manufacturing",
    "332": "Fabricated Metal Product Manufacturing",
    "333": "Machinery Manufacturing",
    "334": "Computer & Electronic Product Manufacturing",
    "335": "Electrical Equipment, Appliance & Component Manufacturing",
    "336": "Transportation Equipment Manufacturing",
    "337": "Furniture & Related Product Manufacturing",
    "339": "Miscellaneous Manufacturing"
}

# Step 3: Assign Subcategories Based on NAICS Prefix
df["Subcategory"] = df["naics"].str[:3].map(core_manufacturing_subcategories)

# Step 4: Remove Non-Core Manufacturing Categories (Rows where Subcategory is NaN)
df = df.dropna(subset=["Subcategory"])

# Reset index for clean output
df.reset_index(drop=True, inplace=True)

# %%
df


# %%
# Define feature and target variables
features = ['emp', 'pay', 'prode', 'invest', 'cap', 'matcost', 'vadd']
targets = ['dtfp5', 'dtfp4']

# Unique NAICS Groups
unique_naics = df['naics'].unique()

# Iterate Over Each NAICS Group
for naics_code in unique_naics:
    subset_df = df[df['naics'] == naics_code].copy()

    # 1️⃣ Handle Moderate Missingness with Rolling Median
    moderate_missing_cols = ['invest', 'cap', 'equip', 'plant', 'piinv', 'tfp5', 'tfp4']
    subset_df[moderate_missing_cols] = subset_df[moderate_missing_cols].rolling(5, min_periods=1).median()
    subset_df[moderate_missing_cols] = subset_df[moderate_missing_cols].fillna(subset_df[moderate_missing_cols].median())

    # 2️⃣ Apply Industry-Specific Interpolation for Continuous Trends
    continuous_cols = ['emp', 'pay', 'prode', 'prodh', 'prodw', 'vship', 'matcost', 'vadd', 'energy', 'invent']
    subset_df[continuous_cols] = subset_df[continuous_cols].interpolate(method='spline', order=3)

    # 3️⃣ Apply Macroeconomic Adjustments with Spline Interpolation
    macro_cols = ['piship', 'pimat', 'piinv', 'pien']
    subset_df[macro_cols] = subset_df[macro_cols].interpolate(method='spline', order=3)

    # 4️⃣ Predict Missing Values using RandomForest
    train_data = subset_df.dropna(subset=targets + features)
    if not train_data.empty:
        for target in targets:
            if subset_df[target].isna().sum() > 0:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(train_data[features], train_data[target])

                # Predict only for missing values
                missing_rows = subset_df[subset_df[target].isna() & subset_df[features].notnull().all(axis=1)]
                if not missing_rows.empty:
                    df.loc[missing_rows.index, target] = rf.predict(missing_rows[features])

    # 5️⃣ Apply KNN Imputation for Remaining Missing Values
    knn_imputer = KNNImputer(n_neighbors=5)
    df.loc[df['naics'] == naics_code, targets] = knn_imputer.fit_transform(subset_df[targets])

# Verify missing values
print("Final Missing Values:\n", df.isnull().sum())


# %%
df.drop(columns=['naics'], inplace=True)

# %%
# Visualizing missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values in Dataset")
plt.show()

# %%
# Correlation Matrix
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix of the Dataset")
plt.show()

# %%
def plot_feature_distributions(df, max_cols=3):
    """
    Plots histograms for all columns in the dataframe.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing features to plot.
        max_cols (int): Maximum number of columns per row in the subplot grid.
    """
    num_features = len(df.columns)
    num_rows = math.ceil(num_features / max_cols)  # Calculate required rows

    plt.figure(figsize=(max_cols * 5, num_rows * 4))  # Adjust figure size dynamically

    for i, feature in enumerate(df.columns, 1):
        plt.subplot(num_rows, max_cols, i)
        sns.histplot(df[feature].dropna(), bins=50, kde=True)
        plt.title(f"Distribution of {feature}")

    plt.tight_layout()
    plt.show()
plot_feature_distributions(df, max_cols=3)

# %%
df.skew()

# %%
def preprocess_nans(df, strategy='median'):
    """
    Handles NaN values before transformations.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - strategy (str): Method to handle NaNs ('mean', 'median', 'zero').

    Returns:
    - df_filled (pd.DataFrame): Dataframe with NaNs handled.
    - nan_replacements (dict): Dictionary storing replacements for restoration.
    """
    df_filled = df.copy()
    nan_replacements = {}  # Store original NaN locations

    for col in df.columns:
        if df[col].isna().sum() > 0:
            fill_value = df[col].median() if strategy == 'median' else df[col].mean()
            df_filled[col].fillna(fill_value, inplace=True)
            nan_replacements[col] = fill_value  # Store for later restoration

    return df_filled, nan_replacements

def transform_skewed_features(df, exclude_columns=['naics', 'year', 'Subcategory']):
    """
    Transforms skewed features while excluding specified columns.
    Handles NaNs before transformation.

    Returns:
    - Transformed dataframe.
    - Dictionary of applied transformations.
    - Dictionary storing NaN replacements.
    """
    df_cleaned, nan_replacements = preprocess_nans(df)  # Handle NaNs first
    df_transformed = df_cleaned.copy()
    transformations = {}  # Store transformations

    skewness = df_cleaned.skew()
    columns_to_transform = [col for col in df_cleaned.columns if col not in exclude_columns]

    for col in columns_to_transform:
        min_val = df_cleaned[col].min()

        if skewness[col] > 3 or skewness[col] < -3:  # EXTREME skewness
            shift = abs(min_val) + 1 if min_val <= 0 else 0  # Ensure positivity
            df_transformed[col] = np.log1p(df_transformed[col] + shift)
            transformations[col] = ('log1p_shift', shift)
        
        elif 1.5 < abs(skewness[col]) <= 3:  # MODERATE skewness
            df_transformed[col] = np.cbrt(df_transformed[col])
            transformations[col] = ('cbrt', None)

        elif 0.5 < abs(skewness[col]) <= 1.5:  # MILD skewness
            shift = abs(min_val) if min_val < 0 else 0
            df_transformed[col] = np.sqrt(df_transformed[col] + shift)
            transformations[col] = ('sqrt_shift', shift)

    return df_transformed, transformations, nan_replacements

def restore_original_values(df_transformed, transformations, nan_replacements):
    """
    Restores transformed values to their original scale and reintroduces NaNs.

    Parameters:
    - df_transformed (pd.DataFrame): Transformed dataframe.
    - transformations (dict): Mapping of column names to transformations.
    - nan_replacements (dict): Original NaN replacements.

    Returns:
    - Restored dataframe.
    """
    df_original = df_transformed.copy()
    
    for col, transform in transformations.items():
        if transform[0] == 'log1p_shift':
            df_original[col] = np.expm1(df_original[col]) - transform[1]  # Undo shift
        elif transform[0] == 'cbrt':
            df_original[col] = df_original[col] ** 3  # Inverse of cube root
        elif transform[0] == 'sqrt_shift':
            df_original[col] = (df_original[col] ** 2) - transform[1]  # Undo shift

    for col, fill_value in nan_replacements.items():
        df_original[col] = df_original[col].replace(fill_value, np.nan)  # Restore NaNs

    return df_original


# %%


# %%


# %%
# Step 1: Transform Data (including handling NaNs)
df_transformed, transformations, nan_replacements = transform_skewed_features(df)

print(nan_replacements)
# Step 2: Restore to Original Values
df_original = restore_original_values(df_transformed, transformations, nan_replacements)



# %%
# Ensure only numeric columns are compared
numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
numeric_df_original = df_original.select_dtypes(include=[np.number]).fillna(0)

# Compare only numeric values
print(np.allclose(numeric_df, numeric_df_original, atol=1e-6))  # Should return True

# %%
plot_feature_distributions(df_transformed, max_cols=3)

# %%
df_transformed.skew()

# %%
# Define expanded features for scatter plots
relevant_features = ["emp", "pay", "prode", "invest", "cap", "equip", "plant", "matcost", "vadd", "energy", "invent", "piship", "pimat", "piinv", "pien"]

# Set up scatter plot grid
plt.figure(figsize=(15, 15))
for i, feature in enumerate(relevant_features, 1):
    plt.subplot(5, 3, i)  # 5 rows, 3 columns of subplots
    sns.scatterplot(x=df_transformed[feature], y=df_transformed["vship"])
    plt.title(f"Turnover (vship) vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("Turnover (vship)")

plt.tight_layout()
plt.show()

# %%
df.to_csv("../data/processed_data.csv", index=False)    

# %%
df.Subcategory.value_counts()

# %%



