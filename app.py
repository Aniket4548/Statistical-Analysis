import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import itertools
import streamlit as st

from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn import feature_selection
from matplotlib import pyplot as plt
from streamlit_pandas_profiling import st_profile_report
from statsmodels.graphics.gofplots import qqplot

# Set up Streamlit app layout to full screen
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="Statistical Analysis",
    page_icon="📊"
)

st.markdown("<h1 style='text-align: center;'>Statistical Analysis of Auto-MPG Dataset</h1>", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("auto-mpg-Dataset.csv")

st.markdown("<h1 style='text-align: center;'>Pre-processing of the data</h1>", unsafe_allow_html=True)

# Generate profiling report
pr = df.profile_report(title="Auto-MPG EDA", correlations={"auto": {"calculate": False}}, dark_mode=True, explorative=True, lazy=False, progress_bar=False)

# Use the entire screen for the profiling report
st.write("##### Below is the dataset preview:")

# Center align the table
st.markdown("""
<style>
    table {
        width: 50%;
        margin-left: auto;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

st.table(df.head(10))

st.write("##### Below is the profiling report:")

# Creating a container that takes up the full width of the screen
with st.container():
    st_profile_report(pr)

# Inject custom CSS if needed
st.markdown(
    """
    <style>
    .css-1d391kg {  # Adjust the class as per the Streamlit's internal styling class for container
        padding: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write('#### Make two distinct lists for categorical and numerical columns')

def classify_columns(df):
    categorical_cols = []
    numerical_cols = []

    for col in df.columns:
        if df[col].dtype == 'object':
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

    return categorical_cols, numerical_cols

# Get categorical and numerical column names
categorical_cols, numerical_cols = classify_columns(df)

# Display column names divided into categorical and numerical categories
st.write("##### Categorical Columns:")
st.write(categorical_cols)

st.write("##### Numerical Columns:")
st.write(numerical_cols)

st.markdown('#### Update lists as cylinders and model_year are also categorical')
categorical_cols.extend(['cylinders', 'model_year'])
numerical_cols.remove('cylinders')
numerical_cols.remove('model_year')

st.write("##### Updated Categorical Columns:")
st.write(categorical_cols)

st.write("##### Updated Numerical Columns:")
st.write(numerical_cols)

st.write('#### Original shape:')
st.write(df.shape) # (392, 9)

# let's print these 6 `nan` containing rows 
st.markdown('#### NaNs in the dataset:')
st.table(df[df.isnull().any(axis=1)])

st.markdown('#### Drop the rows with NaNs')
df = df.dropna().reset_index(drop=True)

st.markdown('#### New shape:')
st.write(df.shape) # (392, 9)

st.markdown('#### Check for duplicates:')
st.write(f'Total duplicate rows: {df.duplicated().sum()}')

st.markdown('#### Remove extra spaces if any:')
for col in ['origin', 'name']:
    df[col] = df[col].str.strip()

st.markdown('#### Dividing mpg into three regions: Low, Medium, and High')
df['mpg_level'] = df['mpg'].apply(lambda x: 'low' if x < 17 else 'high' if x > 29 else 'medium')
categorical_cols.append('mpg_level')

st.markdown('#### Updated Categorical Variables:')
st.write(categorical_cols)

st.markdown('#### Group all variables together having the same type:')
df = pd.concat((df[categorical_cols], df[numerical_cols]), axis=1)
st.table(df.head())



st.markdown("<h1 style='text-align: center;'>Statistical Analysis</h1>", unsafe_allow_html=True)
ALPHA = 0.05
st.write("##### ALPHA =",ALPHA)

st.markdown('## Tests for independence between two categorical variables')

st.markdown('#### Contingency Table (aka frequency table)')

st.write('##### Contingency Table for origin and model year')
st.table(pd.crosstab(df.origin, df.model_year))

st.write('##### Contingency Table for origin and mpg_level')
observed_values = pd.crosstab(df.origin, df.mpg_level)
st.table(observed_values)


st.markdown('#### Chi-Square Test')
st.write('##### Use chi2_contingency function of scipy')

chi2, p, dof, expected_values = stats.chi2_contingency(observed_values)
st.write("##### chi2 =",chi2)
st.write("##### p =",p)
st.write("##### dof =",dof)
st.table(expected_values)

if p <= ALPHA:
    st.write(f'##### Rejected H0 under significance level {ALPHA} `origin` & `model_year` are dependent.')
else:
    st.write(f'##### Fail to reject H0 due to lack of evidence under significance level {ALPHA} `origin` & `model_year` are independent.')

st.markdown('#### Use chi2 to test dependency of all categorical attributes with mpg_level')

df_cat_label =  pd.concat([df.loc[:, ['origin', 'mpg_level']].apply(lambda x: LabelEncoder().fit_transform(x)),
                           df.loc[: , 'cylinders': 'model_year']], axis=1)

st.table(df_cat_label.head())


chi2_res = feature_selection.chi2(df_cat_label, df.mpg_level)

df_chi2 = pd.DataFrame({
    'attr1': 'mpg_level',
    'attr2': df_cat_label.columns,
    'chi2': chi2_res[0],
    'p': chi2_res[1],
    'alpha': ALPHA
})

df_chi2['H0'] = df_chi2.p.apply(lambda x: 'rejected' if x <= ALPHA else 'fail to reject')
df_chi2['relation'] = df_chi2.H0.apply(lambda x: 'dependent' if x=='rejected' else 'independent')

st.table(df_chi2)

st.markdown('## Statistical Tests for Numerical Attributes')

st.write("#### Numerical Columns",numerical_cols)

st.write('### Visual Noramlity Checks')
st.write('#### Check whether mpg and weight are log-normal or not')
# Create a figure for the histograms with log transformation
fig1 = pyplot.figure(1, (10, 4))

ax1 = pyplot.subplot(1, 2, 1)
sns.histplot(np.log2(df['mpg']), kde=True, stat="density")
ax1.set_title('Log2 of mpg Distribution')
pyplot.tight_layout()

ax2 = pyplot.subplot(1, 2, 2)
sns.histplot(np.log2(df['weight']), kde=True, stat="density")
ax2.set_title('Log2 of weight Distribution')
pyplot.tight_layout()

# Display the histogram figure in Streamlit
st.pyplot(fig1)


st.write(f'##### After applying log transformation we find that weight is not log-normal but mpg visually looks like log-normal')

# quantile-quantile plots on original data
# Create a figure for the Q-Q plots on original data
fig2 = pyplot.figure(figsize=(18, 8))

# Loop through the numerical columns and create Q-Q plots
for i, num in enumerate(numerical_cols):
    ax = pyplot.subplot(2, 3, i + 1)
    qqplot(df[num], line='s', ax=ax)
    ax.set_title(f'Q-Q Plot - {num}')
    pyplot.tight_layout()

# Display the Q-Q plots figure in Streamlit
st.pyplot(fig2)

st.write('##### Both histplot & qqplot of acceleration indicates that it is indeed close to gaussian')


st.write('### Statistical Normality Tests')

st.write('#### Hypothesis testing for the normality of numerical attributes using the shapiro wilk test')

def shapiro_wilk_test(df: pd.DataFrame, cols: list, alpha=0.05):
    # test the null hypothesis for columns given in `cols` of the dataframe `df` under significance level `alpha`.
    for col in cols:
        _,p = stats.shapiro(df[col])
        if p <= alpha:
            st.write(f'''\nRejected H0 under significance level {alpha}\n{col} doesn't seems to be normally distributed''')
        else:
            st.write(f'''##### \nFail to reject H0 due to lack of evidence under significance level {alpha}\n{col} seem to be normally distributed''')

     
_, p = stats.shapiro(df.acceleration)
st.write("##### p = ", p) # 0.03054318018257618

shapiro_wilk_test(df, numerical_cols)

st.write('#### Apply power transform to make the data more gaussian like')

from sklearn.preprocessing import PowerTransformer

df_tfnum = pd.DataFrame(PowerTransformer().fit_transform(df[numerical_cols]), columns=numerical_cols)
st.table(df_tfnum.head())


fig3 = pyplot.figure(figsize=(18, 8))

for i, num in enumerate(['mpg', 'displacement', 'horsepower', 'acceleration']):
    ax = pyplot.subplot(2, 2, i + 1)
    sns.histplot(df_tfnum[num], kde=True)
    ax.set_xlabel(f'Transformed {num}')
    ax.set_title(f'Transformed {num} Distribution')
    pyplot.tight_layout()

# Display the histogram figure for the transformed columns in Streamlit
st.pyplot(fig3)

st.write('##### acceleration is still gaussian, skewness is removed from mpg & weight making mpg gaussian-like. Also the distribution for displacement is improved now it\'s bimodal which respects the observation.')

fig4 = pyplot.figure(figsize=(18, 8))

for i, num in enumerate(['mpg', 'displacement', 'horsepower', 'acceleration']):
    ax = pyplot.subplot(2, 3, i + 1)
    qqplot(df_tfnum[num], line='s', ax=ax)
    ax.set_title(f'Q-Q Plot - Transformed {num}')
    pyplot.tight_layout()

# Display the Q-Q plots figure for the transformed columns in Streamlit
st.pyplot(fig4)

shapiro_wilk_test(df_tfnum, ['mpg', 'displacement', 'horsepower', 'acceleration'])
_, p = stats.shapiro(df_tfnum.acceleration)
st.write("##### p = ", p) 
st.write('##### So, acceleration is normally distributed both visually and statistically.')


st.write('## Tests for correlation between two continuous variables')


st.write('##### H_0: mpg and other attribute are not correlated, alpha=0.05')
for num in numerical_cols:
    if num == 'mpg':
        continue
    
    corr, p = stats.spearmanr(df.mpg, df[num])

    st.write(f'\n* `mpg` & `{num}`\n')
    st.write(f'corr: {round(corr, 4)} \t p: {p}')

    if p <= ALPHA:
        st.write(f'Rejected H0 under significance level {ALPHA}, mpg & {num} are correlated')
    else:
        st.write(f'''Fail to reject H0 due to lack of evidence under significance level {ALPHA}, 
              mpg & {num} are not correlated''')

st.write('##### Create a data-frame for the correlation b/w every pair')


def test_correlation(x1, x2, method='spearman', alpha=0.05):
    # this function returns correlation, p-value and H0 for `x1` & `x2`
    
    ALLOWED_METHODS = ['pearson', 'spearman', 'kendall']
    if method not in ALLOWED_METHODS:
        raise ValueError(f'allowed methods are {ALLOWED_METHODS}')
        
    if method=='pearson':
        corr, p = stats.pearsonr(x1,x2)
    elif method=='spearman':
        corr, p = stats.spearmanr(x1,x2)
    else:
        corr, p = stats.kendalltau(x1,x2)
    
    h0 = (
    'rejected'
    if p<=ALPHA else
    'fail to reject')
    
    return corr, p, h0
  
  
df_corr = pd.DataFrame(columns=['attr1', 'attr2', 'corr', 'p', 'H0'])

# Loop through combinations of numerical columns
for combo in itertools.combinations(numerical_cols, r=2):
    # Compute correlation between each pair of columns
    corr, p, h0 = test_correlation(df[combo[0]], df[combo[1]])
    # Append results to df_corr DataFrame
    df_corr = pd.concat([df_corr, pd.DataFrame({'attr1': [combo[0]], 'attr2': [combo[1]],
                                                 'corr': [round(corr, 5)], 'p': [p], 'H0': [h0]})], ignore_index=True)

# Print the resulting DataFrame
st.table(df_corr)


st.write('##### Correlation of pairs (mpg, acceleration), (displacement, acceleration) and (weight, acceleration) is moderate whereas remaining all pairs has very high correlation between them.')

st.write('## Parametric and Non-Parametric test for samples')


st.write('##### Test whether acceleration in japan and usa has the same mean')
shapiro_wilk_test(df[df.origin=='JPN'], ['acceleration'])
shapiro_wilk_test(df[df.origin=='USA'], ['acceleration'])

st.write(('##### So both are normally distributed so we can apply parametric test.'))

st.write('### Parametric Statistical Significance Test')


# because the variance is not same for the two distributions hence equal_var=False
_, p = stats.ttest_ind(df[df.origin=='JPN'].acceleration, df[df.origin=='USA'].acceleration, equal_var=False)

st.write('##### H_0: acceleration of japan and acceleration of usa has same sample mean, alpha=0.05')

if p <= ALPHA:
    st.write(f'Rejected H0 under {ALPHA*100}% significance, Different distributions.')
else:
    st.write(f'Fail to Reject H0 under {ALPHA*100}% significance, Same distributions.')


_, p = stats.f_oneway(df[df.origin=='JPN'].acceleration, df[df.origin=='USA'].acceleration, df[df.origin=='GER'].acceleration)

if p <= ALPHA:
    st.write(f'Rejected H0 under {ALPHA*100}% significance, Different distributions.')
else:
    st.write(f'Fail to Reject H0 under {ALPHA*100}% significance, Same distributions.')

shapiro_wilk_test(df[df.origin=='JPN'], ['horsepower'])
shapiro_wilk_test(df[df.origin=='GER'], ['horsepower'])
shapiro_wilk_test(df[df.origin=='USA'], ['horsepower'])

st.write('##### So all of them are not normally distributed so we will apply non-parametric test.')

st.write('### Non-Parametric Statistical Significance Test')

st.write('##### H_0: Sample distributions are equal for horsepower across region, alpha=0.05')
st.write('##### Test whether acceleration has same distribution for samples with mpg_level high & medium')
_, p = stats.mannwhitneyu(df[df.mpg_level=='high'].acceleration, df[df.mpg_level=='medium'].acceleration)

if p <= ALPHA:
    st.write(f'Rejected H0 under {ALPHA*100}% significance, Different distributions.')
else:
    st.write(f'Fail to Reject H0 under {ALPHA*100}% significance, Same distributions.')


st.write('##### Test for mpg distribution across the years')

acc_gb_year = df.groupby('model_year')['mpg']

acc_yr = []
for yr in df.model_year.unique():
    acc_yr.append(list(acc_gb_year.get_group(yr)))
    
_, p = stats.kruskal(*acc_yr)

if p <= ALPHA:
    st.write(f'Rejected H0 under {ALPHA*100}% significance, Different distributions.')
else:
    st.write(f'Fail to Reject H0 under {ALPHA*100}% significance, Same distributions.')


st.write('## Relation between Categorical and Continuous attributes')

result_f = feature_selection.f_classif(df.loc[:, 'mpg': 'acceleration'], df.cylinders)

anova_test_cat = pd.DataFrame({
    'cat-attr': 'cylinders',
    'cont-attr': df.loc[:, 'mpg': 'acceleration'].columns,
    'f': result_f[0],
    'p': result_f[1],
    'alpha': ALPHA
})

anova_test_cat['H0'] = anova_test_cat.p.apply(lambda x: 'rejected' if x <= ALPHA else 'fail to reject')
anova_test_cat['relation'] = anova_test_cat.H0.apply(lambda x: 'dependent' if x=='rejected' else 'independent')

st.table(anova_test_cat)

result_f = feature_selection.f_classif(df_cat_label[['origin', 'cylinders', 'model_year']], df.mpg)

anova_test_cat = pd.DataFrame({
    'cont-attr': 'mpg',
    'cat-attr': ['origin', 'cylinders', 'model_year'],
    'f': result_f[0],
    'p': result_f[1],
    'alpha': ALPHA
})

anova_test_cat['H0'] = anova_test_cat.p.apply(lambda x: 'rejected' if x <= ALPHA else 'fail to reject')
anova_test_cat['relation'] = anova_test_cat.H0.apply(lambda x: 'dependent' if x=='rejected' else 'independent')

st.table(anova_test_cat)

