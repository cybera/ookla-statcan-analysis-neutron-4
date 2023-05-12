# app speed model
import streamlit as st

import src.config
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt 

from PIL import Image

data_name = "BestEstimate_On_DissolvedSmallerCitiesHexes_Time"
data_dir = src.config.DATA_DIRECTORY / "processed" / "statistical_geometries" / "time"
data = pd.read_csv(data_dir / (data_name+".csv"), index_col=0)

st.markdown("# Exploratory Data Analysis")
st.sidebar.markdown("# Exploratory Data Analysis")
st.write("""
Our goal for conducting EDA was to identify new features
that could be use to improve upon the provided Linear Model.
Particularly, we were interested in seeing whether other features
including, time, network conditions, testing frequency among others
could be included in the regression analysis.
""")

st.markdown("## Distribution Analyses")
st.sidebar.markdown("## Distribution Analyses️")

# plot 1: distribution plot of speeds across Canada from 2019 to 2023

st.write("""
We began by investigating the distribution of internet speeds in Canada
""")

# Create the Streamlit app
st.subheader('Distribution of Average Internet Speeds across Canada')

# Create a selectbox for the speed type
speed_type = st.selectbox('Select a speed type', ['Download', 'Upload'])

# Select the appropriate column
if speed_type == 'Download':
    column = 'avg_d_kbps'
    bin_range = range(0, 300000, 3000)
    threshold = 50000
    xlabel = 'Download Speed (kbps)'
elif speed_type == 'Upload':
    column = 'avg_u_kbps'
    bin_range = range(0, 100000, 1000)
    threshold = 10000
    xlabel = 'Upload Speed (kbps)'

# Create the plot
fig, ax = plt.subplots(figsize=(10, 5))
data.hist(column, bins=bin_range, grid=False, ax=ax)
ax.axvline(threshold, color='r', ls='--')
ax.set_xlabel(xlabel, fontsize=18)
ax.set_ylabel('Frequency', fontsize=18)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
ax.set_title('')
fig.suptitle('Distribution of Average Internet Speeds', fontsize=20)

# Display the plot
st.pyplot(fig)

st.write("""
Both download and upload speeds were right-skewed, with the majority of the 
data points before the 50/10 Mbps cut-offs. However, since this is aggregated 
data, it is important to look at our dataset across time (demonstrated here in
years).
""")

# Create the Streamlit app
st.subheader('Average Download and Upload Speeds by Year')

# Create a selectbox for the year
year = st.selectbox('Select a year', data['year'].unique())

# Filter the data by year
filtered_data = data[data['year'] == year]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 5))
filtered_data.boxplot(column=['avg_d_kbps', 'avg_u_kbps'], 
                      grid=False, 
                      showfliers=False,
                      ax=ax)
ax.set_title(f'Average Download and Upload Speeds in {year}')
ax.set_xlabel('Speed Type')
ax.set_ylabel('Speed (kbps)')
labels = ['download', 'upload']
ax.xaxis.set_ticklabels(labels)

# Display the plot
st.pyplot(fig)

# Create the Streamlit app
st.subheader('Comparing Average Download and Upload Speeds Over Time')

# Add sliders for the year
year_range = st.slider('Select a range of years:', 
                        int(data['year'].min()), 
                        int(data['year'].max()), 
                        (int(data['year'].min()),
                         int(data['year'].max())))
year_filter = (data['year'] >= year_range[0]) & (data['year'] <= year_range[1])

# Filter the data
filtered_data = data.loc[year_filter]

# Create the plot
fig, ax = plt.subplots(figsize=(20, 10), ncols=2)
filtered_data.boxplot(column=['avg_d_kbps', 'avg_u_kbps'], 
                      by='year',
                      grid=False, 
                      showfliers=False,
                      sharey=False,
                      ax=ax)
ax[0].set_title('Average Download Speed')
ax[1].set_title('Average Upload Speed')
ax[0].set_xlabel('Year')
ax[1].set_xlabel('Year')
ax[0].set_ylabel('Download Speed (kbps)')
ax[1].set_ylabel('Upload Speed (kbps)')

# Display the plot
st.pyplot(fig)


st.write("""
Another variable we though relevant to the regression analysis was location.
Consequently, we wanted to identify trends between location/population 
indicators and internet access.
""")

# Create the Streamlit app
st.subheader('Distribution of Average Internet Speeds Across Population Centers')

# Set up selector box for average download and upload speeds
speed_type = st.selectbox("Select speed type:", ["Downloads", "Uploads"])

# Filter data based on the selected speed type
if speed_type == "Downloads":
    column = "P75_d_kbps"
else:
    column = "P75_u_kbps"

# Set up the plot
fig, axs = plt.subplots(1, 1, figsize=(20,10))

# Plot histograms
if speed_type == "Downloads":
    data.loc[lambda s:~(s.PCCLASS == 4.0)][column].plot.hist(bins=range(0,400000,4000), 
                                                               density=True, alpha=0.75,
                                                               ax=axs, grid=False,
                                                               label='Small/Medium')
    data.loc[lambda s:(s.PCCLASS == 4.0)][column].plot.hist(bins=range(0,400000,4000), 
                                                              density=True,alpha=0.75,
                                                              ax=axs, grid=False,
                                                              label='Large Urban')
    axs.axvline(50000, color='r', ls='--')
else:
    data.loc[lambda s:~(s.PCCLASS == 4.0)][column].plot.hist(bins=range(0,200000,2000), 
                                                               density=True, alpha=0.75,
                                                               ax=axs, grid=False,
                                                               label='Small/Medium')
    data.loc[lambda s:(s.PCCLASS == 4.0)][column].plot.hist(bins=range(0,200000,2000), 
                                                              density=True,alpha=0.75,
                                                              ax=axs, grid=False,
                                                              label='Large Urban')
    axs.axvline(10000, color='r', ls='--')    


# Set up plot labels and title
axs.set_xlabel(f'{speed_type} Speed (kbps)', fontsize=18)
axs.set_ylabel('')
axs.set_title('')

axs.xaxis.set_tick_params(labelsize=15)
axs.yaxis.set_tick_params(labelsize=15)
fig.supylabel('Frequency', fontsize=18)
fig.suptitle(f'Distribution of average {speed_type.lower()} speeds across ' \
             'population centers over the last four years', fontsize=20)
fig.tight_layout(pad=3)

# Show the plot
st.pyplot(fig)


st.write("""
Overall, urban population centers have greater internet access than small and  
medium population centers. Nevertheless, it may be interesting to see whether 
internet access has increased over time for rural areas as well. 
""")

# Create the Streamlit app
st.subheader('Average Download and Upload Speeds Over Time For ' \
             'Small/Medium Population Centers')

data_loc = data.loc[lambda s:~(s.PCCLASS == 4.0)]

# Add sliders for the year
year_range = st.slider('Select a range of years:', 
                        int(data_loc['year'].min()), 
                        int(data_loc['year'].max()), 
                        (int(data_loc['year'].min()),
                         int(data_loc['year'].max())),
                        key=2)
year_filter = (data_loc['year'] >= year_range[0]) & (data_loc['year'] <= year_range[1])

# Filter the data
filtered_data = data_loc.loc[year_filter]

# Create the plot
fig, ax = plt.subplots(figsize=(20, 10), ncols=2)
filtered_data.boxplot(column=['avg_d_kbps', 'avg_u_kbps'], 
                      by='year',
                      grid=False, 
                      showfliers=False,
                      sharey=False,
                      ax=ax)
ax[0].set_title('Average Download Speed')
ax[1].set_title('Average Upload Speed')
ax[0].set_xlabel('Year')
ax[1].set_xlabel('Year')
ax[0].set_ylabel('Download Speed (kbps)')
ax[1].set_ylabel('Upload Speed (kbps)')

# Display the plot
st.pyplot(fig)

st.write("""
Lastly, we investigated the relationship between reported internet access by 
Stats Canada and internet access determined by Ookla to get an idea of how 
satisfied Canadians were with their internet. We considered 
testing frequency across unique devices for the same reason.
""")

# Create the Streamlit app
st.subheader('Population (%) with 50/10 Mbps Speeds As Reported by ' \
             'Statistics Canada and Ookla')

# Add sliders for the year
year_range = st.slider('Select a range of years:', 
                        int(data['year'].min()), 
                        int(data['year'].max()), 
                        (int(data['year'].min()),
                         int(data['year'].max())),
                        key=3)
year_filter = (data['year'] >= year_range[0]) & (data['year'] <= year_range[1])

# Filter the data
filtered_data = data.loc[year_filter]

# Create the plot
fig, ax = plt.subplots(figsize=(20, 10), ncols=2)
filtered_data.boxplot(column=['Pop_Avail_50_10', 'ookla_50_10_percentile'], 
                      by='year',
                      grid=False, 
                      showfliers=False,
                      sharey=False,
                      ax=ax)
ax[0].set_title('Statistics Canada')
ax[1].set_title('Ookla')
ax[0].set_xlabel('Year')
ax[1].set_xlabel('Year')
ax[0].set_ylabel('Download Speed (kbps)')
ax[1].set_ylabel('Upload Speed (kbps)')

# Display the plot
st.pyplot(fig)

st.write("""
The distribution analysis suggest that Stats Canada's reports on internet access,
especially between 2019 and 2021 to be overly positive. Fortunately, both agencies 
report and increase in internet access across the country in the last couple of 
years.
""")

st.markdown("## Correlation Analyses")
st.sidebar.markdown("## Correlation Analyses️")

st.write("""
Thus far, we have identified trends in our dataset that point to potentially 
relevant features, namely time, location and population size, and testing.
Consequently, we conducted a correlation analysis with numerical variables in 
our dataset that were not included in the original model.
""")

st.subheader('Correlation Matrix')

# Create a selector box to choose which correlation to display
options = ['pearson', 'kendall', 'spearman']
selected_corr = st.selectbox('Select correlation method:', options)

# Compute correlation matrix
corr = data.corr(method=selected_corr, numeric_only=True)

# Plot heatmap
fig, ax = plt.subplots(figsize=(20, 17))
sns.heatmap(corr, cmap='bwr_r', ax=ax)

# Show plot
st.pyplot(fig)

st.subheader('Feature Correlation to Internet Speeds')

# correlation to the average internet speeds
corr2 = corr[['avg_d_kbps', 
              'avg_u_kbps']].drop(['avg_d_kbps', 
                                   'avg_u_kbps'], axis=0).reset_index()


# Sidebar selector
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(25, 15), sharey=True)

# Plot for Downloads
sns.barplot(data=corr2, x='avg_d_kbps', y='index', ax=axs[0],
            palette=corr2['avg_d_kbps'].apply(lambda x: 'b' if x > 0 else 'r'),
            orient='h')
axs[0].set(xlim=(-1, 1), ylabel='variable', title='Download Speeds (kpbs)')
axs[0].set_xlabel('Download Speed (kpbs)', fontsize=15)
axs[0].set_ylabel('Feature', fontsize=15)


# Plot for Uploads
sns.barplot(data=corr2, x='avg_u_kbps', y='index', ax=axs[1],
            palette=corr2['avg_u_kbps'].apply(lambda x: 'b' if x > 0 else 'r'),
            orient='h')
axs[1].set(xlim=(-1, 1), ylabel=None, title='Upload Speeds (kpbs)')
axs[1].set_xlabel('Upload Speed (kpbs)', fontsize=15)


# Show plot
st.pyplot(fig)

st.write("""
Interestingly, most features are positively correlated with internet speed. To 
avoid the effects of intercorrelation and bias, we do not recommend using features 
that are calculated from speed testing (e.g., `avg_lat_ms`). Furthermore, although 
there are many categorical variables to choose from, we decided to select features 
that were identified in our EDA and leave out complementary features. For instance,
`Pop_Avail_50_10`, `TDwell_Avail_50_10` and  `UDwell_Avail_50_10` were used in the 
regression model as indicators of location and population size but 
`Pop2016_at_50_10_Combined`, `TDwell2016_at_50_10_Combined` and `UDwell2016_at_50_10_Combined` were not.

The features used to improve the linear model were `PRCODE`, `PCCLASS`, `year`, `quarter`, 
`PHH_Count`, `Common_Type`, `Pop_Avail_50_10`, `TDwell_Avail_50_10`, `URDwell_Avail_50_10` 
and `test_frequency`
											
""")

st.markdown("# Improving the Simple Linear Model")
st.sidebar.markdown("# Improving the Simple Linear Model")

st.write("""
We built a simple linear regression model with Ridge using the features outlined above.
Although our model was better than the initial model, the perfect fit between the actual 
and model data indicates that it is overifitting the data and may not be suitable for 
our purposes. Hence, we decided to give time series analysis a shot.
""")

#uploaded_file = st.file_uploader(src.config.DATA_DIRECTORY / "processed" / "statistical_geometries" / "time" / "downloads_ml.png", type="png")
d_filename = "downloads_ml.png"
image_path = data_dir / d_filename

d_image = Image.open(image_path)
st.image(d_image)

u_filename = "uploads_ml.png"
image_path = data_dir / u_filename

u_image = Image.open(image_path)
st.image(u_image)

st.write("""

""")

