import subprocess
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import io
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



API_URL = "http://195.91.221.96:55555/docs"

st.title("Data Analysis Project")
st.markdown("""
    Mountains vs. Beaches Preferences.
    This dataset aims to analyze public preferences between two popular vacation types: mountains and beaches. It provides insights into various demographic and lifestyle factors that may influence these preferences. By exploring this dataset, users can apply multiple machine learning techniques to predict whether individuals prefer mountains or beaches based on their attributes.
    Features
    The dataset contains the following features:

    - Age: Age of the individual (numerical).
    - Gender: Gender identity of the individual (categorical: male, female, non-binary).
    - Income: Annual income of the individual (numerical).
    - Education Level: Highest level of education attained (categorical: high school, bachelor, master, doctorate).
    - Travel Frequency: Number of vacations taken per year (numerical).
    - Preferred Activities: Activities preferred by individuals during vacations (categorical: hiking, swimming, skiing, sunbathing).
    - Vacation Budget: Budget allocated for vacations (numerical).
    - Location: Type of residence (categorical: urban, suburban, rural).
    - Proximity to Mountains: Distance from the nearest mountains (numerical, in miles).
    - Proximity to Beaches: Distance from the nearest beaches (numerical, in miles).
    - Favorite Season: Preferred season for vacations (categorical: summer, winter, spring, fall).
    - Pets: Indicates whether the individual owns pets (binary: 0 = No, 1 = Yes).
    - Environmental Concerns: Indicates whether the individual has environmental concerns (binary: 0 = No, 1 = Yes).
    """)

@st.cache_data
def fetch_all_data(batch_size=100):
    """
    Fetch the entire dataset from the REST API.
    """
    start = 0
    all_data = []

    while True:
        response = requests.get(f"{API_URL}/data", params={"start": start, "limit": batch_size})
        response.raise_for_status()
        data_batch = response.json()

        if not data_batch:
            break

        all_data.extend(data_batch)
        start += batch_size

    return pd.DataFrame(all_data)

def add_new_column(API_URL, column_name, values):
    """
    Send a request to the backend to add a new column to the dataset.

    Parameters:
    - api_url (str): The base URL of the backend API (e.g., 'http://localhost:8000').
    - column_name (str): The name of the new column to add.
    - values (list): A list of values for the new column.

    Returns:
    - dict: The response from the backend API.
    """

    payload = {
        "column_name": column_name,
        "values": values
    }

    response = requests.post(f"{API_URL}/add_column", json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}

def plot_preference_distribution(df):
    fig = px.pie(df,
                names='Preference_Words',
                title='Preference Distribution',
                color_discrete_sequence=['rgb(50, 168, 82)', 'rgb(245, 110, 124)']
                )
    st.plotly_chart(fig)

def plot_gender(df):
    gender_preference_counts = df.groupby(['Gender', 'Preference_Words']).size().reset_index(name='Count')

    fig = px.bar(
        gender_preference_counts, 
        x='Gender', 
        y='Count', 
        color='Preference_Words', 
        barmode='group',
        title='Gender Impact on Mountain vs Beaches Preferences',
        color_discrete_sequence=['rgb(50, 168, 82)', 'rgb(245, 110, 124)']
    )
    st.plotly_chart(fig)

def plot_histogram(df,column_name):
    counts = df.groupby([column_name]).size().reset_index(name='Count')
    fig = px.histogram(counts,
                    x=column_name,
                    y="Count"
                    )
    st.plotly_chart(fig)

def plot_scatter(df, x, y, color, symbol, category_orders):
    fig = px.scatter(df,
                 x=x,
                 y=y,
                 color=color,
                 symbol=symbol,
                 color_discrete_sequence=['rgb(50, 168, 82)', 'rgb(245, 110, 124)'],
                 category_orders=category_orders)
    st.plotly_chart(fig)

def plot_heatmap(df, x, y, title):
    fig = px.density_heatmap(df, 
                            x=x, 
                            y=y,
                            title=title,
                            marginal_x="histogram", 
                            marginal_y="histogram",
                            color_continuous_scale=px.colors.sequential.Viridis)

    st.plotly_chart(fig)

def plot_clustermap(pt):
    fig = sns.clustermap(
        pt,
        cmap="viridis",
        figsize=(10, 8),
        linewidths=0,
        cbar_kws={'label': 'Count'})
    st.pyplot(fig)

def plot_correlation(df):
    df_copy = df.copy()
    df_copy = df_copy.drop(columns=['Preference_Words'])
    df_copy = df_copy.drop(columns=['Income_Category'])
    df_copy = df_copy.drop(columns=['Budget_Category'])
    df_copy = df_copy.drop(columns=['Travel_Frequency_Category'])

    categorical_cols = df_copy.select_dtypes(include='object').columns
    le = LabelEncoder()

    for col in categorical_cols:
        df_copy[col] = le.fit_transform(df[col])

    correlation_matrix = df_copy.corr()

    fig = px.imshow(
        correlation_matrix,
        text_auto='.2f',
        color_continuous_scale='magma', 
        labels={'color': 'Correlation'},
        title="Correlation Heatmap"
    )

    fig.update_layout(
        title_x=0.5,
        width=1000,
        height=800,
    )

    st.plotly_chart(fig)




st.write("## Data clean up")
st.write("Read the dataset and output few first rows.")
df = fetch_all_data()
st.dataframe(df.head())
st.write("Let's look at our data and understand whether we have to clean it or no. What values do we have?")

info_table = pd.DataFrame({
    "Column": df.columns,
    "Non-Null Count": [df[col].notnull().sum() for col in df.columns],
    "Dtype": [df[col].dtype for col in df.columns]
})

st.write("Let's look at our data and understand whether we have to clean it or not. What values do we have?")
st.text(info_table)

st.write("Objects and int64s. No nulls in the table!")
st.dataframe(df.isnull().sum())
st.write("As we can see there are no no NaNs in the table. Also there are no numbers written as “1M+”, “10k+”, “1k+”.")

st.write("Let's check if there's any incorrect values in the data frame. For example, if there are any negative numbers.")
row1_col1, row1_col2 = st.columns(2)  
row2_col1, row2_col2 = st.columns(2)  
row3_col1, row3_col2 = st.columns(2)
row4_col1, row4_col2 = st.columns(2)

row1_col1.subheader("Income")
row1_col1.dataframe(df['Income'].describe())
row1_col2.subheader("Travel Frequency")
row1_col2.dataframe(df['Travel_Frequency'].describe())
row2_col1.subheader("Vacation Budget")
row2_col1.dataframe(df['Vacation_Budget'].describe())
row2_col2.subheader("Proximity to Mountains")
row2_col2.dataframe(df['Proximity_to_Mountains'].describe())
row3_col1.subheader("Proximity to Beaches")
row3_col1.dataframe(df['Proximity_to_Beaches'].describe())
row3_col2.subheader("Pets")
row3_col2.dataframe(df['Pets'].describe())
row4_col1.subheader("Environmental Concerns")
row4_col1.dataframe(df['Environmental_Concerns'].describe())
row4_col2.subheader("Preference")
row4_col2.dataframe(df['Preference'].describe())

st.write("Looks good!")


st.write("# Overview")

df["Preference_Words"] = df["Preference"].map({1: "Mountains", 0: "Beaches"})
values = df["Preference_Words"].tolist()
add_new_column(API_URL, 'Prefrence_Words', values)

st.write("Here's statistics.")
cols = ['Age', 'Income', 'Travel_Frequency', 'Vacation_Budget', 'Proximity_to_Mountains', 'Proximity_to_Beaches']
st.dataframe(df[cols].describe())
st.write("here's what our data looks like.")
st.write("Let's look at preference distribution.")
pie_chart = plot_preference_distribution(df)
st.write("This pie-chart shows us that there are more people choosing beaches than people who choose mountains.Now let's look at how gender impact on the preference.")

bar_plot = plot_gender(df)
st.write("Preference distribution looks almost the same for any gender.")

st.write("Here's the income and vacational budget distribution for our dataset.")
plot_histogram(df,'Income')
st.write('The quantity of people with different incomes is more over the same within the given range (20,000 to 120,000), with no significant skew or outliers.')

plot_histogram(df,'Vacation_Budget')
st.write('The vacation budget seems to be evenly distributed as well within the given range from 1,000 to 5,000.')

st.write('What are the median incomes of people who choose mountains and people who choose beaches?')
median_income_beaches = df[df['Preference_Words'] == 'Beaches']['Income'].median()
median_income_mountains = df[df['Preference_Words'] == 'Mountains']['Income'].median()
st.write(f"Median income for people who prefer Beaches: {median_income_beaches}")
st.write(f"Median income for people who prefer Mountains: {median_income_mountains}")
st.write('Median incomes are more over the same for any preference.')

st.write('# More detailed overview')
st.write("Now, let's look at how proximity to either beaches or mountains influences preferences of people with different educational levels. First let's look at people with high educational level such as doctrate and master.")
plot_scatter(df[(df['Education_Level'] == 'doctorate') | (df['Education_Level'] == 'master')],
             "Proximity_to_Mountains",
             "Proximity_to_Beaches",
             'Preference_Words',
             None,
             {'Preference_Words': ['Beaches', 'Mountains']})
st.write("Insights: ")
st.write("1. Most people will choose beaches.")
st.write("2. Less people will choose beaches if mountains are closer than beaches.")
st.write("3. People choose mountain only if they are closer than beaches.")

st.write("People with lower educational level:")
plot_scatter(df[(df['Education_Level'] == 'high school') | (df['Education_Level'] == 'bachelor')],
             "Proximity_to_Mountains",
             "Proximity_to_Beaches",
             'Preference_Words',
             None,
             {'Preference_Words': ['Beaches', 'Mountains']})
st.write('We get the same results for people with lower educational level. Thus, we can conclude that the relationship between the educational level and preference is not significant. What is closer to people influences their choice the most.')

st.write('For a more detailed analysis we can look at each of the categories one by one.')
plot_scatter(df,
                "Proximity_to_Mountains",
                "Proximity_to_Beaches",
                'Preference_Words',
                "Education_Level",
                {'Preference_Words': ['Beaches', 'Mountains']}
                )
st.write('It is worth to mention that individuals tend to select mountains as their preferred destination only when they are situated closer to them than beaches.')


df['Income_Category'] = pd.qcut(df['Income'], q=5, labels=['low income', 'low-middle income', 'middle income', 'high-middle income', 'high income'])
df['Budget_Category'] = pd.qcut(df['Vacation_Budget'], q=5, labels=['low budget', 'low-middle budget', 'middle budget', 'high-middle budget', 'high budget'])
pivot_table = pd.pivot_table(df, index=['Income_Category'],
    columns=['Budget_Category'], values="Preference", aggfunc="mean").round(3)
pivot_table_reset = pivot_table.reset_index()
st.dataframe(pivot_table_reset)
st.write('This table calculates the percentage of people in each income-budget group who prefer mountains. Thus the repcentage of people who prefer beaches for each group will be (100 - income-budget group %).')
st.write('1. Preferences for mountains vs beaches are fairly balanced across all income-budget groups, with values flactuating between 0.23 and 0.27. This proves that neither income nor budget is a strong determinant of vacation preferences.')
st.write("2. There are some changes, but it’s quite small, so it’s not clear whether it’s significant or not.")

mountains_df = df[df['Preference_Words'] == 'Mountains']
plot_heatmap(mountains_df,
            "Income",
            "Vacation_Budget",
            "Density Heatmap of Income and Vacation Budget for Mountains Preference")
beaches_df = df[df['Preference_Words'] == 'Beaches']
plot_heatmap(beaches_df,
            "Income",
            "Vacation_Budget",
            "Density Heatmap of Income and Vacation Budget for Beaches Preference")
st.write("Looking at this heatmaps together, we can make few conlusions.")
st.write("- Distribution is more uniform across income levels on the density heatmap for people who prefer beaches. It happens due to a broader appeal of beaches compared to mountains, potentially because of accessibility or popularity.")
st.write("- There are about 3 times more people with high income (115k - 120k) who prefer beaches than those who have the same income but choose mountains.")

mountains_df = df[df['Preference_Words'] == 'Mountains'].copy()
income_bins = pd.cut(mountains_df['Income'], bins=10)
budget_bins = pd.cut(mountains_df['Vacation_Budget'], bins=10)
income_bin_labels = [f"{int(i.left)}\\$ - {int(i.right)}\\$" for i in income_bins.cat.categories]
budget_bin_labels = [f"{int(i.left)}\\$ - {int(i.right)}\\$" for i in budget_bins.cat.categories]
mountains_df['Income_Bin'] = pd.cut(mountains_df['Income'], bins=10, labels=income_bin_labels)
mountains_df['Vacation_Budget_Bin'] = pd.cut(mountains_df['Vacation_Budget'], bins=10, labels=budget_bin_labels)

pivot_table = mountains_df.pivot_table(
    index='Income_Bin', 
    columns='Vacation_Budget_Bin', 
    values='Income',
    aggfunc='count',
    fill_value=0
)
plot_clustermap(pivot_table)
st.write("Here's a cluster map for people who prefer mountains. This is what we can couclude from it.")
st.write("- People with higher income tend to choose a higher vacation budget.")
st.write("- Lower budgets are usually chosen by people with lower or middle income.")

st.write("Now, let's look at how preferd activities impact on the choice.")
cols = [ "Proximity_to_Mountains","Proximity_to_Beaches" ,"Preferred_Activities","Preference_Words" ]
dfff = df[cols]
act = ['skiing', 'hiking', 'swimming', 'sunbathing']
for activity in act:
        plot_scatter(df[(df['Preferred_Activities'] == activity)],
                "Proximity_to_Mountains",
                "Proximity_to_Beaches",
                "Preference_Words",
                "Preferred_Activities",
                {'Preference_Words': ['Beaches', 'Mountains']}
                )
st.write("Here's what these graphs show us:")
st.write("1. Choice of people who enjoy skiing depends on proximity, they will most likely choose what is closer to them.")
st.write("2. Choice of people who's hobby is hiking also depends on proximity, they probably will choose what is closer to them.")
st.write("2. People who like swimming choose only beaches.")
st.write("4. Same goes for people who prefer sunbathing, they all choose beaches.")

st.write("# Hypothesis:")
st.write("Individuals with higher incomes and vacation budgets who travel frequently are more likely to prefer beaches over mountains, while those with lower incomes and vacation budgets but similar travel frequency prefer mountains. Gender differences may amplify this trend.")
df['Income_Category'] = pd.qcut(df['Income'], q=3, labels=['Low Income', 'Medium Income', 'High Income'])
df['Budget_Category'] = pd.qcut(df['Vacation_Budget'], q=3, labels=['Low Budget', 'Medium Budget', 'High Budget'])
df['Travel_Frequency_Category'] = pd.qcut(df['Travel_Frequency'], q=3, labels=['Rarely', 'Sometimes', 'Often'])
grouped_data = df.groupby(['Income_Category', 'Budget_Category', 'Travel_Frequency_Category', 'Preference_Words']).size().reset_index(name='Count')
pivot_data = grouped_data.pivot_table(
    index=['Income_Category', 'Budget_Category', 'Travel_Frequency_Category'],
    columns=['Preference_Words'],
    values='Count',
    fill_value=0
)
pivot_table_reset = pivot_table.reset_index()
st.dataframe(pivot_table_reset)
fig = px.bar(
    grouped_data, 
    x='Income_Category', 
    y='Count', 
    color='Preference_Words', 
    facet_col='Budget_Category', 
    facet_row='Travel_Frequency_Category', 
    barmode='group', 
    title='Preference Analysis by Income, Budget, Travel Frequency, and Gender', 
    color_discrete_sequence=['rgb(50, 168, 82)', 'rgb(245, 110, 124)'], 
    text='Count',
    height=800,
    width=1000
)
fig.update_layout(
    xaxis_title="Income Category",
    yaxis_title="Count",
    showlegend=True,
    font=dict(size=10),
)
st.plotly_chart(fig)

grouped_data = df.groupby(['Income_Category', 'Budget_Category', 'Travel_Frequency_Category', 'Gender', 'Preference_Words']).size().reset_index(name='Count')
fig = px.bar(
    grouped_data, 
    x='Income_Category', 
    y='Count', 
    color='Preference_Words',
    pattern_shape='Gender',
    facet_col='Budget_Category', 
    facet_row='Travel_Frequency_Category', 
    barmode='group', 
    title='Preference Analysis by Income, Budget, Travel Frequency, and Gender', 
    color_discrete_sequence=['rgb(50, 168, 82)', 'rgb(245, 110, 124)'], 
    category_orders={
        'Income_Category': ['Low Income', 'Medium Income', 'High Income'],
        'Budget_Category': ['Low Budget', 'Medium Budget', 'High Budget'],
        'Travel_Frequency_Category': ['Rarely', 'Sometimes', 'Often'],
        'Gender': ['Male', 'Female'],
        'Preference_Words': ['Mountains', 'Beaches']
    },
    height=1000,
    width=1200
)
fig.update_layout(
    xaxis_title="Income Category",
    yaxis_title="Count",
    legend=dict(
        orientation="h",
        x=0.5,
        y=-0.2,
        xanchor="center",
        yanchor="top",
        title="Preference and Gender"
    ),
    margin=dict(l=50, r=50, t=100, b=150),
    font=dict(size=10),
    xaxis_tickangle=-45,
    barmode='group',
    bargap=0.2,
    bargroupgap=0.1
)
fig.update_traces(
    marker_line_width=1.5,
    marker_line_color="black"
)
fig.for_each_trace(lambda t: t.update(name=t.name + " - " + t.legendgroup))
fig.for_each_trace(
    lambda t: t.update(marker=dict(pattern=dict(fillmode='overlay', size=10)))
)
st.plotly_chart(fig, use_container_width=False)

stats = df.groupby(['Income_Category', 'Budget_Category', 'Travel_Frequency_Category', 'Preference_Words']).agg(
    Mean_Income=('Income', 'mean'),
    Median_Income=('Income', 'median'),
    Mean_Budget=('Vacation_Budget', 'mean'),
    Median_Budget=('Vacation_Budget', 'median')
).reset_index()
st.write(stats)
st.write("So, as we can see, it is indeed true that individuals with higher incomes and vacation budgets who travel frequently are more likely to prefer beaches over mountains. However the second part of the hypothesis is not supported by the data. Individuals with lower incomes and vacation budgets but similar travel frequency tend to choose beaches as well.")
st.write("Here are some insights:")
st.write("1. High income and budget:")
st.write("- The count of individuals who prefer beaches in indeed higher than those who prefer mountains. This lead us to a conclusion: people with higher incomes and budgets do lean toward beaches, supporting this part of our hypothesis.")
st.write("2. Low income and budget:")
st.write("- The count of individuals prefering mountains is still lower than of those who prefer beaches. But, the gap between is narrower compared to higher-income categories. So, even though mountain preference does increase in lower income and buget groups, beaches still are a way more popular choice.")
st.write("3. There is no significant trend in travel preferences based on the gender.")

st.write("Let's look at what actually influences the preference of the people.")
plot_correlation(df)
st.write("As we see factors that affect preferences the most are prefered activities and proximity to both beaches and mountains.")

st.write("# Data transformation")
df['Income_to_Budget_Ratio'] = df['Income'] / df['Vacation_Budget']
values = df["Income_to_Budget_Ratio"].tolist()
add_new_column(API_URL, 'Income_to_Budget_Ratio', values)

bins = [0, 25, 45, 65, 100]
labels = ['Youth', 'Adult', 'Middle-aged', 'Senior']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
values = df["Age_Group"].tolist()
add_new_column(API_URL, 'Age_Group', values)
st.dataframe(df.head())