# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:31:21 2024

@author: panne
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns

# Define countries as a global variable
countries = ['India', 'United States', 'China']

def read_file(filename):
    """ Extracts data from a CSV file and provides cleaned dataframes with columns for years and countries.

    Args:
    filename (str): The CSV file name from which the data is to be read.

    Returns:
    df_years (pandas.DataFrame): Dataframe having years as columns and nations and indicators as rows.
    df_countries (pandas.DataFrame): Dataframe having countries as columns and years and indicators as rows.
    """

    # read the CSV file and skip the first 4 rows
    df = pd.read_csv(filename, skiprows=4)

    # drop unnecessary columns
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    df = df.drop(cols_to_drop, axis=1)

    # rename remaining columns
    df = df.rename(columns={'Country Name': 'Country'})

    # melt the dataframe to convert years to a single column
    df = df.melt(id_vars=['Country', 'Indicator Name'],
                 var_name='Year', value_name='Value')

    # convert year column to integer and value column to float
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # separate dataframes with years and countries as columns
    df_years = df.pivot_table(
        index=['Country', 'Indicator Name'], columns='Year', values='Value')
    df_countries = df.pivot_table(
        index=['Year', 'Indicator Name'], columns='Country', values='Value')

    # clean the data
    df_years = df_years.dropna(how='all', axis=1)
    df_countries = df_countries.dropna(how='all', axis=1)

    return df_years, df_countries


def data_subset(df_years, countries, indicators, start_year, end_year):
    """ Subsets the data for the chosen indicators, nations, and years.

    Args:
    df_years (pandas.DataFrame): Dataframe with years as columns and nations and indicators as rows.
    countries (list): List of country names to subset.
    indicators (list): List of indicator names to subset.
    start_year (int): The starting year for the subset.
    end_year (int): The ending year for the subset.

    Returns:
    df_subset (pandas.DataFrame): Subset of the data containing specified indicators, nations, and years.
    """

    years = list(range(start_year, end_year + 1))
    df = df_years.loc[(countries, indicators), years]
    df = df.transpose()
    return df


def plot_clustered_bar_chart(df, size=6):
    """Generates a clustered bar chart for visualizing the correlation matrix.

    Args:
    df (pandas.DataFrame): DataFrame containing data for correlation calculation.
    size (int): Plot's vertical and horizontal dimensions (in inches).

    There is no plt.show() at the conclusion of the function to allow the user to save the figure.
    """

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))

    # Create a clustered bar chart
    bar_width = 0.35
    bar_positions = np.arange(len(corr.columns))
    ax.bar(bar_positions, corr.mean(), bar_width, label='Mean Correlation', color='skyblue')

    # setting ticks to column names
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(corr.columns, rotation=90)

    # add title and adjust layout
    ax.set_title('Average Correlation for Selected Countries', fontsize=14)
    plt.tight_layout()

    # Display the correlation values on top of each bar
    for i, value in enumerate(corr.mean()):
        ax.text(i, value + 0.01, f'{value:.2f}', ha='center', va='bottom')

    # Display the colorbar as a reference for correlation values
    cmap = plt.cm.inferno_r
    norm = plt.Normalize(corr.min().min(), corr.max().max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Correlation', rotation=270, labelpad=15)



def data_normalize(df):
    """Normalizes the data using StandardScaler.

    Args:
    df (pandas.DataFrame): Dataframe to be normalized.

    Returns:
    df_normalized (pandas.DataFrame): Normalized dataframe.
    """

    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized


def perform_kmeans_clustering(df, num_clusters):
    """Applies k-means clustering to the specified dataframe.

    Args:
    df (pandas.DataFrame): DataFrame to cluster.
    num_clusters (int): Number of clusters.

    Returns:
    cluster_labels (numpy.ndarray): Cluster labels for each data point.
    """

    # Create a KMeans instance with the specified number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Fit the model and predict the cluster labels for each data point
    cluster_labels = kmeans.fit_predict(df)

    return cluster_labels


def plot_clustered_data(df, cluster_labels, cluster_centers):
    """Plots the data points and cluster centers.

    Args:
    df (pandas.DataFrame): Dataframe containing the data points.
    cluster_labels (numpy.ndarray): Array of cluster labels for each data point.
    cluster_centers (numpy.ndarray): Array of cluster centers.
    """
    # Set the style of the plot
    plt.style.use('seaborn')

    # Create a scatter plot of the data points, colored by cluster label
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1],
                         c=cluster_labels, cmap='rainbow')

    # Plot the cluster centers as black X's
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, marker='h', c='black')

    # Set the x and y axis labels and title
    ax.set_xlabel(df.columns[0], fontsize=12)
    ax.set_ylabel(df.columns[1], fontsize=12)
    ax.set_title("K-Means Clustering Results", fontsize=14)

    # Add a grid and colorbar to the plot
    ax.grid(True)
    plt.colorbar(scatter)

    # Show the plot
    plt.show()


def print_cluster_summary(cluster_labels):
    """Prints a summary of the number of data points in each cluster.

    Args:
    cluster_labels (numpy.ndarray): Array of cluster labels for each data point.
    """
    cluster_counts = np.bincount(cluster_labels)
    for i, count in enumerate(cluster_counts):
        print(f"Cluster {i + 1}: {count} data points")


def filter_environment_data(filename, countries, indicators, start_year, end_year):
    """Reads a CSV file with environmental data, filters it based on indicators and nations,
    and outputs a dataframe with rows representing indicators and countries and columns representing years.

    Args:
    filename (str): Path to the CSV file.
    countries (list): List of country names to filter by.
    indicators (list): List of indicator names to filter by.
    start_year (int): Starting year for data filtering.
    end_year (int): Ending year for data filtering.

    Returns:
    env_data (pandas.DataFrame): Filtered and rotated data on environmental indicators.
    """

    # read the CSV file and skip the first 4 rows
    env_data = pd.read_csv(filename, skiprows=4)

    # drop unnecessary columns
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    env_data = env_data.drop(cols_to_drop, axis=1)

    # rename remaining columns
    env_data = env_data.rename(columns={'Country Name': 'Country'})

    # filter data by selected countries and indicators
    env_data = env_data[env_data['Country'].isin(countries) &
                        env_data['Indicator Name'].isin(indicators)]

    # melt the dataframe to convert years to a single column
    env_data = env_data.melt(id_vars=['Country', 'Indicator Name'],
                             var_name='Year', value_name='Value')

    # convert year column to integer and value column to float
    env_data['Year'] = pd.to_numeric(env_data['Year'], errors='coerce')
    env_data['Value'] = pd.to_numeric(env_data['Value'], errors='coerce')

    # pivot the dataframe to create a single dataframe with years as columns and countries and indicators as rows
    env_data = env_data.pivot_table(index=['Country', 'Indicator Name'],
                                    columns='Year', values='Value')

    # select specific years
    env_data = env_data.loc[:, start_year:end_year]

    return env_data


def exponential_growth(x, a, b):
    return a * np.exp(b * x)


def confidence_intervals(xdata, ydata, popt, pcov, alpha=0.05):
    n = len(ydata)
    m = len(popt)
    df = max(0, n - m)
    tval = -1 * stats.t.ppf(alpha / 2, df)
    residuals = ydata - exponential_growth(xdata, *popt)
    stdev = np.sqrt(np.sum(residuals**2) / df)
    ci = tval * stdev * np.sqrt(1 + np.diag(pcov))
    return ci


def predict_future_values(env_data, countries, indicators, start_year, end_year):
    # select data for the given countries, indicators, and years
    data = filter_environment_data(env_data, countries, [indicators], start_year, end_year)

    # calculate the growth rate for each country and year
    growth_rate = np.zeros(data.shape)
    for i in range(data.shape[0]):
        popt, pcov = curve_fit(exponential_growth, np.arange(data.shape[1]), data.iloc[i])
        ci = confidence_intervals(np.arange(data.shape[1]), data.iloc[i], popt, pcov)
        growth_rate[i] = popt[1]

    # plot the growth rate for each country
    fig, ax = plt.subplots()
    for i in range(data.shape[0]):
        ax.plot(np.arange(data.shape[1]), data.iloc[i],
                label=data.index.get_level_values('Country')[i])
    ax.set_xlabel('Year')
    ax.set_ylabel('Indicator Value')
    ax.set_title(indicators)
    ax.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    # Read the data
    df_years, df_countries = read_file(r"C:\Users\panne\Downloads\worlddata.csv")

    # Define new indicators and years
    new_indicators = [
        'Urban population (% of total population)',
        'CO2 emissions (metric tons per capita)'
    ]
    countries = ['United Kingdom', 'China', 'United States']

    # Subset the data for the new indicators and selected countries
    df = data_subset(df_years, countries, new_indicators, 1999, 2018)

    # Normalize the data
    df_normalized = data_normalize(df)

    # Perform clustering
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(df_normalized)
    cluster_centers = kmeans.cluster_centers_

    print("Clustering Results points cluster_centers")
    print(cluster_centers)

    # Plot the results
    plot_clustered_data(df_normalized, cluster_labels, cluster_centers)

    # Predict future values for the new indicators
    predict_future_values(r"C:\Users\panne\Downloads\worlddata.csv", countries, 'CO2 emissions (metric tons per capita)', 1999, 2018)

    # Create a correlation heatmap for the new indicators
    plot_clustered_bar_chart(df, size=6)

    # Print a summary of the number of data points in each cluster
    print_cluster_summary(cluster_labels)
