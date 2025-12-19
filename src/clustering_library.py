import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class DataCleaner:
    """
    Cleaning and preprocessing retail transaction data

    This class handles data loading, cleaning operations, and basic exploratory
    data analysis for online retail datasets.
    """

    def __init__(self, data_path):
        """
        Args:
            data_path (str): Path to raw data file
        """
        self.data_path = data_path
        self.df = None
        self.df_ul = None
        self.rfm_data = None

    def load_data(self):
        """
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        dtype = dict(
            InvoiceNo=np.object_,
            StockCode=np.object_,
            Description=np.object_,
            Quantity=np.int64,
            UnitPrice=np.float64,
            CustomerID=np.object_,
            Country=np.object_,
        )
        
        self.df = pd.read_csv(
            self.data_path,
            encoding="ISO-8859-1",
            parse_dates=["InvoiceDate"],
            dtype=dtype,
        )

        self.df["CustomerID"] = (
            self.df["CustomerID"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(6)
        )
        
        print(f"Data size: {self.df.shape}")
        print(f"Number of Records: {len(self.df):,}")
        
        return self.df
    
    def clean_data(self):
        """
        Clean the dataset by removing invalid records and focusing or UK customers.add()
        
        Returns:
            pd.DataFrame: Cleaned UK dataset
        """
        # Add TotalPrice column
        self.df["TotalPrice"] = self.df["Quantity"] * self.df["UnitPrice"]
        # Remove cancelled invoices (starting with 'C')
        self.df = self.df[~self.df["InvoiceNo"].astype(str).str.startswith("C")]
        # Focus on UK customers
        self.df_uk = self.df[self.df["Country"] == "United Kingdom"].copy()
        
        # Remove invalid quantiy or unit price
        self.df_uk = self.df_uk[
            (self.df_uk["Quantity"] > 0) & (self.df_uk["UnitPrice"] > 0)
        ]
        
        return self.df_uk
    
    def create_time_features(self):
        """
        Create time-based features for analysis.
        """
        self.df_uk["DayOfWeek"] = self.df_uk["InvoiceDate"].dt.dayofweek
        self.df_uk["HourOfDay"] = self.df_uk["InvoiceDate"].dt.hour
        
    def calculate_rfm(self):
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics.
        
        Returns:
            pd.DataFrameL: RFM data for each customer
        """
        snapshot_date = self.df_uk["InvoiceDate"].max() + pd.Timedelta(days=1)
        
        self.rfm_data = self.df_uk.groupby("CustomerID").agg(
            {
                "InvoiceDate": lambda x: (snapshot_date - x.max()).days, # Recency
                "InvoiceNo": lambda x: len(x.unique()), # Frequency
                "TotalPrice": lambda x: x.sum(), # Monetary       
            }
        )
        
        self.rfm_data.columns = ["Recency", "Frequency", "Monetary"]
        return self.rfm_data
    
    def save_cleaned_data(self, output_dir="../data/processed"):
        """
        Save cleaned data to specified directory.
        
        Args:
            output_dir (str): Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        self.df_uk.to_csv(f"{output_dir}/cleaned_uk_data.csv", index=False)
        print(f"Cleaned data saved to {output_dir}/cleaned_uk_data.csv")


class FeatureEngineer:
    def __init__(self):
        pass


class ClusterAnalyzer:
    def __init__(self):
        pass


class DataVisualizer():
    """
    Creating visualizations for customer segmentation analysis.

    This class provides methods for plotting various aspects of the data 
    including temporal patterns, customer behvior, and cluster analyis.
    """

    def __init__(self):
        """Initialize the DataVisualize with plotting  settings."""
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("viridis")

    def plot_revenue_over_time(self, df):
        """
        Plot daily and monthly revenue patterns.

        Args:
            df (pd.DataFrame): Dataframe with InvoiceDate and TotalPrice columns
        """
        # Daily revenue
        plt.figure(figsize=(12, 5))
        daily_revenue = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()
        daily_revenue.plot()
        plt.title("Daily Revenue")
        plt.xlabel("Day")
        plt.ylabel("Revenue (GBP)")
        plt.tight_layout()
        plt.show()
        
        # Monthly revenue
        plt.figure(figsize=(12, 5))
        monthly_revenue = df.groupby(pd.Grouper(key="InvoiceDate", freq="M"))[
            "TotalPrice"
        ].sum()
        monthly_revenue.plot(kind="bar")
        plt.title("Monthly Revenue")
        plt.xlabel("Month")
        plt.ylabel("Revenue (GPB)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_time_patterns(self, df):
        """
        Plot purchase patterns by day and hour.
        
        Args:
            df (pd.DataFrame): DataFrame with time features
        """
        plt.figure(figsize=(12, 5))
        day_hour_counts = (
            df.groupby(["DayOfWeek", "HourOfDay"]).size().unstack(fill_value=0)
        )
        sns.heatmap(day_hour_counts, cmap="viridis")
        plt.xlabel("Shopping activity by day time")
        plt.ylabel("Time in day")
        plt.tight_layout()
        plt.show()
        
    def plot_product_analysis(self, df, top_n=10):
        """
        Plot top products by quantity and revenue.

        Args:
            df (pd.DataFrame): Transaction dataframe
            top_n (int): Number of top products to show
        """
        plt.figure(figsize=(12, 5))
        top_products = (
            df.groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_products.values, y=top_products.index)
        plt.title(f"Top {top_n} products by sales volume")
        plt.xlabel("Quantity")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 5))
        top_revenue_products = (
            df.groupby("Description")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_revenue_products.values, y=top_revenue_products.index)
        plt.title(f"Top {top_n} by sales volume")
        plt.xlabel("Revenue (BGP)")
        plt.tight_layout()
        plt.show()
        
    def plot_customer_distribution(self, df):
        """
        Plot customer behavior distributions.
        
        Args:
            df (pd.DataFrame): Transaction dataframe
        """
        plt.figure(figsize=(10, 5))
        transactions_per_customer = df.groupby("CustomerID")["InvoiceNo"].nunique()
        sns.histplot(transactions_per_customer, bins=30, kde=True)
        plt.title("Distribution of Transactions per Customer")
        plt.xlabel("Number of transactions")
        plt.ylabel("Number of Customers")
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10, 5))
        spend_per_customer = df.groupby("CustomerID")["TotalPrice"].sum()
        spend_filter = spend_per_customer < spend_per_customer.quantile(0.99)
        sns.histplot(spend_per_customer[spend_filter], bins=30, kde=True)
        plt.title("Distribution of Spend per Customer")
        plt.xlabel("Total Spend (GBP)")
        plt.ylabel("Number of Customers")
        plt.tight_layout()
        plt.show()
        
    def plot_rfm_analysis(self, rfm_data):
        """
        Plot RFM score distributions.
        
        Args:
            rfm_data (pd.DataFrame): DataFrame with RFM scores
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        sns.histplot(rfm_data["Recency"], bins=30, kde=True, ax=axes[0])
        axes[0].set_title("Recency Distribution")
        axes[0].set_xlabel("Recency (days)")
        
        sns.histplot(rfm_data["Frequency"], bins=30, kde=True, ax=axes[1])
        axes[1].set_title("Frequency Distribution")
        axes[1].set_xlabel("Frequency (number of purchases)")
        
        monetary_filter = rfm_data["Monetary"] < rfm_data["Monetary"].quantile(0.99)
        sns.histplot(
            rfm_data["Monetary"][monetary_filter], bins=30, kde=True, ax=axes[2]
        )
        axes[2].set_title("Monetary Distribution")
        axes[2].set_xlabel("Monetary (GBP)")
        plt.tight_layout()
        plt.show()