import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
            .str.replace(".0",)
        )


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