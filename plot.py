import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def create_box_plot(data, x, y, hue=None, whis=1.5, showmeans=True, showfliers=True):
    """
    Creates a box plot with customizable options.

    Parameters:
    - data: DataFrame containing the data.
    - x: String, name of the column for x-axis.
    - y: String, name of the column for y-axis.
    - hue: String, name of the column for hue (grouping variable).
    - whis: Float, whisker length for the box plot.
    - showmeans: Boolean, whether to show mean points.
    - showfliers: Boolean, whether to show outlier points.
    - cap_color: String, color for the caps of the boxes.
    """
    
    # Set a ggplot style
    sns.set_theme(style="whitegrid")

    # Create a figure and axis
    plt.figure(figsize=(10, 6))

    # Plot the box plot
    box_plot = sns.boxplot(
        data=data, 
        x=x, 
        y=y, 
        hue=hue, 
        gap=0.1,
        whis=whis,  # Set whiskers length
        showmeans=showmeans, 
        showfliers=showfliers,  # Show individual outliers
        flierprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},  # Set outlier color
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 8},
        capprops={"linewidth": 0 }  # Customize the caps' color and line width
    )

    # Define global y-limits for the entire plot based on data range
    global_min = data[y].min()
    global_max = data[y].max()

    # Add a long vertical line for each hue within each segment, placing it behind the boxes
    if hue is not None:
        for i, segment in enumerate(data[x].unique()):
            for j, hue_val in enumerate(data[hue].unique()):
                # Position the line between each hue in the segment
                plt.plot(
                    [i - 0.2 + 0.4 * j, i - 0.2 + 0.4 * j], 
                    [global_min, global_max], 
                    color='black', 
                    linestyle='-', 
                    linewidth=0.7,
                    zorder=1  # Lower z-order to place the line behind the boxes
                )

    # Set the y-limits for the entire plot to maintain uniformity
    plt.ylim(global_min, global_max)

    # Make the horizontal grid lines lighter
    plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.5)  # Only horizontal grid lines

    # Remove plot borders, keeping only the x-axis border
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.gca().spines['bottom'].set_visible(True)  # Keep the bottom spine (x-axis)
    plt.gca().spines['bottom'].set_color('lightgrey')  # Change the x-axis spine color
    plt.gca().spines['bottom'].set_linewidth(0.7)  # Make the x-axis spine line width 0.7

    # Move the legend outside the plot and remove its outline
    if hue is not None:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=hue, frameon=False)

    # Show the plot
    plt.show()

# Example usage
np.random.seed(10)
data = {
    'Segment': np.repeat(['Segment 1', 'Segment 2', 'Segment 3', 'Segment 4'], 50),
    'Sample Type': np.tile(['Type A', 'Type B'], 100),
    'Value': np.random.normal(size=200)
}
df = pd.DataFrame(data)

create_box_plot(df, x='Segment', y='Value', hue='Sample Type')




import seaborn as sns
import matplotlib.pyplot as plt

def plot_bar_no_hue(data, x, y, title="Sample Composition"):
    # Create a bar plot without hue, using a single color
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x=x, y=y, color='#1DF91')

    # Adding percentage labels to each bar
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt="%.1f%%", label_type='edge', fontsize=10, color='#63666A')

    # Customizing labels and title
    plt.ylabel("Percent", fontsize=10, color="#63666A")
    plt.xlabel(x.replace('_', ' ').title(), fontsize=10, color="#63666A")
    plt.title(title, fontsize=12, color="#63666A")

    # Customizing tick parameters and grid
    plt.xticks(color='#63666A')
    plt.yticks(color='#63666A')
    plt.grid(axis='y', color='#E0E0E1', linestyle='-', linewidth=0.75)
    plt.gca().set_facecolor('white')

    # Styling for the plot spines
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#E0E0E1')
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.show()

# Example of calling the function
# plot_bar_no_hue(data=my_data, x='category_column', y='percentage_column')


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_kde_distribution(data, x_col, title, hue_col=None, palette="coolwarm"):
    plt.figure(figsize=(9, 6))
    sns.set(style="white")
    
    # Plot with or without hue based on the hue_col parameter
    if hue_col:
        sns.kdeplot(
            data=data, x=x_col, hue=hue_col, fill=True, common_norm=False,
            palette=palette, alpha=0.8
        )
    else:
        sns.kdeplot(
            data=data, x=x_col, fill=True, color="#3666AA", alpha=0.8
        )
    
    # Set labels and title
    plt.ylabel("Density", fontsize=10, color="#3666AA")
    plt.xlabel(f"{x_col.replace('_', ' ').title()} Score", fontsize=10, color="#3666AA")
    plt.suptitle(title, fontsize=12, color="#3666AA")

    # Customize plot appearance
    plt.gca().grid(False)
    plt.gca().xaxis.grid(True, color="#E0E0E1", linestyle="--", linewidth=0.75)
    plt.gca().set_facecolor("white")
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    # Customize tick and spine colors
    plt.tick_params(axis='x', colors="#3666AA")
    plt.tick_params(axis='y', colors="#3666AA")

    # Customize legend
    if hue_col:
        legend = plt.legend()
        for text in legend.get_texts():
            text.set_color("#3666AA")

    plt.tight_layout()
    plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_facet_bar_plot(data, hex_colors, x="Subcategory", y="Value", col="Category", 
                          height=4, title="Facet Bar Plot with Custom Colors"):
    """
    Create a facet bar plot with unique colors for each category and horizontal gridlines.
    Also, removes ticks' dashes and axis lines.

    Parameters:
    - data: DataFrame containing the data.
    - hex_colors: List of hex color codes for the categories.
    - x: The name of the column to use for the x-axis.
    - y: The name of the column to use for the y-axis.
    - col: The name of the column to facet the grid by.
    - height: Height of each facet plot.
    - title: Title of the plot.
    """
    # Map each unique category to a color, cycling through hex colors if needed
    unique_categories = data[col].unique()
    category_colors = {cat: hex_colors[i % len(hex_colors)] for i, cat in enumerate(unique_categories)}

    # Initialize the FacetGrid
    g = sns.FacetGrid(data, col=col, height=height, aspect=0.7)

    # Plot each bar with the assigned color for each Category
    for idx, ax in enumerate(g.axes.flat):
        category = unique_categories[idx]  # Get the Category for the current facet
        color = category_colors[category]  # Retrieve the color for the current Category
        sns.barplot(
            data=data[data[col] == category],
            x=x, y=y, color=color, ax=ax
        )
        # Send gridlines to the background
        ax.set_axisbelow(True)

        # Display only horizontal gridlines (on the y-axis)
        ax.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5)

        # Set font properties for x and y ticks
        ax.tick_params(axis='x', labelrotation=45, labelbottom=True, length=0)  # Remove x ticks' dashes
        ax.tick_params(axis='y', length=0)  # Remove y ticks' dashes

        # Remove the axis lines (spines)
        ax.spines['top'].set_visible(False)  # Hide the top axis line
        ax.spines['right'].set_visible(False)  # Hide the right axis line
        ax.spines['left'].set_visible(False)  # Hide the left axis line
        ax.spines['bottom'].set_visible(False)  # Hide the bottom axis line

    # Set titles and labels
    g.set_titles("Category {col_name}").set_axis_labels(x, y)
    g.fig.suptitle(title, y=1.05)

    plt.show()

# Example Usage
data = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E', 'F', 'F'],
    'Subcategory': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'],
    'Value': [5.5, 3.4, 6, 4, 7, 2, 5, 3, 6.5, 3.5, 7.2, 4.1]
})

# Define a list of six hex color codes
hex_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# Call the function with custom parameters
create_facet_bar_plot(data, hex_colors, 
                       x="Subcategory", 
                       y="Value", 
                       col="Category", 
                       height=4, 
                       title="Custom Facet Bar Plot with Horizontal Gridlines")



import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

def ks_and_auc_by_segment(df, segment_column, true_column, pred_column):
    """
    Computes the Kolmogorov-Smirnov (KS) statistic and AUC by segment and returns the results as a table.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data.
    segment_column (str): The name of the column for segmentation (categorical variable).
    true_column (str): The name of the column containing the true binary labels.
    pred_column (str): The name of the column containing the predicted probabilities.

    Returns:
    pd.DataFrame: A table (DataFrame) with segments, their corresponding KS statistics, and AUC values.
    """
    results = []

    # Group by segment and calculate KS and AUC for each group
    for segment, group in df.groupby(segment_column):
        y_true = group[true_column]
        y_pred = group[pred_column]

        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        # Calculate KS statistic
        ks_statistic = max(tpr - fpr)

        # Calculate AUC
        auc_value = roc_auc_score(y_true, y_pred)

        results.append({'Segment': segment, 'KS Statistic': ks_statistic, 'AUC': auc_value})

    # Create a DataFrame for the results
    results_df = pd.DataFrame(results)

    return results_df

# Example usage
# Sample Data: Replace these with your actual data
data = {
    'y_true': np.random.randint(0, 2, size=1000),  # True labels (0 or 1)
    'y_pred': np.random.rand(1000),  # Predicted probabilities
    'segment': np.random.choice(['A', 'B', 'C'], size=1000)  # Segments for evaluation
}

df = pd.DataFrame(data)

# Generate the table with both KS and AUC metrics by segment
metrics_table = ks_and_auc_by_segment(df, 'segment', 'y_true', 'y_pred')

# Display the table with KS and AUC metrics
print(metrics_table)




import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

def plot_roc_by_category_with_rounded_rect(y_true, y_pred, category_column=None, colors=None):
    """
    Plot ROC curve segmented by categories with a rounded rectangle in the background and custom hex colors.
    
    Parameters:
    - y_true: Array-like, true binary labels
    - y_pred: Array-like, predicted probabilities for the positive class
    - category_column: Array-like or None, category labels for segmentation. If None, plot as a single group.
    - colors: Dictionary, optional mapping of categories to hex color codes for the lines
    
    Returns:
    - None: The function will plot the ROC curves.
    """
    
    # If category_column is None, treat the entire data as one category
    if category_column is None:
        category_column = ['All'] * len(y_true)
    
    # Create a DataFrame from the input arrays
    data = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'category': category_column
    })
    
    # Create custom plots for each category
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create a rounded rectangle background
    rounded_rect = FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.05,rounding_size=0.1", 
        linewidth=2, edgecolor='black', facecolor='#f7f7f7', zorder=0
    )
    ax.add_patch(rounded_rect)  # Add the rounded rectangle

    # Loop through unique categories
    for category in data['category'].unique():
        # Filter data by category
        category_data = data[data['category'] == category]
        
        # Get true labels and predicted probabilities for this category
        y_true_cat = category_data['y_true']
        y_pred_cat = category_data['y_pred']
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_cat, y_pred_cat)
        
        # Calculate AUC (Area Under Curve)
        roc_auc = auc(fpr, tpr)
        
        # Use custom color for each category, if provided
        color = colors.get(category, 'b') if colors else 'b'  # Default to blue if no colors specified
        
        # Plot the ROC curve for this category
        ax.plot(fpr, tpr, lw=2, label=f'Category {category} (AUC = {roc_auc:.2f})', color=color)

    # Add the diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')

    # Customizing the plot
    ax.set_title('Custom ROC Curve by Category with Rounded Rectangle Background')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    # Set the axis limits to fit the circle within the plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Set grid inside the plot
    ax.grid(True, linestyle='--', alpha=0.5)

    # Remove the ticks and spines outside the rounded rectangle
    for _, spine in ax.spines.items():
        spine.set_visible(False)
        
    # Set equal aspect ratio for the plot
    ax.set_aspect('equal')  

    # Move the legend below the graph and remove the border
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10, frameon=False)
    
    # Show the plot
    plt.show()

# Example usage:
# Sample data
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 1])
y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.4, 0.9, 0.3, 0.5, 0.85])

# Categories to segment the data by
category_column = ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']

# Define custom hex colors for categories
category_colors = {
    'A': '#1f77b4',  # Blue
    'B': '#ff7f0e',  # Orange
}

# Call the function to plot ROC curves with a rounded rectangle background and legend below
plot_roc_by_category_with_rounded_rect(y_true, y_pred, category_column, colors=category_colors)


