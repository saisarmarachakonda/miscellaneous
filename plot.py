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



import seaborn as sns
import matplotlib.pyplot as plt

def customized_facetgrid_plot(data, x_col, y_col, facet_col, height=4, aspect=1, colors=None):
    # Create FacetGrid
    g = sns.FacetGrid(data, col=facet_col, height=height, aspect=aspect)
    
    # Apply colors iteratively if a list of colors is provided
    if colors:
        for i, ax in enumerate(g.axes.flat):
            color = colors[i % len(colors)]  # Cycle through colors if there are more facets than colors
            sns.barplot(data=data, x=x_col, y=y_col, ax=ax, color=color)
    else:
        # Default color if no custom colors are provided
        g.map_dataframe(sns.barplot, x=x_col, y=y_col, color="skyblue")
    
    # Loop through each axis in the FacetGrid to customize grids, spines, and remove labels
    for ax in g.axes.flat:
        # Customizing Grid Parameters
        ax.grid(axis='y', color="gray", linestyle="--", linewidth=0.5, visible=True)  # Customize y-axis grid
        ax.grid(axis='x', visible=False)  # Disable x-axis grid
        
        # Customizing the spines
        for spine in ax.spines.values():
            spine.set_visible(False)  # Hide all spines
        ax.spines['bottom'].set_visible(True)  # Enable bottom spine only
        ax.spines['bottom'].set_color("#63666A")  # Customize color of bottom spine
        ax.spines['bottom'].set_linewidth(0.7)  # Set linewidth of bottom spine
        
        # Remove x and y labels
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    # Customize facet titles to show only the value
    for ax, title in zip(g.axes.flat, g.col_names):
        ax.set_title(title)  # Set each title to show only the facet value
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# Example usage with a custom color list:
# colors = ["#FF6347", "#4682B4", "#32CD32"]
# customized_facetgrid_plot(industry_dpd_by_bin, x_col='Twentiles', y_col='dpd30_90d', facet_col='tl_industry_segment', colors=colors)
