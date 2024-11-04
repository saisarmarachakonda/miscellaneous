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
