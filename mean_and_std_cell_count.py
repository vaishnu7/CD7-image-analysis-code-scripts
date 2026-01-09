import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_file_statistics(directories):
    """Get mean and std dev for each Excel file separately."""
    file_means = []
    file_stds = []
    file_names = []
    
    for directory in directories:
        print(f"\nSearching: {directory}")
        
        if not os.path.exists(directory):
            print(f"  ✗ Directory not found")
            continue
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.xlsx', '.xls')):
                    file_path = os.path.join(root, file)
                    print(f"  Found: {file}")
                    
                    try:
                        df = pd.read_excel(file_path)
                        if 'total' in df.columns:
                            counts = df['total'].dropna().values
                            
                            file_mean = np.mean(counts)
                            file_std = np.std(counts)
                            
                            file_means.append(file_mean)
                            file_stds.append(file_std)
                            file_names.append(file)
                            
                            print(f"    Loaded {len(counts)} cell counts")
                            print(f"    Mean: {file_mean:.2f}, Std Dev: {file_std:.2f}")
                    except Exception as e:
                        print(f"    Error reading file: {e}")
    
    return file_means, file_stds, file_names


def main():
    print("\n" + "="*60)
    print("CELL COUNT ANALYSIS - Mean and Standard Deviation")
    print("="*60)
    
    directories = [
        r'C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\12dec25_data\output\P9\cell_counts',
        r'C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\12dec25_data\output\P23\cell_counts',
    ]
    
    print("\nLoading cell count data...")
    file_means, file_stds, file_names = get_file_statistics(directories)
    
    if len(file_means) == 0:
        print("\n✗ No cell count data found!")
        return
    
    # Calculate average of means and average of std devs
    mean = np.mean(file_means)
    std_dev = np.mean(file_stds)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Number of files: {len(file_means)}")
    print(f"Mean of means: {mean:.2f}")
    print(f"Mean of std devs: {std_dev:.2f}")
    print("\nPer-file statistics:")
    for fname, fmean, fstd in zip(file_names, file_means, file_stds):
        print(f"  {fname}: mean={fmean:.2f}, std={fstd:.2f}")
    print("="*60)
    
    # Create plot
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Mean', 'Std Dev']
    values = [mean, std_dev]
    colors = ['steelblue', 'coral']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Cell Count Analysis - Mean and Standard Deviation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.getcwd(), 'cell_count_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")