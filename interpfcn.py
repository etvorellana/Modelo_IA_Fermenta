import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class DataInterpolator:
    def __init__(self, file_path):
        """Initialize the interpolator with data from file."""
        self.file_path = file_path
        self.data = None
        self.column_names = None
        self.header_lines = []
        self.load_data()
    
    def load_data(self):
        """Load data from CSV file, preserving header structure."""
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        # Store the first 3 header lines exactly as they are
        self.header_lines = lines[:3]
        
        # Parse data from line 4 onwards (index 3)
        data_lines = lines[3:]
        
        # Parse the data
        data = []
        for line in data_lines:
            if line.strip():
                # Remove quotes and split by spaces
                clean_line = line.strip().strip('"')
                parts = clean_line.split()
                if len(parts) >= 8:  # We expect at least 8 columns
                    data.append([float(x) for x in parts])
        
        self.data = np.array(data)
        self.column_names = ['TimeStep', 'flow-time', 'delta-time', 
                            'iters-per-timestep', 'mon_x', 'mon_sacarose', 
                            'mon_glicose', 'mon_etanol']
        
        if len(self.data) == 0:
            raise ValueError("No data found in file")
        
        print(f"Loaded {len(self.data)} data points")
        print(f"Header lines preserved: {len(self.header_lines)} lines")
    
    def interpolate(self, N, method='linear'):
        """
        Interpolate N points between each consecutive data line.
        
        Parameters:
        N: int - number of interpolation points between each pair
        method: str - interpolation method ('linear', 'cubic', 'quadratic')
        
        Returns:
        DataFrame with interpolated data
        """
        interpolated_rows = []
        
        for i in range(len(self.data) - 1):
            point1 = self.data[i]
            point2 = self.data[i + 1]
            
            # Add the first point
            interpolated_rows.append(point1.copy())
            
            # Interpolate N points
            for j in range(1, N + 1):
                t = j / (N + 1)
                
                if method == 'linear':
                    interpolated_point = point1 + t * (point2 - point1)
                elif method in ['cubic', 'quadratic']:
                    # For higher-order interpolation, use scipy
                    times = np.array([0, 1])
                    values = np.column_stack([point1, point2])
                    f = interp1d(times, values, kind=method, axis=1)
                    interpolated_point = f(t)[0]
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                interpolated_rows.append(interpolated_point)
            
            # Add the last point
            if i == len(self.data) - 2:
                interpolated_rows.append(point2.copy())
        
        return pd.DataFrame(interpolated_rows, columns=self.column_names)
    
    def plot_comparison(self, original_df, interpolated_df, column='mon_sacarose'):
        """Plot original vs interpolated data for comparison."""
        plt.figure(figsize=(12, 6))
        
        # Plot original data
        plt.scatter(original_df['flow-time'], original_df[column], 
                   color='red', label='Original', s=50, zorder=5)
        
        # Plot interpolated data
        plt.plot(interpolated_df['flow-time'], interpolated_df[column], 
                'b-', label='Interpolated', alpha=0.7)
        
        plt.xlabel('Flow Time')
        plt.ylabel(column)
        plt.title(f'Original vs Interpolated Data ({column})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def save_data(self, df, output_file):
        """
        Save interpolated data with EXACTLY the same format as input file.
        Preserves header structure and data format.
        """
        with open(output_file, 'w') as f:
            # Write the header lines exactly as they were in the input file
            for header_line in self.header_lines:
                f.write(header_line)
            
            # Write the data with the same format
            for _, row in df.iterrows():
                # Format each value with the same precision as input
                row_str = ' '.join([f'{val:.15g}' for val in row])
                # Wrap in quotes like the input file
                f.write(f'"{row_str}"\n')
        
        print(f"Data saved to: {output_file}")
        print(f"Format preserved: {len(self.header_lines)} header lines + {len(df)} data lines")

def interpolate_data_simple(file_path, N, method='linear'):
    """
    Simple function to interpolate data from CSV file.
    
    Parameters:
    file_path: str - path to the data file
    N: int - number of interpolation points between each pair of lines
    method: str - interpolation method ('linear', 'cubic', 'quadratic')
    
    Returns:
    interpolated_df: DataFrame with interpolated data
    """
    interpolator = DataInterpolator(file_path)
    interpolated_df = interpolator.interpolate(N, method)
    return interpolated_df, interpolator.column_names

def main():
    # Example usage
    file_path = 'report-file-1.csv'
    N = 20  # Number of interpolation points between each pair of lines
    
    try:
        # Create interpolator
        interpolator = DataInterpolator(file_path)
        
        # Get original data as DataFrame
        original_df = pd.DataFrame(interpolator.data, columns=interpolator.column_names)
        
        print("\n" + "="*60)
        print("INTERPOLATION SUMMARY")
        print("="*60)
        print(f"Original data points: {len(original_df)}")
        print(f"Interpolation points between each pair: {N}")
        print(f"Total interpolated points: {len(original_df) + (len(original_df)-1) * N}")
        
        # Perform interpolation
        interpolated_df = interpolator.interpolate(N, method='linear')
        #interpolated_df = interpolator.interpolate(N, method='cubic',bc_type = 'natural')
        
        print(f"\nInterpolated data points: {len(interpolated_df)}")
        print(f"Time range: {interpolated_df['flow-time'].min():.2f} to {interpolated_df['flow-time'].max():.2f}")
        
        # Show sample of interpolated data
        print("\n" + "="*60)
        print("SAMPLE OF INTERPOLATED DATA (first 10 rows)")
        print("="*60)
        sample_cols = ['flow-time', 'mon_sacarose', 'mon_glicose', 'mon_etanol']
        print(interpolated_df[sample_cols].head(10).to_string(index=False))
        
        # Save results with original format
        output_file = f'interpolated_N{N}_data.csv'
        interpolator.save_data(interpolated_df, output_file)
        
        # Also save as clean CSV for easy viewing in other applications
        clean_output = f'interpolated_N{N}_data_clean.csv'
        interpolated_df.to_csv(clean_output, index=False)
        print(f"Clean CSV saved to: {clean_output}")
        
        # Verify the output file format
        print("\n" + "="*60)
        print("VERIFYING OUTPUT FILE FORMAT")
        print("="*60)
        with open(output_file, 'r') as f:
            first_lines = f.readlines()[:5]
            print("First 5 lines of output file:")
            for i, line in enumerate(first_lines):
                print(f"Line {i+1}: {line.strip()}")
        
        # Optional: Plot comparison
        # interpolator.plot_comparison(original_df, interpolated_df, 'mon_sacarose')
        
    except Exception as e:
        print(f"Error: {e}")

def main_with_custom_parameters():
    """
    Example with custom parameters - demonstrates flexibility
    """
    file_path = 'report-file-1.csv.csv'
    
    # Different interpolation scenarios
    scenarios = [
        {'N': 3, 'method': 'linear', 'description': 'Linear interpolation with 3 points'},
        {'N': 10, 'method': 'cubic', 'description': 'Cubic interpolation with 10 points'},
    ]
    
    for scenario in scenarios:
        print("\n" + "="*60)
        print(f"SCENARIO: {scenario['description']}")
        print("="*60)
        
        try:
            interpolator = DataInterpolator(file_path)
            interpolated_df = interpolator.interpolate(scenario['N'], method=scenario['method'])
            
            # Save with descriptive filename
            output_file = f'interpolated_N{scenario["N"]}_{scenario["method"]}.csv'
            interpolator.save_data(interpolated_df, output_file)
            
            print(f"Generated {len(interpolated_df)} points")
            
        except Exception as e:
            print(f"Error in scenario {scenario['description']}: {e}")

if __name__ == "__main__":
    # Run the main example
    main()
    
    # Uncomment to run with custom parameters
    # main_with_custom_parameters()
