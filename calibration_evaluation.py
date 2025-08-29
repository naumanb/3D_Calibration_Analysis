import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")

class PoseCalibrationEvaluator:
    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)
        self.prepare_data()
        
    def prepare_data(self):
        self.fp_positions = self.df[['used_fp_x', 'used_fp_y', 'used_fp_z']].values
        self.unity_positions = self.df[['used_unity_x', 'used_unity_y', 'used_unity_z']].values
        
        self.fp_quaternions = self.df[['used_fp_qx', 'used_fp_qy', 'used_fp_qz', 'used_fp_qw']].values
        self.unity_quaternions = self.df[['used_unity_qx', 'used_unity_qy', 'used_unity_qz', 'used_unity_qw']].values
        
        self.fp_timestamps = self.df['timestamp_fp_ms'].values
        self.unity_timestamps = self.df['timestamp_unity_ms'].values
        self.timestamp_diffs = self.df['timestamp_diff_ms'].values
        
        ## self.relative_time = (self.fp_timestamps - self.fp_timestamps[0])
        self.relative_time = np.arange(len(self.fp_positions), dtype=float)
        
        print(f"Loaded {len(self.fp_positions)} valid data points")
        print(f"Time span: {self.relative_time[-1]:.1f} seconds")

class PositionCalibrationMethods:
    @staticmethod
    def linear_transformation(train_fp, train_unity):

        X = np.hstack([train_fp, np.ones((train_fp.shape[0], 1))])
        transformation_matrix = np.linalg.lstsq(X, train_unity, rcond=None)[0]
        A = transformation_matrix[:3, :].T  # 3x3 transformation matrix
        b = transformation_matrix[3, :]     # 3x1 translation vector

        return {'A': A, 'b': b, 'method': 'linear'}
    
    @staticmethod
    def rigid_body_transformation(train_fp, train_unity):
        fp_centered = train_fp - np.mean(train_fp, axis=0)
        unity_centered = train_unity - np.mean(train_unity, axis=0)
        
        H = fp_centered.T @ unity_centered
        
        U, S, Vt = np.linalg.svd(H)
        R_matrix = Vt.T @ U.T
        
        if np.linalg.det(R_matrix) < 0:
            Vt[-1, :] *= -1
            R_matrix = Vt.T @ U.T
        
        # Compute translation
        t = np.mean(train_unity, axis=0) - R_matrix @ np.mean(train_fp, axis=0)
        
        return {'R': R_matrix, 't': t, 'method': 'rigid_body'}
    
    @staticmethod
    def svd_based_transformation(train_fp, train_unity):
        # Center the data
        fp_mean = np.mean(train_fp, axis=0)
        unity_mean = np.mean(train_unity, axis=0)
        
        fp_centered = train_fp - fp_mean
        unity_centered = train_unity - unity_mean
        
        # Cross-covariance matrix
        H = fp_centered.T @ unity_centered
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # Optimal rotation matrix
        R_matrix = Vt.T @ U.T
        
        # Correct improper rotation
        if np.linalg.det(R_matrix) < 0:
            Vt[-1, :] *= -1
            R_matrix = Vt.T @ U.T
        
        # Translation vector
        t = unity_mean - R_matrix @ fp_mean
        
        return {'R': R_matrix, 't': t, 'method': 'svd_based'}
    
    @staticmethod
    def opencv_affine_transformation(train_fp, train_unity):    
        try:
            src = train_fp.astype(np.float64)
            dst = train_unity.astype(np.float64)
            
            retval, affine_matrix, inliers = cv2.estimateAffine3D(src, dst)
            
            if retval:
                return {'affine_matrix': affine_matrix, 'inliers': inliers, 'method': 'opencv_affine'}
            else:
                # Fallback if OpenCV method fails
                return PositionCalibrationMethods.rigid_body_transformation(train_fp, train_unity)
        except Exception as e:
            print(f"OpenCV method failed: {e}, using fallback")
            return PositionCalibrationMethods.rigid_body_transformation(train_fp, train_unity)
    
    @staticmethod
    def polynomial_transformation(train_fp, train_unity, degree=2):
        poly_features = PolynomialFeatures(degree=degree, include_bias=True)
        fp_poly = poly_features.fit_transform(train_fp)
        
        reg = LinearRegression(fit_intercept=False)
        reg.fit(fp_poly, train_unity)
        
        return {'poly_features': poly_features, 'regressor': reg, 'method': 'polynomial'}
    
    @staticmethod
    def random_forest_transformation(train_fp, train_unity):
        rf_regressors = []
        for i in range(3):  # One regressor per coordinate
            rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            rf.fit(train_fp, train_unity[:, i])
            rf_regressors.append(rf)
        
        return {'regressors': rf_regressors, 'method': 'random_forest'}

def apply_transformation(test_fp, calibration_params):
    method = calibration_params['method']
    
    if method == 'linear':
        A, b = calibration_params['A'], calibration_params['b']
        return (A @ test_fp.T).T + b
    
    elif method in ['rigid_body', 'svd_based']:
        R_matrix, t = calibration_params['R'], calibration_params['t']
        return (R_matrix @ test_fp.T).T + t
    
    elif method == 'opencv_affine':
        if 'affine_matrix' in calibration_params:
            affine_matrix = calibration_params['affine_matrix']
            # Apply 3x4 affine transformation
            ones = np.ones((test_fp.shape[0], 1))
            homogeneous = np.hstack([test_fp, ones])
            return (affine_matrix @ homogeneous.T).T
        else:
            # Fallback case
            R_matrix, t = calibration_params['R'], calibration_params['t']
            return (R_matrix @ test_fp.T).T + t
    
    elif method == 'polynomial':
        poly_features = calibration_params['poly_features']
        regressor = calibration_params['regressor']
        test_fp_poly = poly_features.transform(test_fp)
        return regressor.predict(test_fp_poly)
    
    elif method == 'random_forest':
        regressors = calibration_params['regressors']
        predictions = np.zeros((test_fp.shape[0], 3))
        for i, rf in enumerate(regressors):
            predictions[:, i] = rf.predict(test_fp)
        return predictions

def evaluate_calibration_methods(evaluator):

    fp_pos = evaluator.fp_positions
    unity_pos = evaluator.unity_positions
    
    # Split data into train/test (70/30 split) but maintaing order for time-series
    test_indices = np.arange(0, len(fp_pos), 3)  # Every 3rd point: 0, 3, 6, 9, ...
    train_indices = np.array([i for i in range(len(fp_pos)) if i not in test_indices])
    
    # Split the data using these indices
    train_fp = fp_pos[train_indices]
    train_unity = unity_pos[train_indices]
    test_fp = fp_pos[test_indices]
    test_unity = unity_pos[test_indices]

    print(f"Training set: {len(train_fp)} samples")
    print(f"Test set: {len(test_fp)} samples")
    
    methods = { 
        'Linear Transformation': PositionCalibrationMethods.linear_transformation,
        'Rigid Body (Kabsch)': PositionCalibrationMethods.rigid_body_transformation,
        'SVD-Based': PositionCalibrationMethods.svd_based_transformation,
        'OpenCV Affine3D': PositionCalibrationMethods.opencv_affine_transformation,
        'Polynomial (Degree 2)': lambda x, y: PositionCalibrationMethods.polynomial_transformation(x, y, degree=2),
        'Random Forest': PositionCalibrationMethods.random_forest_transformation
    }
    
    results = {}
    predictions = {}
    
    for method_name, method_func in methods.items():
        print(f"\nEvaluating {method_name}...")
        
        try:
            # train calibration
            calibration_params = method_func(train_fp, train_unity)
            
            # predict on test set
            pred_unity = apply_transformation(test_fp, calibration_params)
            predictions[method_name] = pred_unity
            
            # calculate metrics
            rmse = np.sqrt(mean_squared_error(test_unity, pred_unity))
            mae = mean_absolute_error(test_unity, pred_unity)
            
            rmse_x = np.sqrt(mean_squared_error(test_unity[:, 0], pred_unity[:, 0]))
            rmse_y = np.sqrt(mean_squared_error(test_unity[:, 1], pred_unity[:, 1]))
            rmse_z = np.sqrt(mean_squared_error(test_unity[:, 2], pred_unity[:, 2]))
            
            # calculate maximum error
            max_error = np.max(np.linalg.norm(test_unity - pred_unity, axis=1))
            
            results[method_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'RMSE_X': rmse_x,
                'RMSE_Y': rmse_y,
                'RMSE_Z': rmse_z,
                'Max_Error': max_error,
                'predictions': pred_unity
            }
            
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE:  {mae:.6f}")
            print(f"  Max Error: {max_error:.6f}")
            print(f"  Per-axis RMSE - X: {rmse_x:.6f}, Y: {rmse_y:.6f}, Z: {rmse_z:.6f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    return results, test_unity, test_fp, train_fp, train_unity, test_indices

def create_evaluation_report(results):
    report_data = []
    for method, metrics in results.items():
        report_data.append({
            'Method': method,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'Max_Error': metrics['Max_Error'],
            'RMSE_X': metrics['RMSE_X'],
            'RMSE_Y': metrics['RMSE_Y'],
            'RMSE_Z': metrics['RMSE_Z']
        })
    
    df_report = pd.DataFrame(report_data)
    df_report = df_report.sort_values('RMSE')
    
    print("\n" + "="*80)
    print("CALIBRATION METHODS EVALUATION REPORT")
    print("="*80)
    print(df_report.to_string(index=False, float_format='%.6f'))
    print("="*80)
    
    return df_report

def analyze_data_characteristics(evaluator):

    print("\n" + "="*60)
    print("DATA CHARACTERISTICS ANALYSIS")
    print("="*60)
    
    fp_pos = evaluator.fp_positions
    unity_pos = evaluator.unity_positions
    
    # Position statistics
    print("FPose Position Statistics:")
    print(f"  X: [{fp_pos[:,0].min():.3f}, {fp_pos[:,0].max():.3f}], std: {fp_pos[:,0].std():.3f}")
    print(f"  Y: [{fp_pos[:,1].min():.3f}, {fp_pos[:,1].max():.3f}], std: {fp_pos[:,1].std():.3f}")
    print(f"  Z: [{fp_pos[:,2].min():.3f}, {fp_pos[:,2].max():.3f}], std: {fp_pos[:,2].std():.3f}")
    
    print("\nUnity Position Statistics:")
    print(f"  X: [{unity_pos[:,0].min():.3f}, {unity_pos[:,0].max():.3f}], std: {unity_pos[:,0].std():.3f}")
    print(f"  Y: [{unity_pos[:,1].min():.3f}, {unity_pos[:,1].max():.3f}], std: {unity_pos[:,1].std():.3f}")
    print(f"  Z: [{unity_pos[:,2].min():.3f}, {unity_pos[:,2].max():.3f}], std: {unity_pos[:,2].std():.3f}")
    
    # Timestamp analysis
    print(f"\nTiming Analysis:")
    print(f"  Timestamp differences: [{evaluator.timestamp_diffs.min():.0f}, {evaluator.timestamp_diffs.max():.0f}] ms")
    print(f"  Average timestamp diff: {evaluator.timestamp_diffs.mean():.1f} ms")
    
    # Scale analysis
    fp_scale = np.linalg.norm(fp_pos, axis=1).mean()
    unity_scale = np.linalg.norm(unity_pos, axis=1).mean()
    print(f"\nScale Analysis:")
    print(f"  Average FPose position magnitude: {fp_scale:.3f}")
    print(f"  Average Unity position magnitude: {unity_scale:.3f}")
    print(f"  Scale ratio (Unity/FPose): {unity_scale/fp_scale:.3f}")
    
    return fp_scale, unity_scale

def plot_time_series_comparison(evaluator, results, test_indices, test_unity, max_points=500):

    test_times = evaluator.relative_time[test_indices]
    
    if len(test_times) > max_points:
        step = len(test_times) // max_points
        plot_indices = np.arange(0, len(test_times), step)
    else:
        plot_indices = np.arange(len(test_times))
    
    plot_times = test_times[plot_indices]
    plot_unity = test_unity[plot_indices]
    
    n_methods = len(results)
    fig, axes = plt.subplots(3, n_methods, figsize=(5*n_methods, 12))
    
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    colors = ['blue', 'red', 'green']
    axis_labels = ['X', 'Y', 'Z']
    
    for j, (method_name, metrics) in enumerate(results.items()):
        plot_pred = metrics['predictions'][plot_indices]
        
        for i in range(3):  # X, Y, Z axes
            ax = axes[i, j]
            
            # Plot ground truth and predictions
            ax.plot(plot_times, plot_unity[:, i], 'o-', color=colors[i], 
                   alpha=0.7, markersize=3, linewidth=1, label=f'Unity {axis_labels[i]} (Ground Truth)')
            ax.plot(plot_times, plot_pred[:, i], 's--', color='black', 
                   alpha=0.8, markersize=2, linewidth=1, label=f'Predicted {axis_labels[i]}')
            
            # Calculate and display error statistics for this axis
            axis_error = np.abs(plot_unity[:, i] - plot_pred[:, i])
            mean_error = np.mean(axis_error)
            max_error = np.max(axis_error)
            
            ax.set_title(f'{method_name}\n{axis_labels[i]}-axis (RMSE: {metrics[f"RMSE_{axis_labels[i]}"]:.4f})')
            ax.set_xlabel('Time (au)')
            ax.set_ylabel(f'Position {axis_labels[i]} (meters)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add error statistics as text
            ax.text(0.02, 0.98, f'Mean Error: {mean_error:.4f}\nMax Error: {max_error:.4f}', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('time_series_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_error_time_series(evaluator, results, test_indices, test_unity, max_points=500):
    """
    Plot error magnitude over time for all methods
    """
    test_times = evaluator.relative_time[test_indices]
    
    # Subsample for cleaner plots if too many points
    if len(test_times) > max_points:
        step = len(test_times) // max_points
        plot_indices = np.arange(0, len(test_times), step)
    else:
        plot_indices = np.arange(len(test_times))
    
    plot_times = test_times[plot_indices]
    plot_unity = test_unity[plot_indices]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Error magnitude over time
    for method_name, metrics in results.items():
        plot_pred = metrics['predictions'][plot_indices]
        error_magnitude = np.linalg.norm(plot_unity - plot_pred, axis=1)
        
        ax1.plot(plot_times, error_magnitude, 'o-', alpha=0.7, markersize=3, 
                linewidth=1, label=f'{method_name} (RMSE: {metrics["RMSE"]:.4f})')
    
    ax1.set_title('Prediction Error Magnitude Over Time')
    ax1.set_xlabel('Time (au)')
    ax1.set_ylabel('Error Magnitude (meters)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Per-axis error comparison for best method
    best_method = min(results.keys(), key=lambda k: results[k]['RMSE'])
    best_pred = results[best_method]['predictions'][plot_indices]
    
    axis_labels = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']
    
    for i in range(3):
        axis_error = np.abs(plot_unity[:, i] - best_pred[:, i])
        ax2.plot(plot_times, axis_error, 'o-', color=colors[i], alpha=0.7, 
                markersize=3, linewidth=1, label=f'{axis_labels[i]}-axis Error')
    
    ax2.set_title(f'Per-Axis Error Over Time - Best Method ({best_method})')
    ax2.set_xlabel('Time (au)')
    ax2.set_ylabel('Absolute Error (meters)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_time_series.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_trajectory_comparison(evaluator, results, test_indices, test_unity, max_points=300):
    test_times = evaluator.relative_time[test_indices]
    
    if len(test_times) > max_points:
        step = len(test_times) // max_points
        plot_indices = np.arange(0, len(test_times), step)
    else:
        plot_indices = np.arange(len(test_times))
    
    plot_unity = test_unity[plot_indices]
    
    n_methods = len(results)
    fig = plt.figure(figsize=(6*n_methods, 5))
    
    for j, (method_name, metrics) in enumerate(results.items()):
        ax = fig.add_subplot(1, n_methods, j+1, projection='3d')
        plot_pred = metrics['predictions'][plot_indices]
        
        # Plot trajectories
        ax.plot(plot_unity[:, 0], plot_unity[:, 1], plot_unity[:, 2], 
               'b-o', alpha=0.7, markersize=3, linewidth=2, label='Unity Ground Truth')
        ax.plot(plot_pred[:, 0], plot_pred[:, 1], plot_pred[:, 2], 
               'r--s', alpha=0.8, markersize=2, linewidth=1, label='Predicted Path')
        
        # Mark start and end points
        ax.scatter(plot_unity[0, 0], plot_unity[0, 1], plot_unity[0, 2], 
                  c='green', s=100, marker='^', label='Start')
        ax.scatter(plot_unity[-1, 0], plot_unity[-1, 1], plot_unity[-1, 2], 
                  c='orange', s=100, marker='v', label='End')
        
        ax.set_title(f'{method_name}\nRMSE: {metrics["RMSE"]:.4f}')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.legend(fontsize=8)
        
        # Set equal aspect ratio
        max_range = np.array([plot_unity.max()-plot_unity.min()]).max() / 2.0
        mid_x = (plot_unity[:, 0].max()+plot_unity[:, 0].min()) * 0.5
        mid_y = (plot_unity[:, 1].max()+plot_unity[:, 1].min()) * 0.5
        mid_z = (plot_unity[:, 2].max()+plot_unity[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_detailed_error_analysis(evaluator, results, test_indices, test_unity):

    test_times = evaluator.relative_time[test_indices]
    
    # Find best method
    best_method = min(results.keys(), key=lambda k: results[k]['RMSE'])
    best_pred = results[best_method]['predictions']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Error distribution histogram
    error_magnitudes = np.linalg.norm(test_unity - best_pred, axis=1)
    axes[0, 0].hist(error_magnitudes, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(error_magnitudes), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(error_magnitudes):.4f}')
    axes[0, 0].set_title(f'Error Distribution - {best_method}')
    axes[0, 0].set_xlabel('Error Magnitude (meters)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error vs time
    axes[0, 1].plot(test_times, error_magnitudes, 'b-', alpha=0.7)
    axes[0, 1].set_title('Error Magnitude vs Time')
    axes[0, 1].set_xlabel('Time (au)')
    axes[0, 1].set_ylabel('Error Magnitude (meters)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Per-axis error correlation
    for i, (label, color) in enumerate(zip(['X', 'Y', 'Z'], ['red', 'green', 'blue'])):
        axes[0, 2].scatter(test_unity[:, i], best_pred[:, i], alpha=0.6, 
                          s=10, c=color, label=f'{label}-axis')
    
    # Perfect prediction line
    all_vals = np.concatenate([test_unity.flatten(), best_pred.flatten()])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
    axes[0, 2].set_title('Predicted vs Ground Truth')
    axes[0, 2].set_xlabel('Ground Truth (meters)')
    axes[0, 2].set_ylabel('Predicted (meters)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Method comparison bar chart
    methods = list(results.keys())
    rmse_values = [results[m]['RMSE'] for m in methods]
    bars = axes[1, 0].bar(range(len(methods)), rmse_values)
    
    # Highlight best method
    best_idx = rmse_values.index(min(rmse_values))
    bars[best_idx].set_color('red')
    
    axes[1, 0].set_title('Method Comparison (RMSE)')
    axes[1, 0].set_xlabel('Methods')
    axes[1, 0].set_ylabel('RMSE (meters)')
    axes[1, 0].set_xticks(range(len(methods)))
    axes[1, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Residuals plot
    residuals = test_unity - best_pred
    axes[1, 1].scatter(np.arange(len(residuals)), residuals[:, 0], alpha=0.6, s=10, c='red', label='X')
    axes[1, 1].scatter(np.arange(len(residuals)), residuals[:, 1], alpha=0.6, s=10, c='green', label='Y')
    axes[1, 1].scatter(np.arange(len(residuals)), residuals[:, 2], alpha=0.6, s=10, c='blue', label='Z')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.8)
    axes[1, 1].set_title('Residuals Plot')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Residual (meters)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Cumulative error
    cumulative_error = np.cumsum(error_magnitudes) / np.arange(1, len(error_magnitudes)+1)
    axes[1, 2].plot(test_times, cumulative_error, 'b-', linewidth=2)
    axes[1, 2].set_title('Cumulative Mean Error')
    axes[1, 2].set_xlabel('Time (au)')
    axes[1, 2].set_ylabel('Cumulative Mean Error (meters)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_error_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_all_visualizations(evaluator, results, test_indices, test_unity):

    print("\nGenerating time series comparison plots...")
    plot_time_series_comparison(evaluator, results, test_indices, test_unity)
    
    print("Generating error time series plots...")
    plot_error_time_series(evaluator, results, test_indices, test_unity)
    
    print("Generating trajectory comparison plots...")
    plot_trajectory_comparison(evaluator, results, test_indices, test_unity)
    
    print("Generating detailed error analysis...")
    plot_detailed_error_analysis(evaluator, results, test_indices, test_unity)

# Main execution function
def main(csv_path='calibration_data_sample.csv'):

    print("="*80)
    print("FPOSE TO UNITY CALIBRATION EVALUATION")
    print("="*80)
    
    print("Loading and preparing calibration data...")
    evaluator = PoseCalibrationEvaluator(csv_path)
    
    print("Analyzing data characteristics...")
    analyze_data_characteristics(evaluator)
    
    print("\nEvaluating calibration methods...")
    results, test_unity, test_fp, train_fp, train_unity, test_indices = evaluate_calibration_methods(evaluator)
    
    print("\nCreating evaluation report...")
    report_df = create_evaluation_report(results)

    print("Creating time series visualizations...")
    create_all_visualizations(evaluator, results, test_indices, test_unity)
    
    # Save results
    report_df.to_csv('fpose_unity_calibration_report.csv', index=False)
    evaluator.df.to_csv('processed_calibration_data.csv', index=False)
    
    print(f"\nReport saved to 'fpose_unity_calibration_report.csv'")
    print(f"Processed data saved to 'processed_calibration_data.csv'")
    
    # Find best method
    if len(results) > 0:
        best_method = report_df.iloc[0]['Method']
        best_rmse = report_df.iloc[0]['RMSE']
        print(f"\nBest performing method: {best_method} (RMSE: {best_rmse:.6f})")
    
    return evaluator, results, report_df

# Usage example:
if __name__ == "__main__":
    try:
        evaluator, results, report = main('calibration_data.csv')
        print("\nCalibration evaluation completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find CSV file. Please ensure 'calibration_data_sample.csv' is in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
