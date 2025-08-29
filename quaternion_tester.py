import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from itertools import permutations, product
from scipy.spatial.transform import Rotation as R


class QuaternionArrangementTester:
    def __init__(self, csv_path):
        self.load_data(csv_path)
        self.create_train_test_split()

    def load_data(self, csv_path):

        self.df = pd.read_csv(csv_path)
        self.pos_fp = self.df[['used_fp_x', 'used_fp_y', 'used_fp_z']].values
        self.pos_unity = self.df[['used_unity_x', 'used_unity_y', 'used_unity_z']].values
        self.quat_fp = self.df[['used_fp_qx', 'used_fp_qy', 'used_fp_qz', 'used_fp_qw']].values
        self.quat_unity = self.df[['used_unity_qx', 'used_unity_qy', 'used_unity_qz', 'used_unity_qw']].values

    def create_train_test_split(self):
        test_indices = np.arange(0, len(self.pos_fp), 3)
        train_indices = np.array([i for i in range(len(self.pos_fp)) if i not in test_indices])

        self.train_pos_fp = self.pos_fp[train_indices]
        self.train_pos_unity = self.pos_unity[train_indices]
        self.train_quat_fp = self.quat_fp[train_indices]
        self.train_quat_unity = self.quat_unity[train_indices]

        self.test_pos_fp = self.pos_fp[test_indices]
        self.test_pos_unity = self.pos_unity[test_indices]
        self.test_quat_fp = self.quat_fp[test_indices]
        self.test_quat_unity = self.quat_unity[test_indices]
        
        print(f"Train: {len(self.train_pos_fp)}, Test: {len(self.test_pos_fp)}")  

    def generate_quaternion_arrangements(self):
        arrangements = []
        
        # Different component orders (qx,qy,qz,qw permutations)
        component_orders = list(permutations([0, 1, 2, 3]))
        # Sign flips
        sign_flips = list(product([-1, 1], repeat=4))

        components = ['qx', 'qy', 'qz', 'qw']

        print(f"{len(component_orders)} orders and {len(sign_flips)} sign flips")
        print(f"Total arrangements: {len(component_orders) * len(sign_flips)}")
 
        for order in component_orders:
            for signs in sign_flips:
                order_name = ','.join([components[i] for i in order])
                signs_name = ','.join(['+' if sign > 0 else '-' for sign in signs])
                arrangements.append({
                    'order': list(order),
                    'signs': list(signs),
                    'name': f"[{order_name}]_signs{signs_name}"
                })
        
        return arrangements
    
    def apply_arrangement(self, quaternions, arrangement):
        reordered = quaternions[:, arrangement['order']]
        signs = np.array(arrangement['signs'])
        return reordered * signs

    def linear_transform(self, train_pos_fp, train_quat_fp, train_pos_unity, train_quat_unity):

        train_fp = train_pos_fp
        train_unity = train_pos_unity

        # Add bias column for affine transformation
        X = np.hstack([train_fp, np.ones((train_fp.shape[0], 1))]) # (Nx4) [x,y,z,1]
        
        try:
            W = np.linalg.lstsq(X, train_unity, rcond=None)[0] # (4x3)
            A = W[:3, :].T # 3x3 rotation/scale
            t = W[3, :] # translation
            return {'A': A, 't': t, 'W': W, 'method': 'linear'}
        except np.linalg.LinAlgError as e:
            print(f"Linear transform error: {e}")

    def rigid_body_transform(self, train_pos_fp, train_quat_fp, train_pos_unity, train_quat_unity):

        train_fp = train_pos_fp
        train_unity = train_pos_unity

        # Center the data
        fp_mean = np.mean(train_fp, axis=0)
        unity_mean = np.mean(train_unity, axis=0)
        fp_centered = train_fp - fp_mean
        unity_centered = train_unity - unity_mean
        
        # Cross-covariance matrix 
        H = fp_centered.T @ unity_centered # 3x3
        U, S, Vt = np.linalg.svd(H)
        R_matrix = Vt.T @ U.T # Rotation matrix
        
        # Ensure proper rotation (for 4D this is approximate)
        if np.linalg.det(R_matrix) < 0:
            Vt[-1, :] *= -1
            R_matrix = Vt.T @ U.T
            
        t = unity_mean - R_matrix @ fp_mean # Translation vector
        return {'R': R_matrix, 't': t, 'method': 'rigid_body'}

    def random_forest(self, train_pos_fp, train_quat_fp, train_pos_unity, train_quat_unity):
        train_fp = np.hstack([train_pos_fp, train_quat_fp])
        train_unity = np.hstack([train_pos_unity, train_quat_unity])
        
        regressors = []
        for i in range(train_unity.shape[1]):  # One per quaternion component
            rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            rf.fit(train_fp, train_unity[:, i])
            regressors.append(rf)
        
        return {'regressors': regressors, 'method': 'random_forest'}

    def transform_quaternion(fpose_quat, R_coord):
        # quat → rotation matrix → transform → quat
        rot_matrix = quaternion_to_rotation_matrix(fpose_quat)
        transformed_matrix = R_coord @ rot_matrix @ R_coord.T
        return rotation_matrix_to_quaternion(transformed_matrix)

    def apply_transform(self, test_pos_fp, test_quat_fp, transform_params):
        method = transform_params['method' ]
        use_rotation = transform_params.get('use_rotation', True)
        
        if method == 'linear':
            X_test = np.hstack([test_pos_fp, np.ones((test_pos_fp.shape[0], 1))]) # (Nx4)
            pred_pos = X_test @ transform_params['W']

            pred_quat = []
            R_mat = transform_params['A'] # 3x3
            for q in test_quat_fp:
                orientation = R.from_quat(q).as_matrix() # quat → 3x3
                transformed_rot_mat = R_mat @ orientation @ R_mat.T # 3x3 (transform)
                transformed_q = R.from_matrix(transformed_rot_mat).as_quat() # back to quat
                pred_quat.append(transformed_q)

            pred_quat = np.array(pred_quat)
            pred = np.hstack([pred_pos, pred_quat])
        
        elif method == 'rigid_body':
            pred_pos = (transform_params['R'] @ test_pos_fp.T).T + transform_params['t'] # (Nx3)
            pred_quat = []
            for q in test_quat_fp:
                orientation = R.from_quat(q).as_matrix()
                transformed_rot_mat = transform_params['R'] @ orientation @ transform_params['R'].T
                transformed_q = R.from_matrix(transformed_rot_mat).as_quat()
                pred_quat.append(transformed_q)

            pred_quat = np.array(pred_quat)
            pred = np.hstack([pred_pos, pred_quat])

        elif method == 'random_forest':
            test_fp = np.hstack([test_pos_fp, test_quat_fp])
            pred = np.column_stack([rf.predict(test_fp) for rf in transform_params['regressors']])

            quat_norms = np.linalg.norm(pred[:, 3:], axis=1, keepdims=True)
            pred[:, 3:] /= quat_norms

        return pred

    def evaluate_all_combinations(self):
        print("Generating quaternion arrangements...")
        arrangements = self.generate_quaternion_arrangements()
        
        methods = {
            'Linear Transform': self.linear_transform,
            'Rigid Body': self.rigid_body_transform,
            'Random Forest': self.random_forest
        }
        
        results = []
        
        print(f"\nTesting {len(arrangements)} arrangements × {len(methods)} methods = {len(arrangements) * len(methods)} combinations...")
        
        for i, arrangement in enumerate(arrangements):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(arrangements)} arrangements tested")
                
            # Apply arrangement to FPose quaternions
            arranged_train_quat_fp = self.apply_arrangement(self.train_quat_fp, arrangement)
            arranged_test_quat_fp = self.apply_arrangement(self.test_quat_fp, arrangement)
            
            for method_name, method_func in methods.items():
                try:
                    # Train transformation
                    transform_params = method_func(self.train_pos_fp, arranged_train_quat_fp, self.train_pos_unity, self.train_quat_unity)
                    
                    # Apply to test data
                    pred_poses = self.apply_transform(
                        self.test_pos_fp, arranged_test_quat_fp, transform_params
                    )
                    # Calculate errors
                    pos_errors = np.linalg.norm(self.test_pos_unity - pred_poses[:, :3], axis=1)
                    angular_errors = quaternion_angular_distance(self.test_quat_unity, pred_poses[:, 3:])

                    combined_error = np.mean(pos_errors) + 0.1 * np.mean(angular_errors) # weighted rotation

                    results.append({
                        'arrangement': arrangement['name'],
                        'method': method_name,
                        'mean_position_error': np.mean(pos_errors),
                        'mean_angular_error': np.mean(angular_errors),
                        'combined_error': combined_error,
                        'max_position_error': np.max(pos_errors),
                        'max_angular_error': np.max(angular_errors),
                        'order': arrangement['order'],
                        'signs': arrangement['signs']
                    })
                    
                except Exception as e:
                    print(f"Error with {arrangement['name']} + {method_name}: {e}")
                    continue
        
        return pd.DataFrame(results)
    
    def analyze_results(self, results_df):
        print("\n" + "="*80)
        print("QUATERNION ARRANGEMENT & TRANSFORM EVALUATION RESULTS")
        print("="*80)

        # Sort by combined error
        best_results = results_df.sort_values('combined_error').head(15) #change for top # amount
        
        print("\nTop # Best Combinations (Position + Angular Error):")
        display_cols = ['arrangement', 'method', 'mean_position_error', 'mean_angular_error', 'combined_error']
        print(best_results[display_cols].to_string(index=False, float_format='%.3f'))

        # Best per method
        print("\n" + "="*60)
        print("BEST RESULT PER METHOD:")
        print("="*60)
        
        for method in results_df['method'].unique():
            method_best = results_df[results_df['method'] == method].sort_values('combined_error').iloc[0]
            print(f"\n{method}:")
            print(f"  Best arrangement: {method_best['arrangement']}")
            print(f"  Mean position error: {method_best['mean_position_error']:.3f}")
            print(f"  Mean angular error: {method_best['mean_angular_error']:.2f}°")
            print(f"  Combined error: {method_best['combined_error']:.3f}")
            print(f"  Component order: {method_best['order']}")
            print(f"  Sign flips: {method_best['signs']}")
        
        return best_results
    
    # def plot_results(self, results_df):
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
    #     # Plot 1: Method comparison (box plot)
    #     methods = results_df['method'].unique()
    #     method_errors = [results_df[results_df['method'] == method]['mean_angular_error'].values for method in methods]
        
    #     ax1.boxplot(method_errors, labels=methods)
    #     ax1.set_title('Angular Error Distribution by Method')
    #     ax1.set_ylabel('Mean Angular Error (degrees)')
    #     ax1.grid(True, alpha=0.3)
        
    #     # Plot 2: Best results comparison
    #     best_per_method = results_df.groupby('method')['mean_angular_error'].min()
    #     bars = ax2.bar(best_per_method.index, best_per_method.values)
        
    #     # Highlight the overall best
    #     best_idx = best_per_method.values.argmin()
    #     bars[best_idx].set_color('red')
        
    #     ax2.set_title('Best Result Per Method')
    #     ax2.set_ylabel('Mean Angular Error (degrees)')
    #     ax2.grid(True, alpha=0.3)
        
    #     plt.tight_layout()
    #     plt.savefig('quaternion_arrangement_results.png', dpi=150, bbox_inches='tight')
    #     plt.show()

def quaternion_angular_distance(q1, q2):
    q1_norm = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
    q2_norm = q2 / np.linalg.norm(q2, axis=1, keepdims=True)
    
    dot_products = np.abs(np.sum(q1_norm * q2_norm, axis=1))
    dot_products = np.clip(dot_products, -1, 1)
    
    angular_distances = 2 * np.arccos(dot_products) * 180 / np.pi
    return angular_distances

def main():
    # Initialize tester
    tester = QuaternionArrangementTester('calibration_data.csv')
    
    # Run all combinations
    print("Starting comprehensive quaternion arrangement testing...")
    results_df = tester.evaluate_all_combinations()
    
    # Analyze results
    best_results = tester.analyze_results(results_df)
    
    ## Plot results
    # tester.plot_results(results_df)
    
    # Save results
    results_df.to_csv('quaternion_arrangement_results.csv', index=False)
    best_results.to_csv('best_quaternion_arrangements.csv', index=False)
    
    print(f"\nResults saved")
    
    return results_df, best_results

if __name__ == "__main__":
    try:
        results_df, best_results = main()
        
    except Exception as e:
        print(f"Error: {e}")
