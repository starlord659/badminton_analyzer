"""
Validation script to check preprocessing quality before training
Run this BEFORE training to catch issues early
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Configuration
PREPROCESSED_DATA_DIR = Path('preprocessed_multimodal_v5_engineered/')
METADATA_DIR = Path('preprocessed_metadata/')
EXPECTED_SEQUENCE_LENGTH = 16
EXPECTED_FEATURES = 82
VALIDATION_RESULTS_DIR = Path('validation_results/')

VALIDATION_RESULTS_DIR.mkdir(exist_ok=True)

class PreprocessingValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.statistics = defaultdict(list)
    
    def validate_shape(self, data_path: Path):
        """Validate data shape matches expectations"""
        try:
            data = np.load(data_path)
            
            if data.shape != (EXPECTED_SEQUENCE_LENGTH, EXPECTED_FEATURES):
                self.errors.append({
                    'file': data_path.name,
                    'error': f'Shape mismatch: {data.shape}, expected ({EXPECTED_SEQUENCE_LENGTH}, {EXPECTED_FEATURES})'
                })
                return False
            
            return True
        except Exception as e:
            self.errors.append({
                'file': data_path.name,
                'error': f'Failed to load: {str(e)}'
            })
            return False
    
    def validate_values(self, data_path: Path):
        """Check for NaN, Inf, and extreme values"""
        try:
            data = np.load(data_path)
            
            # Check for NaN
            nan_count = np.isnan(data).sum()
            if nan_count > 0:
                self.warnings.append({
                    'file': data_path.name,
                    'warning': f'Contains {nan_count} NaN values ({nan_count / data.size * 100:.2f}%)'
                })
            
            # Check for Inf
            inf_count = np.isinf(data).sum()
            if inf_count > 0:
                self.errors.append({
                    'file': data_path.name,
                    'error': f'Contains {inf_count} Inf values'
                })
                return False
            
            # Check for extreme values
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                min_val, max_val = valid_data.min(), valid_data.max()
                
                if min_val < -100 or max_val > 100:
                    self.warnings.append({
                        'file': data_path.name,
                        'warning': f'Extreme values: min={min_val:.2f}, max={max_val:.2f}'
                    })
            
            return True
        except Exception as e:
            self.errors.append({
                'file': data_path.name,
                'error': f'Validation failed: {str(e)}'
            })
            return False
    
    def collect_statistics(self, data_path: Path, class_name: str):
        """Collect statistics for later analysis"""
        try:
            data = np.load(data_path)
            
            self.statistics['class'].append(class_name)
            self.statistics['mean'].append(np.nanmean(data))
            self.statistics['std'].append(np.nanstd(data))
            self.statistics['min'].append(np.nanmin(data))
            self.statistics['max'].append(np.nanmax(data))
            self.statistics['nan_ratio'].append(np.isnan(data).mean())
            
            # Feature-specific stats
            pose_features = data[:, :72]  # 36*2 players
            engineered_features = data[:, 72:78]  # 6 engineered
            shuttle_features = data[:, 78:]  # 4 shuttle
            
            self.statistics['pose_mean'].append(np.nanmean(pose_features))
            self.statistics['engineered_mean'].append(np.nanmean(engineered_features))
            self.statistics['shuttle_mean'].append(np.nanmean(shuttle_features))
            
            # Check for invalid engineered features (-1.0 markers)
            invalid_engineered = (engineered_features == -1.0).mean()
            self.statistics['invalid_engineered_ratio'].append(invalid_engineered)
            
        except Exception as e:
            print(f"Warning: Could not collect stats for {data_path.name}: {e}")
    
    def validate_metadata(self):
        """Validate metadata files exist and are consistent"""
        print("\nValidating metadata...")
        
        if not METADATA_DIR.exists():
            self.warnings.append({
                'file': 'metadata_dir',
                'warning': f'Metadata directory not found: {METADATA_DIR}'
            })
            return
        
        metadata_count = 0
        resolution_stats = defaultdict(int)
        
        for class_dir in METADATA_DIR.iterdir():
            if not class_dir.is_dir():
                continue
            
            for metadata_file in class_dir.glob('*.json'):
                metadata_count += 1
                
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check required fields
                    required_fields = ['frame_width', 'frame_height', 'num_frames', 'fps']
                    missing_fields = [f for f in required_fields if f not in metadata]
                    
                    if missing_fields:
                        self.warnings.append({
                            'file': metadata_file.name,
                            'warning': f'Missing fields: {missing_fields}'
                        })
                    
                    # Track resolutions
                    resolution = f"{metadata.get('frame_width', '?')}x{metadata.get('frame_height', '?')}"
                    resolution_stats[resolution] += 1
                    
                except Exception as e:
                    self.errors.append({
                        'file': metadata_file.name,
                        'error': f'Failed to parse metadata: {str(e)}'
                    })
        
        print(f"  Found {metadata_count} metadata files")
        print(f"  Resolution distribution:")
        for resolution, count in sorted(resolution_stats.items()):
            print(f"    {resolution}: {count} videos")
    
    def generate_report(self):
        """Generate validation report"""
        print("\n" + "="*70)
        print("VALIDATION REPORT")
        print("="*70)
        
        # Errors
        if self.errors:
            print(f"\n❌ ERRORS FOUND: {len(self.errors)}")
            for i, error in enumerate(self.errors[:10], 1):
                print(f"  {i}. {error['file']}: {error['error']}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
        else:
            print("\n✅ NO ERRORS FOUND")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
            for i, warning in enumerate(self.warnings[:10], 1):
                print(f"  {i}. {warning['file']}: {warning['warning']}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
        else:
            print("\n✅ NO WARNINGS")
        
        # Statistics summary
        if self.statistics['class']:
            print(f"\n📊 DATASET STATISTICS")
            print(f"  Total samples: {len(self.statistics['class'])}")
            
            # Class distribution
            class_counts = pd.Series(self.statistics['class']).value_counts()
            print(f"\n  Class distribution:")
            for class_name, count in class_counts.items():
                print(f"    {class_name}: {count} samples")
            
            # Value ranges
            print(f"\n  Value ranges:")
            print(f"    Overall mean: {np.mean(self.statistics['mean']):.4f} ± {np.std(self.statistics['mean']):.4f}")
            print(f"    Overall min: {np.min(self.statistics['min']):.4f}")
            print(f"    Overall max: {np.max(self.statistics['max']):.4f}")
            
            # Feature-specific
            print(f"\n  Feature-specific means:")
            print(f"    Pose features: {np.mean(self.statistics['pose_mean']):.4f}")
            print(f"    Engineered features: {np.mean(self.statistics['engineered_mean']):.4f}")
            print(f"    Shuttle features: {np.mean(self.statistics['shuttle_mean']):.4f}")
            
            # Data quality
            avg_nan_ratio = np.mean(self.statistics['nan_ratio'])
            avg_invalid_eng = np.mean(self.statistics['invalid_engineered_ratio'])
            
            print(f"\n  Data quality:")
            print(f"    Average NaN ratio: {avg_nan_ratio*100:.2f}%")
            print(f"    Average invalid engineered features: {avg_invalid_eng*100:.2f}%")
            
            if avg_invalid_eng > 0.3:
                print(f"    ⚠️  WARNING: High rate of invalid engineered features!")
        
        print("\n" + "="*70)
        
        # Recommendation
        if self.errors:
            print("\n❌ RECOMMENDATION: Fix errors before training!")
            return False
        elif len(self.warnings) > len(self.statistics['class']) * 0.1:  # >10% warnings
            print("\n⚠️  RECOMMENDATION: Review warnings before training")
            return True
        else:
            print("\n✅ RECOMMENDATION: Dataset looks good, proceed with training")
            return True
    
    def plot_statistics(self):
        """Generate visualization plots"""
        if not self.statistics['class']:
            print("No statistics to plot")
            return
        
        print("\nGenerating visualization plots...")
        
        df = pd.DataFrame(self.statistics)
        
        # 1. Class distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        class_counts = df['class'].value_counts()
        axes[0, 0].bar(range(len(class_counts)), class_counts.values)
        axes[0, 0].set_xticks(range(len(class_counts)))
        axes[0, 0].set_xticklabels(class_counts.index, rotation=45, ha='right')
        axes[0, 0].set_title('Class Distribution')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Value ranges by class
        df.boxplot(column='mean', by='class', ax=axes[0, 1])
        axes[0, 1].set_title('Mean Values by Class')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Mean Value')
        plt.sca(axes[0, 1])
        plt.xticks(rotation=45, ha='right')
        
        # 3. Feature type comparison
        feature_means = df[['pose_mean', 'engineered_mean', 'shuttle_mean']].mean()
        axes[1, 0].bar(['Pose', 'Engineered', 'Shuttle'], feature_means.values)
        axes[1, 0].set_title('Average Values by Feature Type')
        axes[1, 0].set_ylabel('Mean Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Data quality metrics
        quality_df = df[['nan_ratio', 'invalid_engineered_ratio']].mean()
        axes[1, 1].bar(['NaN Ratio', 'Invalid Engineered'], quality_df.values)
        axes[1, 1].set_title('Data Quality Metrics')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = VALIDATION_RESULTS_DIR / 'preprocessing_validation.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plots saved to: {save_path}")
        
        # 5. Per-class quality heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        quality_matrix = df.pivot_table(
            values=['nan_ratio', 'invalid_engineered_ratio'],
            index='class',
            aggfunc='mean'
        )
        
        sns.heatmap(quality_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax)
        ax.set_title('Data Quality by Class')
        ax.set_xlabel('Quality Metric')
        ax.set_ylabel('Class')
        
        save_path = VALIDATION_RESULTS_DIR / 'quality_heatmap.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Quality heatmap saved to: {save_path}")

def main():
    print("="*70)
    print("PREPROCESSING VALIDATION")
    print("="*70)
    
    if not PREPROCESSED_DATA_DIR.exists():
        print(f"\n❌ ERROR: Preprocessed data directory not found: {PREPROCESSED_DATA_DIR}")
        print("Please run preprocessing first!")
        return
    
    validator = PreprocessingValidator()
    
    # Validate metadata
    validator.validate_metadata()
    
    # Validate all data files
    print("\nValidating preprocessed data files...")
    
    all_files = list(PREPROCESSED_DATA_DIR.rglob('*.npy'))
    
    if not all_files:
        print(f"\n❌ ERROR: No .npy files found in {PREPROCESSED_DATA_DIR}")
        return
    
    print(f"Found {len(all_files)} files to validate\n")
    
    for data_path in tqdm(all_files, desc="Validating"):
        class_name = data_path.parent.name
        
        if validator.validate_shape(data_path):
            validator.validate_values(data_path)
            validator.collect_statistics(data_path, class_name)
    
    # Generate report and plots
    is_valid = validator.generate_report()
    validator.plot_statistics()
    
    # Save detailed results to JSON
    results = {
        'valid': is_valid,
        'total_files': len(all_files),
        'errors': validator.errors,
        'warnings': validator.warnings
    }
    
    with open(VALIDATION_RESULTS_DIR / 'validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: {VALIDATION_RESULTS_DIR / 'validation_results.json'}")
    
    return is_valid

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)