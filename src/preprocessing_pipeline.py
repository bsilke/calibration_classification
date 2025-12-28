"""
Preprocessing Pipeline with Full Logging

This script runs the complete preprocessing pipeline with logging enabled.
All steps are logged to logs/preprocess.log for reproducibility.

IMPORTANT: Date-Time Feature Engineering
=========================================
The feature `target_gap` represents the difference between End_target_year and
Interim_target_year. This is time-invariant (no reference date needed).

Note: years_to_end_target and years_to_interim_target were removed as they are
redundant with the raw year columns (just shifted by a constant).

Usage:
    python src/preprocessing_pipeline.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.reproducibility import (
    setup_logging, set_random_seeds, load_config, 
    log_dataframe_info, log_transformation, PipelineLogger,
    verify_checksum
)
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================================
# TRAINING REFERENCE DATE
# ============================================================================
# Used for calculating date-time features (years_to_end_target, etc.)
# Update this when retraining the model to maintain temporal relevance
TRAINING_REFERENCE_DATE = "2025-12-20"
TRAINING_REFERENCE_YEAR = 2025
# ============================================================================


def load_and_verify_data(config, logger):
    """Load raw data and verify checksum."""
    with PipelineLogger("Load Raw Data"):
        raw_path = config['data_config']['raw_data_path']
        
        # Verify checksum if available
        checksum_path = f"{raw_path}.sha256"
        if os.path.exists(checksum_path):
            if verify_checksum(raw_path, checksum_path):
                logger.info("Data integrity verified ✓")
            else:
                logger.warning("DATA INTEGRITY CHECK FAILED - proceeding with caution")
        
        # Load data
        df = pd.read_csv(raw_path, sep=';', quotechar='"', quoting=1,
                        engine='python', on_bad_lines='skip',
                        skipinitialspace=True, doublequote=True)
        
        log_dataframe_info(df, "Raw Data")
        
        return df


def filter_to_companies(df, logger):
    """Filter dataset to companies only."""
    with PipelineLogger("Filter to Companies"):
        before_shape = df.shape
        
        keep_columns = [
            'ID_Code', 'Name', 'Country', 'Geographic_region', 'Entity_type',
            'Private_company', 'End_target', 'End_target_year', 'Status_of_end_target',
            'Interim_target', 'Interim_target_year', 'GHGs_covered',
            'Scope_1_coverage', 'Scope_2_coverage', 'Scope_3_coverage',
            'Published_plan', 'Reporting_mechanism', 'Accountability_delivery',
            'Carbon_credits', 'Separate_removal_target', 'Planning_removals',
            'Historical_emissions', 'Race_to_zero_member',
            'Company_annual_revenue', 'Industry', 'Employees', 'GHG_emissions'
        ]
        
        df = df[df['Entity_type'] == 'Company'][keep_columns].copy().reset_index(drop=True)
        df = df.drop(columns=['Entity_type'])
        
        log_transformation("Filter to Companies", before_shape, df.shape,
                          f"Kept {len(keep_columns)-1} columns")
        
        return df


def clean_data(df, config, logger):
    """Basic cleaning before split."""
    target_col = config['preprocessing_config']['target_column']
    missing_threshold = config['preprocessing_config']['missing_threshold']
    
    with PipelineLogger("Data Cleaning"):
        before_shape = df.shape
        
        # Drop high-missing columns
        missing_pct = df.isnull().sum() / len(df)
        high_missing = missing_pct[missing_pct > missing_threshold].index.tolist()
        
        if high_missing:
            logger.info(f"Dropping columns with >{missing_threshold*100:.0f}% missing: {high_missing}")
            df = df.drop(columns=high_missing)
        
        # Convert numeric columns
        numeric_cols = ['GHG_emissions', 'Company_annual_revenue', 'Employees']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
                logger.info(f"Converted {col} to numeric")
        
        # Drop rows with missing target
        rows_before = len(df)
        df = df.dropna(subset=[target_col])
        logger.info(f"Dropped {rows_before - len(df)} rows with missing target")
        
        # Drop rows with missing Industry (needed for imputation)
        rows_before = len(df)
        df = df.dropna(subset=['Industry'])
        logger.info(f"Dropped {rows_before - len(df)} rows with missing Industry")
        
        # Drop rows with invalid year values (e.g., End_target_year = 1000)
        year_cols = ['End_target_year', 'Interim_target_year']
        for col in year_cols:
            if col in df.columns:
                rows_before = len(df)
                invalid_years = (df[col] < 2000) | (df[col] > 2100)
                n_invalid = invalid_years.sum()
                if n_invalid > 0:
                    df = df[~invalid_years]
                    logger.info(f"Dropped {n_invalid} rows with invalid {col} (outside 2000-2100)")
        
        # Drop rows with extreme revenue outliers (e.g., $2.35 trillion)
        if 'Company_annual_revenue' in df.columns:
            rows_before = len(df)
            # Cap at $1 trillion (1e12) - values above are likely data errors
            extreme_revenue = df['Company_annual_revenue'] > 1e12
            n_extreme = extreme_revenue.sum()
            if n_extreme > 0:
                df = df[~extreme_revenue]
                logger.info(f"Dropped {n_extreme} rows with extreme revenue (>$1 trillion)")
        
        log_transformation("Data Cleaning", before_shape, df.shape)
        
        return df


def create_splits(df, config, logger):
    """Create stratified train/val/test splits."""
    target_col = config['preprocessing_config']['target_column']
    test_size = config['split_config']['test_size']
    val_size = config['split_config']['val_size']
    seed = config['random_seeds']['global_seed']
    
    with PipelineLogger("Train/Val/Test Split"):
        # Separate features and target
        identifier_cols = ['ID_Code', 'Name']
        
        # DROP: Scope_1_coverage and Scope_2_coverage due to TARGET LEAKAGE
        # These features are decided simultaneously with Scope_3_coverage during
        # corporate emissions reporting, making them proxies for the target.
        # Including them would inflate model performance artificially.
        leakage_cols = ['Scope_1_coverage', 'Scope_2_coverage']
        drop_cols = [target_col] + identifier_cols + leakage_cols
        drop_cols = [c for c in drop_cols if c in df.columns]  # Only drop if exists
        
        X = df.drop(columns=drop_cols)
        y = df[target_col]
        
        if any(c in df.columns for c in leakage_cols):
            logger.info(f"DROPPED due to target leakage: {[c for c in leakage_cols if c in df.columns]}")
        
        logger.info(f"Features: {X.shape[1]} columns")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        
        # Second split: separate train/validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            stratify=y_temp, random_state=seed
        )
        
        logger.info(f"Training set:   {X_train.shape[0]} rows ({X_train.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"Validation set: {X_val.shape[0]} rows ({X_val.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"Test set:       {X_test.shape[0]} rows ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_imputation_stats(train_df, config, logger):
    """Calculate imputation statistics from training data only."""
    group_col = config['preprocessing_config']['imputation_group_column']
    
    with PipelineLogger("Calculate Imputation Stats"):
        numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        
        if group_col in numeric_cols:
            numeric_cols.remove(group_col)
        if group_col in categorical_cols:
            categorical_cols.remove(group_col)
        
        stats = {'group': {}, 'global': {}}
        
        # Numeric: group median
        for col in numeric_cols:
            stats['group'][col] = train_df.groupby(group_col)[col].median().to_dict()
            stats['global'][col] = train_df[col].median()
        
        # Categorical: group mode
        for col in categorical_cols:
            group_modes = train_df.groupby(group_col)[col].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
            ).to_dict()
            stats['group'][col] = group_modes
            mode_result = train_df[col].mode()
            stats['global'][col] = mode_result.iloc[0] if len(mode_result) > 0 else np.nan
        
        logger.info(f"Imputation stats calculated for {len(numeric_cols)} numeric, {len(categorical_cols)} categorical columns")
        logger.info(f"Grouping by: {group_col}")
        
        return stats, group_col


def apply_imputation(df, group_col, stats, logger, set_name="DataFrame"):
    """Apply imputation using pre-calculated statistics."""
    before_missing = df.isnull().sum().sum()
    
    # Log per-column missingness before imputation
    col_missing = df.isnull().sum()
    cols_with_missing = col_missing[col_missing > 0]
    if len(cols_with_missing) > 0 and set_name == "Train":
        logger.info(f"  Pre-imputation missingness by column:")
        for col, count in cols_with_missing.items():
            pct = count / len(df) * 100
            logger.info(f"    {col}: {count} ({pct:.1f}%)")
    
    df_imputed = df.copy()
    
    # Create missingness indicator for Interim_target_year BEFORE imputation
    # Rationale: 33.5% missing - the fact that a company didn't specify an
    # interim target year may itself be predictive of Scope 3 coverage.
    # Companies with vague commitments may be less likely to report comprehensively.
    if 'Interim_target_year' in df_imputed.columns:
        df_imputed['Interim_target_year_missing'] = df_imputed['Interim_target_year'].isna().astype(int)
        if set_name == "Train":
            n_missing = df_imputed['Interim_target_year_missing'].sum()
            logger.info(f"  Created Interim_target_year_missing indicator: {n_missing} rows marked as missing")
    
    for col in stats['group'].keys():
        if col not in df_imputed.columns:
            continue
        
        group_values = stats['group'][col]
        global_value = stats['global'][col]
        
        for idx in df_imputed[df_imputed[col].isna()].index:
            group = df_imputed.loc[idx, group_col]
            if pd.notna(group) and group in group_values and pd.notna(group_values[group]):
                df_imputed.loc[idx, col] = group_values[group]
            else:
                df_imputed.loc[idx, col] = global_value
    
    after_missing = df_imputed.isnull().sum().sum()
    logger.info(f"{set_name}: {before_missing} → {after_missing} missing values")
    
    return df_imputed


def apply_log_transforms(df, config, logger):
    """Apply log transforms to skewed features."""
    log_cols = config['preprocessing_config']['log_transform_columns']
    
    df_transformed = df.copy()
    
    for col in log_cols:
        if col in df_transformed.columns:
            log_col = f'log_{col.lower().replace("company_annual_", "")}'
            df_transformed[log_col] = np.log1p(df_transformed[col].clip(lower=0))
            df_transformed = df_transformed.drop(columns=[col])
            logger.info(f"Log transformed: {col} → {log_col}")
    
    return df_transformed


def apply_scaling(X_train, X_val, X_test, config, logger):
    """
    Apply standard scaling to numerical features.
    
    CRITICAL: Scaler is fit ONLY on training data to prevent data leakage.
    
    Note: Scaling is optional for tree-based models (RandomForest) but important
    for linear models (Logistic Regression, SVM). We apply it for model flexibility.
    """
    scale_cols = config['preprocessing_config'].get('scale_columns', [])
    
    if not scale_cols:
        # Default: scale all numeric columns except year columns
        scale_cols = ['log_revenue', 'log_employees']
        logger.info(f"No scale_columns in config; defaulting to: {scale_cols}")
    
    # Filter to columns that exist
    scale_cols = [c for c in scale_cols if c in X_train.columns]
    
    if not scale_cols:
        logger.info("No columns to scale")
        return X_train, X_val, X_test, None
    
    with PipelineLogger("Standard Scaling"):
        scaler = StandardScaler()
        
        # Fit on training data ONLY
        X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        
        # Transform validation and test using SAME scaler
        X_val[scale_cols] = scaler.transform(X_val[scale_cols])
        X_test[scale_cols] = scaler.transform(X_test[scale_cols])
        
        # Log statistics
        for i, col in enumerate(scale_cols):
            logger.info(f"Scaled {col}: mean={scaler.mean_[i]:.2f}, std={scaler.scale_[i]:.2f}")
        
        # Save scaler for inference
        scaler_path = 'models/scaler.joblib'
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to: {scaler_path}")
        
        return X_train, X_val, X_test, scaler


def engineer_datetime_features(df, logger):
    """
    Engineer date-time features from year columns.
    
    Features created:
    1. years_to_end_target: Years remaining until end target year
    2. years_to_interim_target: Years remaining until interim target year  
    3. target_gap: Years between interim and end targets
    
    Reference date: TRAINING_REFERENCE_YEAR (defined at top of file)
    """
    with PipelineLogger("Date-Time Feature Engineering"):
        df_fe = df.copy()
        
        # Only create target_gap - the gap between interim and end targets
        # Note: years_to_end_target and years_to_interim_target are redundant
        # (perfectly correlated with raw year columns, just shifted by reference year)
        if 'End_target_year' in df_fe.columns and 'Interim_target_year' in df_fe.columns:
            df_fe['target_gap'] = df_fe['End_target_year'] - df_fe['Interim_target_year']
            logger.info(f"Created: target_gap (range: {df_fe['target_gap'].min():.0f} to {df_fe['target_gap'].max():.0f})")
        
        logger.info(f"Date-time feature added: target_gap (gap between end and interim targets)")
        
        return df_fe


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, config, logger):
    """Save preprocessed splits to CSV."""
    target_col = config['preprocessing_config']['target_column']
    output_dir = config['data_config']['preprocessed_dir']
    
    with PipelineLogger("Save Preprocessed Data"):
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine features and target
        train_df = X_train.copy()
        train_df[target_col] = y_train.values
        
        val_df = X_val.copy()
        val_df[target_col] = y_val.values
        
        test_df = X_test.copy()
        test_df[target_col] = y_test.values
        
        # Save
        train_path = os.path.join(output_dir, config['data_config']['train_file'])
        val_path = os.path.join(output_dir, config['data_config']['val_file'])
        test_path = os.path.join(output_dir, config['data_config']['test_file'])
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Saved: {train_path} ({train_df.shape})")
        logger.info(f"Saved: {val_path} ({val_df.shape})")
        logger.info(f"Saved: {test_path} ({test_df.shape})")
        logger.info(f"Final columns: {train_df.columns.tolist()}")


def main():
    """Run the complete preprocessing pipeline with logging."""
    # Setup
    logger = setup_logging('logs/preprocess.log')
    logger.info("PREPROCESSING PIPELINE STARTED")
    
    # Load configuration
    config = load_config('RUN_CONFIG.json')
    
    # Set random seeds
    set_random_seeds(config['random_seeds']['global_seed'])
    
    try:
        # Step 1: Load and verify data
        df = load_and_verify_data(config, logger)
        
        # Step 2: Filter to companies
        df = filter_to_companies(df, logger)
        
        # Step 3: Clean data
        df = clean_data(df, config, logger)
        
        # Step 4: Create splits (BEFORE imputation)
        X_train, X_val, X_test, y_train, y_val, y_test = create_splits(df, config, logger)
        
        # Step 5: Calculate imputation stats from TRAINING only
        imputation_stats, group_col = calculate_imputation_stats(X_train, config, logger)
        
        # Step 6: Apply imputation
        with PipelineLogger("Apply Imputation"):
            X_train = apply_imputation(X_train, group_col, imputation_stats, logger, "Train")
            X_val = apply_imputation(X_val, group_col, imputation_stats, logger, "Val")
            X_test = apply_imputation(X_test, group_col, imputation_stats, logger, "Test")
        
        # Step 7: Log transforms
        with PipelineLogger("Log Transforms"):
            X_train = apply_log_transforms(X_train, config, logger)
            X_val = apply_log_transforms(X_val, config, logger)
            X_test = apply_log_transforms(X_test, config, logger)
        
        # Step 8: Standard scaling (fit on train only)
        X_train, X_val, X_test, scaler = apply_scaling(X_train, X_val, X_test, config, logger)
        
        # Step 9: Save results
        save_splits(X_train, X_val, X_test, y_train, y_val, y_test, config, logger)
        
        logger.info("=" * 70)
        logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        
        print("\n✅ Preprocessing complete! Check logs/preprocess.log for details.")
        
    except Exception as e:
        logger.error(f"PIPELINE FAILED: {str(e)}")
        raise


if __name__ == '__main__':
    main()
