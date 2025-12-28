"""
Feature Engineering Pipeline

This module provides a scikit-learn compatible Pipeline that:
1. Loads preprocessed data from data/interim/
2. Applies encoding (ordinal, one-hot, frequency-based grouping)
3. Combines with a classifier for end-to-end training and inference

The preprocessing_pipeline.py handles:
- Data cleaning, imputation, log transforms, scaling, missingness indicators

This pipeline handles:
- Categorical encoding (ordinal + one-hot)
- High-cardinality grouping (Country → top 10 + Other)
- Model integration

Usage:
    from src.pipeline import create_pipeline, load_data
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_file='logs/feature_engineering.log'):
    """Configure logging for feature engineering pipeline."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info(f"FEATURE ENGINEERING SESSION STARTED: {datetime.now().isoformat()}")
    logger.info("=" * 70)
    return logger

# Initialize logger (will be set up in main)
logger = logging.getLogger(__name__)

# =============================================================================
# FEATURE DEFINITIONS (from final_features.txt and encoding_strategy.md)
# =============================================================================

TARGET_COL = 'Scope_3_coverage'

# Numeric features (already scaled in preprocessing)
NUMERIC_COLS = [
    'End_target_year',
    'Interim_target_year',
    'log_revenue',
    'log_employees',
    'Interim_target_year_missing'  # Binary indicator (0/1)
]

# Ordinal features with defined order
ORDINAL_FEATURES = {
    # Binary Yes/No
    'Published_plan': ['No', 'Yes'],
    'Private_company': ['No', 'Yes'],
    'Race_to_zero_member': ['No', 'Yes'],
    'Separate_removal_target': ['No', 'Yes'],
    'Historical_emissions': ['No', 'Yes'],
    
    # Ternary
    'Accountability_delivery': ['Not Specified', 'No', 'Yes'],
    'Carbon_credits': ['No', 'Not Specified', 'Yes'],
    
    # Multi-level
    'GHGs_covered': ['Not Specified', 'Carbon dioxide only', 'Carbon dioxide and other GHGs'],
    'Reporting_mechanism': ['No reporting mechanism', 'Less frequently than annually', 'Annual reporting'],
    'Status_of_end_target': [
        'Proposed / in discussion',
        'Declaration / pledge',
        'In policy document',
        'In corporate strategy',
        'In law',
        'Achieved (self-declared)',
        'Achieved (externally validated)'
    ],
    'Planning_removals': [
        'No',
        'Not Specified',
        'Yes (unspecified)',
        'Yes (nature-based removals e.g. Forestation, soil carbon enhancement)',
        'Yes (CCS-based removals e.g. BECCS, DACCS)',
        'Yes (nature-based and CCS-based removals)'
    ]
}

# Nominal features for one-hot encoding
NOMINAL_COLS = [
    'Geographic_region',  # 8 categories
    'Industry',           # 13 categories
    'End_target',         # 7 categories
    'Interim_target'      # 7 categories
]

# High cardinality feature - special handling
HIGH_CARDINALITY_COL = 'Country'
TOP_N_COUNTRIES = 10  # Keep top 10, group rest as "Other"


# =============================================================================
# CUSTOM TRANSFORMERS
# =============================================================================

class CountryGrouper(BaseEstimator, TransformerMixin):
    """
    Groups countries into top N by frequency + "Other".
    
    Rationale: Country has 61 unique values. One-hot encoding would create
    61 sparse columns. Grouping to top 10 + Other reduces to 11 columns
    while retaining the most common countries.
    """
    
    def __init__(self, top_n=10, other_label='Other'):
        self.top_n = top_n
        self.other_label = other_label
        self.top_countries_ = None
    
    def fit(self, X, y=None):
        """Learn top N countries from training data."""
        if isinstance(X, pd.DataFrame):
            country_col = X[HIGH_CARDINALITY_COL]
        else:
            country_col = pd.Series(X.ravel())
        
        # Get top N by frequency
        value_counts = country_col.value_counts()
        self.top_countries_ = value_counts.head(self.top_n).index.tolist()
        return self
    
    def transform(self, X):
        """Replace rare countries with 'Other'."""
        if isinstance(X, pd.DataFrame):
            result = X.copy()
            result[HIGH_CARDINALITY_COL] = result[HIGH_CARDINALITY_COL].apply(
                lambda x: x if x in self.top_countries_ else self.other_label
            )
            return result
        else:
            return np.array([
                x if x in self.top_countries_ else self.other_label
                for x in X.ravel()
            ]).reshape(-1, 1)
    
    def get_feature_names_out(self, input_features=None):
        return [HIGH_CARDINALITY_COL]


# =============================================================================
# PIPELINE CREATION
# =============================================================================

def create_preprocessor():
    """
    Create the ColumnTransformer for feature encoding.
    
    Returns:
        ColumnTransformer: Fitted on train, transforms all splits
    """
    
    # Numeric features: pass through (already scaled in preprocessing_pipeline.py)
    numeric_transformer = 'passthrough'
    
    # Ordinal features: map to integers preserving order
    ordinal_transformer = OrdinalEncoder(
        categories=[ORDINAL_FEATURES[col] for col in ORDINAL_FEATURES.keys()],
        handle_unknown='use_encoded_value',
        unknown_value=-1  # Unseen categories get -1
    )
    
    # Nominal features: one-hot encode
    nominal_transformer = OneHotEncoder(
        handle_unknown='ignore',  # Unseen categories get all zeros
        sparse_output=False,
        drop='first'  # Drop first category to avoid multicollinearity
    )
    
    # Country: one-hot after grouping (handled separately)
    country_transformer = Pipeline([
        ('group', CountryGrouper(top_n=TOP_N_COUNTRIES)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])
    
    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_COLS),
            ('ord', ordinal_transformer, list(ORDINAL_FEATURES.keys())),
            ('nom', nominal_transformer, NOMINAL_COLS),
            ('country', country_transformer, [HIGH_CARDINALITY_COL])
        ],
        remainder='drop',  # Drop any unlisted columns
        verbose_feature_names_out=True
    )
    
    return preprocessor


def create_pipeline(classifier=None):
    """
    Create the full pipeline with optional classifier.
    
    Args:
        classifier: sklearn classifier (e.g., RandomForestClassifier)
                   If None, returns preprocessor only
    
    Returns:
        Pipeline: Ready for fit() and predict()
    """
    preprocessor = create_preprocessor()
    
    if classifier is None:
        return Pipeline([
            ('preprocessor', preprocessor)
        ])
    else:
        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_dir='data/interim'):
    """
    Load preprocessed train/val/test splits.
    
    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Separate features and target
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    
    X_val = val_df.drop(columns=[TARGET_COL])
    y_val = val_df[TARGET_COL]
    
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_target(y_train, y_val, y_test):
    """
    Encode target variable using LabelEncoder.
    
    Returns:
        Tuple: (y_train_enc, y_val_enc, y_test_enc, label_encoder)
    """
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    
    return y_train_enc, y_val_enc, y_test_enc, le


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_pipeline(pipeline, label_encoder, path='models/pipeline.pkl'):
    """Save fitted pipeline and label encoder."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({
        'pipeline': pipeline,
        'label_encoder': label_encoder,
        'feature_names': list(ORDINAL_FEATURES.keys()) + NOMINAL_COLS + [HIGH_CARDINALITY_COL] + NUMERIC_COLS
    }, path)
    print(f"✅ Pipeline saved to: {path}")


def load_pipeline(path='models/pipeline.pkl'):
    """Load fitted pipeline and label encoder."""
    data = joblib.load(path)
    return data['pipeline'], data['label_encoder']


def verify_pipeline(X_train, X_val, X_test, preprocessor):
    """
    Verify pipeline produces consistent output shapes.
    
    Args:
        X_train, X_val, X_test: DataFrames
        preprocessor: Fitted ColumnTransformer
    
    Returns:
        bool: True if verification passes
    """
    logger.info("Transforming datasets...")
    X_train_enc = preprocessor.transform(X_train)
    X_val_enc = preprocessor.transform(X_val)
    X_test_enc = preprocessor.transform(X_test)
    
    logger.info(f"Encoded shapes:")
    logger.info(f"  Train: {X_train_enc.shape}")
    logger.info(f"  Val:   {X_val_enc.shape}")
    logger.info(f"  Test:  {X_test_enc.shape}")
    
    # Assertion: same number of columns
    assert X_train_enc.shape[1] == X_val_enc.shape[1] == X_test_enc.shape[1], \
        "Column mismatch across splits!"
    
    logger.info(f"✅ Verification passed: {X_train_enc.shape[1]} columns across all splits")
    return True


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    # Setup logging
    logger = setup_logging()
    
    # Load data
    logger.info("Loading preprocessed data from data/interim/")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    logger.info(f"Data loaded:")
    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val:   {X_val.shape}")
    logger.info(f"  Test:  {X_test.shape}")
    
    # Encode target
    logger.info("Encoding target variable with LabelEncoder")
    y_train_enc, y_val_enc, y_test_enc, le = encode_target(y_train, y_val, y_test)
    logger.info(f"Target classes: {list(le.classes_)}")
    
    # Create preprocessor
    logger.info("Creating ColumnTransformer preprocessor")
    preprocessor = create_preprocessor()
    
    # Log encoding strategies
    logger.info(f"Numeric features (passthrough): {NUMERIC_COLS}")
    logger.info(f"Ordinal features ({len(ORDINAL_FEATURES)}): {list(ORDINAL_FEATURES.keys())}")
    logger.info(f"Nominal features (one-hot): {NOMINAL_COLS}")
    logger.info(f"High-cardinality feature: {HIGH_CARDINALITY_COL} → top {TOP_N_COUNTRIES} + Other")
    
    # Fit on training data
    logger.info("Fitting preprocessor on training data...")
    preprocessor.fit(X_train)
    
    # Log top countries learned
    country_grouper = preprocessor.named_transformers_['country'].named_steps['group']
    logger.info(f"Top {TOP_N_COUNTRIES} countries learned: {country_grouper.top_countries_}")
    
    # Verify
    verify_pipeline(X_train, X_val, X_test, preprocessor)
    
    # Log feature names
    feature_names = preprocessor.get_feature_names_out()
    logger.info(f"Total encoded features: {len(feature_names)}")
    logger.info(f"Feature name examples: {list(feature_names[:5])}...")
    
    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
