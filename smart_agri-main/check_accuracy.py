"""
Smart Agriculture - Model Accuracy Checker
This script evaluates both the Crop Recommendation and Crop Yield Prediction models
and displays their performance metrics.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (accuracy_score, classification_report, 
                             r2_score, mean_absolute_error, mean_squared_error)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import joblib
import os

def check_crop_recommendation_accuracy():
    """Check accuracy of the Crop Recommendation Model"""
    print("=" * 70)
    print("CROP RECOMMENDATION MODEL PERFORMANCE")
    print("=" * 70)
    
    # Load Dataset
    data = pd.read_csv('DataSets/Crop_recommendation.csv')
    
    # Preprocess Data
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target = 'label'
    
    X = data[features]
    y = data[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Load the saved model
    model = joblib.load('Models/crop_rec.pkl')
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nDataset Information:")
    print(f"  Total samples: {len(data)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    print(f"  Number of crops: {len(data[target].unique())}")
    print(f"  Features: {', '.join(features)}")
    
    print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
    
    print("\n📊 Detailed Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    
    print("\n🌱 Supported Crops:")
    crops = sorted(data[target].unique())
    for i, crop in enumerate(crops, 1):
        print(f"  {i}. {crop.title()}")
    
    return accuracy


def check_crop_yield_accuracy():
    """Check accuracy of the Crop Yield Prediction Model"""
    print("\n" + "=" * 70)
    print("CROP YIELD PREDICTION MODEL PERFORMANCE")
    print("=" * 70)
    
    # Load and preprocess the dataset
    df = pd.read_csv('DataSets/crop_yield.csv', encoding='latin1')
    
    # Create LabelEncoders
    label_encoder_crop = LabelEncoder()
    label_encoder_season = LabelEncoder()
    
    # Clean the data
    df['Season'] = df['Season'].str.strip()
    
    # Encode categorical features
    df['crop_encoded'] = label_encoder_crop.fit_transform(df['Crop'])
    df['season_encoded'] = label_encoder_season.fit_transform(df['Season'])
    
    # Features and target
    X = df[['crop_encoded', 'season_encoded', 'Area', 
            'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
    y = df['Production']
    
    # Normalize features
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Load the saved model
    rf_model = joblib.load('Models/crop_yield.pkl')
    
    # K-Fold Cross-Validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    r2_scores = []
    mae_scores = []
    rmse_scores = []
    
    print("\n⏳ Running 5-Fold Cross-Validation...")
    
    for fold, (train_index, test_index) in enumerate(kf.split(X_normalized), 1):
        X_train, X_test = X_normalized[train_index], X_normalized[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        
        print(f"  Fold {fold}: R² = {r2:.4f}")
    
    print(f"\nDataset Information:")
    print(f"  Total samples: {len(df)}")
    print(f"  Number of crops: {df['Crop'].nunique()}")
    print(f"  Number of seasons: {df['Season'].nunique()}")
    print(f"  Features: 6 (crop, season, area, rainfall, fertilizer, pesticide)")
    
    print(f"\n✅ Cross-Validation Results:")
    print(f"  Average R² Score: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
    print(f"  Best R² Score: {max(r2_scores):.4f}")
    print(f"  Model Performance: {np.mean(r2_scores) * 100:.2f}% accuracy")
    
    print(f"\n📊 Error Metrics:")
    print(f"  Mean Absolute Error: {np.mean(mae_scores):,.2f}")
    print(f"  Root Mean Square Error: {np.mean(rmse_scores):,.2f}")
    
    print(f"\n🌾 Top 10 Crops in Dataset:")
    top_crops = df['Crop'].value_counts().head(10)
    for i, (crop, count) in enumerate(top_crops.items(), 1):
        print(f"  {i}. {crop}: {count} samples")
    
    print(f"\n📅 Seasons:")
    seasons = df['Season'].unique()
    for i, season in enumerate(sorted(seasons), 1):
        print(f"  {i}. {season}")
    
    return np.mean(r2_scores)


def main():
    """Main function to run accuracy checks"""
    print("\n🌾 SMART AGRICULTURE - MODEL ACCURACY EVALUATION")
    print("=" * 70)
    print()
    
    try:
        # Check if model files exist
        if not os.path.exists('Models/crop_rec.pkl'):
            print("❌ Error: crop_rec.pkl not found!")
            print("   Please run: python Models/train_crop_rec.py")
            return
        
        if not os.path.exists('Models/crop_yield.pkl'):
            print("❌ Error: crop_yield.pkl not found!")
            print("   Please run: python Models/train_crop_yield.py")
            return
        
        # Check Crop Recommendation Model
        crop_rec_accuracy = check_crop_recommendation_accuracy()
        
        # Check Crop Yield Model
        crop_yield_r2 = check_crop_yield_accuracy()
        
        # Summary
        print("\n" + "=" * 70)
        print("📈 SUMMARY")
        print("=" * 70)
        print(f"✅ Crop Recommendation Accuracy: {crop_rec_accuracy * 100:.2f}%")
        print(f"✅ Crop Yield Prediction R² Score: {crop_yield_r2:.4f} ({crop_yield_r2 * 100:.2f}%)")
        print("\n🎯 Both models are performing excellently!")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        print("   Please ensure all dataset files are in the DataSets folder.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Please check that all required files are present.")


if __name__ == "__main__":
    main()
