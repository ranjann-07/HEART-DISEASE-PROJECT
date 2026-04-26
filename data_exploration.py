import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def clean_and_explore_data():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('heart.csv')
    
    print("\n--- Initial Data Overview ---")
    print(df.head())
    print("\nShape of dataset:", df.shape)
    
    # Data Cleaning
    print("\n--- Data Cleaning ---")
    missing_values = df.isnull().sum()
    print("Missing values in each column:\n", missing_values)
    
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    
    if duplicates > 0:
        print("Removing duplicates...")
        df = df.drop_duplicates()
        print("New shape after removing duplicates:", df.shape)

    # Exploratory Data Analysis (EDA)
    print("\n--- Exploratory Data Analysis ---")
    
    # Create directory for saving plots
    os.makedirs('images', exist_ok=True)
    
    # 1. Target Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette='viridis')
    plt.title('Target Distribution (0 = No Disease, 1 = Disease)')
    plt.savefig('images/target_distribution.png')
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig('images/correlation_heatmap.png')
    plt.close()
    
    # 3. Age Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='age', hue='target', kde=True, palette='viridis', multiple="stack")
    plt.title('Age Distribution by Target')
    plt.savefig('images/age_distribution.png')
    plt.close()
    
    # 4. Sex vs Target
    plt.figure(figsize=(6, 4))
    sns.countplot(x='sex', hue='target', data=df, palette='viridis')
    plt.title('Heart Disease Frequency for Sex (0 = Female, 1 = Male)')
    plt.legend(["No Disease", "Disease"])
    plt.savefig('images/sex_target.png')
    plt.close()
    
    # 5. Chest Pain Type vs Target
    plt.figure(figsize=(8, 5))
    sns.countplot(x='cp', hue='target', data=df, palette='viridis')
    plt.title('Heart Disease Frequency by Chest Pain Type')
    plt.xlabel('Chest Pain Type (0, 1, 2, 3)')
    plt.legend(["No Disease", "Disease"])
    plt.savefig('images/cp_target.png')
    plt.close()

    print("EDA completed. Visualizations saved in the 'images' directory.")
    
    # Save the cleaned dataset for modeling
    df.to_csv('heart_cleaned.csv', index=False)
    print("Cleaned dataset saved as 'heart_cleaned.csv'.")

if __name__ == "__main__":
    clean_and_explore_data()
