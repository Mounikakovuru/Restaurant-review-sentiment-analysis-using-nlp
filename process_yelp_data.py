import pandas as pd
import os

# --- Configuration ---
DOWNLOADED_FILENAME = 'yelp restaurant reviews.csv'
OUTPUT_FILENAME = 'Restaurant_Reviews.tsv'

# --- Load the downloaded dataset ---
try:
    df_yelp = pd.read_csv(DOWNLOADED_FILENAME)
    print(f"Successfully loaded '{DOWNLOADED_FILENAME}' with {len(df_yelp)} reviews.")
except FileNotFoundError:
    print(f"Error: '{DOWNLOADED_FILENAME}' not found. Please make sure the downloaded Yelp review file is in the same directory.")
    print("Also, ensure the filename above matches the exact name, including capitalization and spaces.")
    exit()
except Exception as e:
    print(f"An error occurred while loading '{DOWNLOADED_FILENAME}': {e}")
    print("Common issues: file might be corrupted, or not a standard CSV format (e.g., uses a different delimiter).")
    exit()

# --- Check for essential columns and rename/process them ---
# !! CORRECTED CANDIDATE NAMES TO MATCH YOUR FILE'S EXACT COLUMN NAMES !!
text_column_candidates = ['Review Text', 'review text', 'text', 'review_text', 'Review_Text', 'review'] # Added 'Review Text'
rating_column_candidates = ['Rating', 'rating', 'stars', 'score'] # Added 'Rating'

text_col_found = None
for cand in text_column_candidates:
    if cand in df_yelp.columns:
        text_col_found = cand
        break

rating_col_found = None
for cand in rating_column_candidates:
    if cand in df_yelp.columns:
        rating_col_found = cand
        break

if not text_col_found:
    print(f"Error: Could not find a review text column. Looked for: {text_column_candidates}")
    print(f"Found columns: {df_yelp.columns.tolist()}")
    exit()

if not rating_col_found:
    print(f"Error: Could not find a rating column. Looked for: {rating_column_candidates}")
    print(f"Found columns: {df_yelp.columns.tolist()}")
    exit()

# 1. Rename the found text column to 'Review'
df_processed = df_yelp.rename(columns={text_col_found: 'Review'})

# 2. Convert the found rating column to 'Liked' (0 or 1)
df_processed.loc[:, 'Liked'] = df_processed[rating_col_found].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else None))

# Drop rows where 'Liked' is None (i.e., the 3-star reviews)
df_processed.dropna(subset=['Liked'], inplace=True)

# Ensure 'Liked' column is integer type
df_processed.loc[:, 'Liked'] = df_processed['Liked'].astype(int)

# --- Select only the 'Review' and 'Liked' columns ---
df_final = df_processed[['Review', 'Liked']]

# --- Check the new class distribution ---
print("\n--- New Dataset Class Distribution (after processing) ---")
print(df_final['Liked'].value_counts())
print("-" * 50)

# --- Save to the output TSV file ---
try:
    df_final.to_csv(OUTPUT_FILENAME, sep='\t', index=False, quoting=3)
    print(f"\nSuccessfully processed {len(df_final)} reviews and saved to '{OUTPUT_FILENAME}'.")
    print("You can now proceed to run 'python train_model.py'.")
except Exception as e:
    print(f"Error saving processed data to '{OUTPUT_FILENAME}': {e}")