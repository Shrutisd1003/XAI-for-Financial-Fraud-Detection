import pandas as pd

# Assuming original_data is your DataFrame
# For demonstration, let's create a sample DataFrame
original_data = pd.DataFrame({
    'A': [1.234567, 2.345678, 3.456789],
    'B': [4.567890, 5.678901, 6.789012]
})

# Round off all numeric data to 2 decimal places
rounded_data = original_data.round(2)

# Print the original and rounded DataFrame
print("Original DataFrame:")
print(original_data)
print("\nRounded DataFrame:")
print(rounded_data)
