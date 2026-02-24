Generate Raw Uid: Concatenate raw strings (card1 + addr1 + etc.) to create a unique ID for the person.

Group by Uid: Sort by time. Now the "Identity" of the user is stored in the structure of your data (which transactions follow which).

Perform Target Encoding: Now convert the features into numbers.

__________________________

# Order of Data Operations

The order you proposed is almost right, but in a production PyTorch pipeline, it usually follows this "Senior" flow:

1. Handling Missing Data & Types: You can't calculate correlations or run PCA if there are NaNs or strings. You must impute and encode first.

2. Feature Engineering: Create your Hour_of_Day, Uid, and TransactionCount features before selection, as these new features might be more valuable than the raw ones.

3. Normalization (Scaling): PCA and Neural Networks (MLP/VAE) require normalized data (Z-scores). If you don't scale, PCA will be dominated by the feature with the largest numbers (like TransactionAmt).

    Feature Selection/Reduction: Now that the data is clean and scaled, you run your Correlation filters and PCA on the V-features.