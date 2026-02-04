import kagglehub

# Download the credit card fraud dataset
path = kagglehub.dataset_download('mlg-ulb/creditcardfraud')
print(f"Dataset downloaded to: {path}")
