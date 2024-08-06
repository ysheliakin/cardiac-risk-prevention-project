import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('dataset.csv')

# Separate features and target
X = data.drop(columns=['HeartDiseaseorAttack'])
y = data['HeartDiseaseorAttack']

# Splitting the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, stratify=y, random_state=45)

# Applying Random Under Sampling to balance the dataset
sampler = RandomUnderSampler(random_state=11)
xs, ys = sampler.fit_resample(xtrain, ytrain)

# Combine features and target into a DataFrame
balanced_dataset = pd.DataFrame(xs, columns=X.columns)
balanced_dataset['HeartDiseaseorAttack'] = ys

# Save the balanced dataset to a new CSV file
balanced_dataset.to_csv('balanced_dataset.csv', index=False)
