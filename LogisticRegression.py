from sklearn.datasets import load_digits
digits = load_digits()

# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
print("Image Data Shape" , digits.data.shape)
# Print to show there are 1797 labels (integers from 0–9)
print("Label Data Shape", digits.target.shape)

#Showing the Images and the Labels
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
 plt.subplot(1, 5, index + 1)
 plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)
 
 #Splitting Data into Training and Test Sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

#Import the logistic regression model
from sklearn.linear_model import LogisticRegression

# instanciate the model
# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()

# Training the model on the data
logisticRegr.fit(x_train, y_train)

# Predict for One Observation (image)
# Returns a NumPy Array
a = logisticRegr.predict(x_test[0].reshape(1,-1))
print(a[0])

# Predict for Multiple Observations (images) at Once
logisticRegr.predict(x_test[0:10])

predictions = logisticRegr.predict(x_test)

# Measuring Model Performance
# Use score method to get accuracy of model
score = logisticRegr.score(x_test, y_test)
print(score)

# Confusion Matrix
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)