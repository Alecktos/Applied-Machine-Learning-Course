#Load the ML library
library(caret)

#Read the dataset
dataset <- read.csv("Iris/iris.csv")

#setup 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# method = "rpart" # decision tree
method = 'knn' # k-Nearest Neighbors

#Train model using CART
set.seed(7)
cart <- train(species~., data=dataset, method=method, metric=metric, trControl=control)

#Print result
print(cart)

# Accuracy is the percentage of correctly classifies instances out of all instances
# Kappa or Cohen’s Kappa is like classification accuracy, except that it is normalized at the baseline of random chance on your dataset. It is a more useful measure to use on problems that have an imbalance in the classes (e.g. 70-30 split for classes 0 and 1 and you can achieve 70% accuracy by predicting all instances are for class 0)

# The complexity parameter (cp) is used to control the size of the decision tree and to select the optimal tree size. If the cost of adding another variable to the decision tree from the current node is above the value of cp, then tree building does not continue. We could also say that tree construction does not continue unless it would decrease the overall lack of fit by a factor of cp.
# Think that the tree will stopp building when cp is zero.

