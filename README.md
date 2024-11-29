# Machine_Learning

In the field of Machine Learning, three main paradigms are commonly used: supervised learning, unsupervised learning, and reinforcement learning. These paradigms differ in the type of data they use and their tasks. Supervised learning, which involves training models on labeled data to predict specific outcomes, is the most widely used. 
Unsupervised learning focuses on finding hidden patterns in unlabeled data, while reinforcement learning involves training models through feedback (rewards or penalties) in an interactive environment. In some cases, these paradigms can be combined, such as in semi-supervised learning or hybrid models, to improve performance or handle more complex tasks.

![mll](https://github.com/user-attachments/assets/1d571f2d-44be-4c92-a175-aefe7df5e721)

# Supervised Leaning 

Supervised learning algorithms are trained on a labeled dataset, which includes both input features (data) and corresponding output labels (targets). The goal of the algorithm is to learn the relationship or mapping between the input and the output, so that it can make accurate predictions on new, unseen data. In supervised learning, the model learns from examples where the correct answers (outputs) are already provided.
https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning

![cls](https://github.com/user-attachments/assets/5d6dc513-2c48-44df-bc4f-ebde93f318fa)

There are two main types of supervised learning problems: they are classification which involves predicting a class label and regression which involves predicting a numerical value.

Classification: Supervised learning problem that involves predicting a class label.
Regression: Supervised learning problem that involves predicting a numerical label.


![sup](https://github.com/user-attachments/assets/43046009-20d3-430b-ba1e-e763c36c163a)


In regression, the goal is to predict a continuous numerical value based on input features.

![reg](https://github.com/user-attachments/assets/ec14580e-e253-4343-b2c0-86f4d4d52b84)

In classification tasks, the goal is to predict a discrete class or category label for the input data.

![cls](https://github.com/user-attachments/assets/a07a35f0-b070-49c2-ae3f-3517bbc49e88)


Linear regression: A simple algorithm that predicts a continuous value based on a set of input values (Regression). It uses a logistic function (also known as the sigmoid function) to map any real-valued number into the range between 0 and 1, making it suitable for predicting probabilities of binary outcomes.

![lr](https://github.com/user-attachments/assets/7b5dafc2-1950-4650-ba67-d1b4bd9ee13f)

Decision trees: A more powerful algorithm that can be used to classify or predict both continuous and categorical values.
![dt](https://github.com/user-attachments/assets/765df4d8-d0a2-44c5-aaf6-90782f2ca449)

Random Forest: An ensemble method using multiple decision trees for classification and regression, known for its robustness, accuracy, and ability to handle large datasets.


Support Vector Machines (SVM): A versatile, powerful algorithm for classification and regression that works by finding the best hyperplane to separate classes, with the added benefit of using kernels for non-linear problems.
![svm](https://github.com/user-attachments/assets/cef7e3d7-7aa1-423a-aa8f-71b6bb8e2d11)


Naive Bayes is a probabilistic classifier that applies Bayes' Theorem with the "naive" assumption that the features are independent. It's simple, fast, and effective for many classification tasks, especially for text data (spam detection, sentiment analysis).
![Kt](https://github.com/user-attachments/assets/ee233470-2a42-4848-823b-2b9500729e16)

k-Nearest Neighbors (k-NN) is a simple, non-parametric algorithm for classification and regression that makes predictions based on the majority class or average of the k-nearest neighbors in the training data.

Artificial Neural Networks (ANNs) are a powerful class of algorithms inspired by the human brain. They are capable of learning complex, non-linear relationships and are widely used for tasks such as image recognition, speech processing, and text classification.

![yt](https://github.com/user-attachments/assets/bddca223-691d-456b-a3a8-cd9d53631a3d)

# Unsupervised learning

Unsupervised learning involves training algorithms on unlabeled data to find hidden patterns or structures, such as grouping data into clusters or reducing the dimensionality of data for easier analysis. It is particularly useful for discovering unknown patterns in data and is widely applied in areas like customer segmentation, anomaly detection, and data visualization.
https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-unsupervised-learning


![un](https://github.com/user-attachments/assets/597b3179-14f3-4214-bb95-5d13d93bd019)


Principal Component Analysis (PCA) is a dimensionality reduction technique that reduces the number of features in a dataset by transforming the original features into a smaller set of uncorrelated principal components.

K-means clustering algorithm used to group similar data points into K clusters. The algorithm iteratively assigns data points to the nearest centroid, then updates the centroids based on the new assignments. It is efficient and widely used, but it has some limitations, including sensitivity to initialization, difficulty handling non-spherical clusters, and sensitivity to outliers.

![kmeans](https://github.com/user-attachments/assets/12d41acb-6a08-4938-aab1-1afcc38e877d)

Latent Semantic Analysis (LSA) is a dimensionality reduction technique used to uncover hidden patterns and relationships in text data. It works by applying Singular Value Decomposition (SVD) to a document-term matrix to capture the latent semantic structure between terms and documents. LSA is widely used in information retrieval, document clustering, topic modeling, and semantic search. While LSA has advantages like capturing hidden meanings and reducing complexity, it also has limitations related to computational cost and interpretability.
![svd](https://github.com/user-attachments/assets/276f094e-d1cd-46a2-9f31-312116939033)


Hierarchical Clustering: The flexibility of not needing to specify the number of clusters is a big advantage, especially for exploratory data analysis. However, its computational complexity grows quickly with larger datasets, making it less suitable for very large data.

Gaussian Mixture Models (GMM): The probabilistic nature allows GMMs to handle uncertainty, but they require careful initialization (e.g., using K-means for initial centroids). They are sensitive to the assumption that the data is Gaussian, which may not always hold.

DBSCAN: The ability to detect noise and outliers is one of DBSCAN’s strengths. However, its performance heavily relies on the right choice of ε (epsilon) and MinPts parameters, which can be challenging in practice. It’s less effective in handling clusters with varying densities.

Association Rule Learning: While it’s a great technique for discovering relationships (especially in retail or market analysis), the sheer number of rules it can generate can be overwhelming. Setting the right support and confidence thresholds is key to finding useful rules.

# Reinforcement Learning 
Reinforcement Learning (RL) involves an agent learning to make decisions by interacting with its environment and receiving feedback in the form of rewards or penalties. The agent uses trial and error to learn an optimal policy that maximizes long-term rewards. RL has been used in applications like robotics, game playing, autonomous vehicles, and more. 

![tr](https://github.com/user-attachments/assets/798c9838-2393-403b-be4e-cef6974a3563)



Q-learning
Q-learning is a model-free, value-based reinforcement learning algorithm. It learns the optimal policy by estimating the expected future rewards for each state-action pair. The goal of Q-learning is to learn a Q-function, which gives the expected cumulative reward for taking an action in a given state and following the optimal policy thereafter.


Policy Gradient Methods
Policy Gradient Methods directly optimize the policy itself instead of learning a value function. Rather than estimating the value of state-action pairs (as in Q-learning), these methods aim to learn a parameterized policy that maps states to actions, and the parameters are updated using gradient ascent.


Actor-Critic Methods
Actor-Critic Methods combine the benefits of both value-based and policy gradient methods. These methods have two components:

    Actor: The actor learns the policy (i.e., the action-selection strategy).
    Critic: The critic estimates the value function (i.e., the expected future reward for a given state).



Scikit-learn is one of the most popular and widely used libraries in Python for machine learning. It provides a simple and efficient set of tools for building and evaluating machine learning models, making it a go-to library for both beginners and experienced data scientists.
![sci](https://github.com/user-attachments/assets/4245d225-dc8d-43d4-8479-292552de7acd)

# Links

Introduction to Machine Learning with Scikit-Learn. Carpentries lesson.
https://carpentries-incubator.github.io/machine-learning-novice-sklearn/index.html
https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python/

https://scikit-learn.org/stable/modules/decomposition.html#decompositions

Using the Scikit-Learn to build ML models.

Steps

![st](https://github.com/user-attachments/assets/fcf9cf3c-1130-4e8b-83ab-ae5f2196b4c3)


Loading the data
Load the as a standard Pandas dataframe.

    import pandas as pd
    df = pd.read_csv(filename)

Split data: Train & Test
Split the dataset into training and test sets for both the X and y variables.

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

Function: sklearn.model_selection.train_test_split

Preprocessing the data
Getting the data ready before the model is fitted.

- Standarization
Standardize the features by removing the mean and scaling to unit variance.

Function: sklearn.preprocessing.StandardScaler

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    standarized_X = scaler.transform(X_train)
    standarized_X_test = scaler.transform(X_test)

- Normalization
Each sample with at least one non-zero component is rescaled independently of other samples so that its norm equals one.

Function: sklearn.preprocessing.Normalizer

    from sklearn.preprocessing import Normalizer
    scaler = Normalizer().fit(X_train)
    normalized_X = scaler.transform(X_train)
    normalized_X_test = scaler.transform(X_test)
   
 - Binarization
Binarize data (set feature values to 0 or 1) according to a threshold.
Function: sklearn.preprocessing.Binarizer

        from sklearn.preprocessing import Binarizer
        binarizer = Binarizer(threshold = 0.0).fit(X)
        binary_X = binarizer.transform(X_test)
Encoding Categorical Features
Encode’s target labels with values between 0 and n_classes-1.

Function: sklearn.preprocessing.LabelEncoder

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit_transform(X_train)

- Imputing Missing Values
Imputation transformer for completing missing values.

Function: sklearn.impute.SimpleImputer

    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values = 0, strategy = 'mean')
    imp.fit_transform(X_train)

Generating Polynomial Features

Generate a new feature matrix consisting of all polynomial combinations of the features with degrees less than or equal to the specified degree.

Function: sklearn.preprocessing.PolynomialFeatures

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(5)
    poly.fit_transform(X)

- Create the model
Creation of various supervised and unsupervised learning models.

Supervised Learning
- Linear Regression

      from sklearn.linear_model import LinearRegression
      lr  = LinearRegression(normalize = True)

Linear Support Vector Machines Classifier (SVC)

      from sklearn.svm import SVC
      svc = SVC(kernel = 'linear')

- Gaussian Naive Bayes

      from sklearn.naive_bayes import GaussianNB
      gnb = GaussianNB()

- k Nearest Neighbors Classifier (k-NN)

      from sklearn import neighbors
      knn = neighbors.KNeighborsClassifier(n_neighbors = 5)

# Unsupervised Learning

Dimension Reduction: Principal Component Analysis (PCA)

    from sklearn.decomposition import PCA
    pca = PCA(n_components = 0.95)

- Cluster: K Means

      from sklearn.cluster import KMeans
      k_means = KMeans(n_clusters = 3, random_state = 0)

- Fit the model

Fitting supervised and unsupervised learning models onto data.

- Supervised Learning

Fit the model to the data

    lr.fit(X, y)
    knn.fit(X_train,y_train)
    svc.fit(X_train,y_train)

- Unsupervised Learning
Fit the model to the data

       k_means.fit(X_train)
 
Fit the data, then transform it

    pca_model = pca.fit_transform(X_train)

Making predictions

Predicting test sets using trained models.

Predict labels

    #Supervised Estimators
    y_pred = lr.predict(X_test)
    
    #Unsupervised Estimators
    y_pred = k_means.predict(X_test)

- Estimate probability of a label

      y_pred = knn.predict_proba(X_test)

- Evaluating model performance

Various regression and classification metrics that determine how well a model performed on a test set.

Classification Metrics

Accuracy Score

    knn.score(X_test,y_test)
    
    from sklearn.metrics import accuracy_score
    accuracy_score(y_test,y_pred)


- Classification Report

      from sklearn.metrics import classification_report
      print(classification_report(y_test,y_pred))

- Confusion Matrix

      from sklearn .metrics import confusion_matrix
      print(confusion_matrix(y_test,y_pred))

- Regression Metrics

Mean Absolute Error

      from sklearn.metrics import mean_absolute_error
      mean_absolute_error(y_test,y_pred)
      

- Mean Squared Error

      from sklearn.metrics import mean_squared_error
      mean_squared_error(y_test,y_pred)

- R² Score
  
      from sklearn.metrics import r2_score
      r2_score(y_test, y_pred)

# Clustering Metrics

- Adjusted Rand Index
  
      from sklearn.metrics import adjusted_rand_score
      adjusted_rand_score(y_test,y_pred)

- Homogeneity Score

      from sklearn.metrics import homogeneity_score
      homogeneity_score(y_test,y_pred)

- V-measure Score

      from sklearn.metrics import v_measure_score
      v_measure_score(y_test,y_pred)

# Cross-Validation Score

Evaluate a score by cross-validation

from sklearn.model_selection import cross_val_score
print(cross_val_score(knn, X_train, y_train, cv=4))
 Model tuning


Finding correct parameter values that will maximize a model’s prediction accuracy.

Grid Search

A Grid Search is an exhaustive search over specified parameter values for an estimator. The example below attempts to find the right amount of clusters to specify for knn to maximize the model’s accuracy.

    from sklearn.model_selection import GridSearchCV
    
    params = {'n_neighbors': np.arange(1,3), 'metric':['euclidean','cityblock']}
    
    grid = GridSearchCV(estimator = knn, param_grid = params)
    grid.fit(X_train, y_train)
    print(grid.best_score_)
    print(grid.best_estimator_.n_neighbors)


# Randomized Parameter Optimization
Randomized search on hyperparameters. In contrast to Grid Search, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions. The number of parameter settings that are tried is given by n_iter.

    from sklearn.model_selection import RandomizedSearchCV
    params = {'n_neighbors':range(1,5), 'weights':['uniform','distance']}
    rsearch = RandomizedSearchCV(estimator = knn, param_distributions = params, cv = 4, n_iter = 8, random_state = 5)
    rseach.fit(X_train, y_train)
    print(rsearch.best_score_)


    
