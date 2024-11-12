# Lab-1: Regression with Pytorch
## Part One: Regression with PyTorch
1. Apply Exploratory Data Analysis (EDA)
   - **Goal**: Understand the dataset and visualize its important features.
   - **Steps**:
     - Load the data: Start by loading the dataset into a Pandas DataFrame.
     - Summary statistics: Use df.describe() to get an overview of numerical features.
     - Check for missing values: Use df.isnull().sum() to identify any missing data.
   - **Visualizations**:
     - Histograms to visualize the distribution of numerical features.
     - Correlation matrix (e.g., using sns.heatmap) to understand the relationships between features.
     - Pair plots or scatter plots to check how different features relate to the target variable.
  2. Build a Deep Neural Network (DNN) for Regression
     - **Goal**: Create a neural network model to predict the continuous target variable.
     - **Steps**:
       - **Preprocessing**:
         - Normalize the data (e.g., using StandardScaler from sklearn).
         - Split the data into training and testing sets.
      - **Define the DNN Model**:
        - Use torch.nn.Module to define the architecture.
        - Use the Mean Squared Error (MSE) loss function for regression tasks.
        - Use Adam optimizer or another optimizer suitable for regression.
  3. Hyperparameter Tuning with GridSearch
     **Goal**: Find the best hyperparameters for your model.
     **Steps**:
       - Use GridSearchCV from sklearn to find the best combination of hyperparameters (learning rate, optimizer, etc.).
  4. Visualize Loss and Accuracy
     **Goal**: Monitor how the model's performance evolves.
     **Steps**:
     - Track both training and testing loss, as well as accuracy over each epoch.
     - Plot the following:
       - Loss vs Epochs: Show the training and test loss over epochs.
       - Accuracy vs Epochs: Show the accuracy over epochs (for regression, accuracy might be less intuitive, but loss can give a good indication of performance).
  5. Regularization Techniques
     **Goal**: Prevent overfitting and improve the model's generalization.
     **Steps**:
       - Apply Dropout, L2 Regularization (Weight Decay), or Early Stopping.
       - Compare the performance after applying regularization techniques to the baseline model.
  ## Part Two: Multi-class Classification with PyTorch
  1. Preprocessing the Data
    **Goal**: Clean and normalize the dataset for classification.
    **Steps**:
     - Handle missing data (e.g., using imputation techniques).
     - Normalize or standardize the features.
     - Encode categorical features if necessary.
  2. Apply EDA
     **Goal**: Understand and visualize the dataset for classification tasks.
     **Steps**:
     - Visualize the distribution of the classes (e.g., using a bar plot).
     - Create pair plots or confusion matrix to check how well the features separate the classes.
  3. Data Augmentation
     **Goal**: Balance the dataset if the classes are imbalanced.
     **Steps**:
     - Use techniques like SMOTE (Synthetic Minority Over-sampling Technique) or random oversampling to create more data for underrepresented classes.
     - Alternatively, use weighting to handle class imbalance in the loss function.
  4. Build a Deep Neural Network (DNN) for Multi-class Classification
     **Goal**: Design a neural network to handle the classification task.
     **Steps**:
     - Define the model architecture with an output layer matching the number of classes (use softmax activation).
     - Use Cross-Entropy Loss for multi-class classification.
  5. Hyperparameter Tuning with GridSearch
     **Goal**: Tune the model's hyperparameters using GridSearchCV.
     **Steps**:
       - Perform a grid search for parameters like learning rate, batch size, and number of layers.
  6. Visualize Loss and Accuracy
     **Goal**: Track the model’s training and test performance.
     **Steps**:
       - Plot training and test loss/accuracy over epochs.
       - This helps you understand the model's learning curve and performance.
  8. Calculate Evaluation Metrics
     **Goal**: Assess model performance using common classification metrics.
     **Steps**:
       - Calculate accuracy, precision, recall, F1-score, and sensitivity on both training and test datasets.
       - Use confusion matrix to evaluate performance for each class.
  9. Regularization Techniques
      **Goal**: Improve generalization by using regularization.
      **Steps**:
        - Apply regularization techniques like Dropout or L2 regularization and compare the results with the baseline model.
