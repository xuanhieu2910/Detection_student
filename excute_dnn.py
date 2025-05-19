import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/Research/Computer_Vision/StudentEngagement_2/runs/training/Sub_8_02_02_05_2025_3.csv')

# Split into features and labels
X = data.drop(columns=['cls','file_name'])
y = data['cls']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training, validation, and test sets (70%, 20%, 10%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))  # Use Input layer instead of input_dim in Dense
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))  # Dropout
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))  # Dropout
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Lists to store training and validation accuracy for plotting later
train_accuracies = []
val_accuracies = []

# Train the model manually using a loop
epochs = 300
batch_size = 16

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Train the model for one epoch
    history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size,
                        validation_data=(X_val, y_val), verbose=1)

    # Save the model every 20 epochs
    if (epoch + 1) % 100 == 0:
        model.save(f"student_behavior_model_v4_epoch_{epoch + 1}.keras")
        print(f"Model saved at epoch {epoch + 1}")

    # Collect the training and validation accuracy for plotting
    train_accuracies.append(history.history['accuracy'][-1])
    val_accuracies.append(history.history['val_accuracy'][-1])

    # Optionally, print progress (accuracy, loss, etc.)
    print(f"Training accuracy: {history.history['accuracy'][-1]:.6f} | Validation accuracy: {history.history['val_accuracy'][-1]:.6f}")

# Evaluate the model after training
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy*100:.8f}%')

# Predict the labels for the test set
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calculate precision, recall, and ROC AUC
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Precision: {precision:.8f}')
print(f'Recall: {recall:.8f}')
print(f'ROC AUC: {roc_auc:.8f}')

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict(X_test))
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Optionally, you can also visualize training history (accuracy/loss over epochs)
plt.plot(train_accuracies, label='Training accuracy')
plt.plot(val_accuracies, label='Validation accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
