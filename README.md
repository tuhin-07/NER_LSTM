# 🧠 Named Entity Recognition using BiLSTM (LSTM + Word Embeddings)

This repository demonstrates a **Named Entity Recognition (NER)** model built using a **Bidirectional LSTM** network in TensorFlow/Keras.

## 📁 Dataset

- The dataset is assumed to be in a CSV format (`dataset.csv`) with columns:
  - `Sentence #`
  - `Word`
  - `POS` (Part of Speech)
  - `Tag` (NER Label)
- Missing values are forward-filled (`ffill`).
- Encoding used: `latin1`

## 🧩 Key Steps

1. **Preprocessing:**
   - Fill missing values
   - Group words by sentences
   - Map words and tags to integer indices
   - Pad sequences to a uniform length of 50

2. **Model Architecture:**
   - **Embedding Layer**: Transforms word indices into dense vectors
   - **Spatial Dropout**: Regularization
   - **Bidirectional LSTM**: Captures both past and future context
   - **TimeDistributed Dense Layer**: Outputs tag probabilities per word

3. **Training:**
   - Optimizer: `Adam`
   - Loss: `categorical_crossentropy`
   - Metric: `accuracy`
   - Train-test split: 90% training, 10% test
   - Validation split: 20% of training data

4. **Evaluation:**
   - Accuracy achieved: **~98%**

## 📊 Training Accuracy Plot

*(A plot of training vs validation accuracy is generated during training)*

## 📦 Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `tensorflow`

Install them with:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

## 🚀 How to Run

1. Place your `dataset.csv` in the same directory.
2. Run the Python script.

```bash
python ner_bilstm.py
```

---

## 📈 Result

- **Model Accuracy**: ~98%
- Suitable for basic Named Entity Recognition tasks using custom datasets.

---

## 💡 Future Improvements

- Use **pre-trained word embeddings** (like GloVe or Word2Vec)
- Add **CRF layer** for sequence-level predictions
- Hyperparameter tuning for better generalization