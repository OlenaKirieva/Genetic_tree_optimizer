# Hyperparameter Optimization for Decision Tree Classifier Using Genetic Algorithm and Random Search

This repository contains an empirical study of hyperparameter optimization methods for a **Decision Tree Classifier** applied to an imbalanced binary classification dataset.
Two optimization strategies are compared:

1. **Genetic Algorithm (GA)**
2. **Random Search (RS)**

The goal is to evaluate whether evolutionary optimization provides performance gains relative to classical randomized search under the same computational budget.

---

## **1. Dataset Description**

The target variable **"Exited"** is severely imbalanced:

| Class | Proportion |
| ----- | ---------- |
| 0     | 79.65%     |
| 1     | 20.35%     |

Correct detection of the minority class (1) is critical.

The data is split into three subsets:

```python
X_train = data['X_train']
train_targets = data['y_train']

X_val = data['X_val']
val_targets = data['y_val']

X_train, X_test, train_targets, test_targets = train_test_split(
    X_train, train_targets,
    test_size=0.2,
    stratify=train_targets,
    random_state=42
)
```

---

## **2. Hyperparameter Search Space**

The following parameter grid was used for both optimization methods:

```python
params_dt = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': np.arange(1, 20),
    'max_leaf_nodes': np.arange(2, 20),
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [None, 'sqrt', 'log2']
}
```

Total combinations: **65,664**, which makes exhaustive grid search impractical.

---

## **3. Methodology**

### **3.1 Genetic Algorithm**

The improved GA implementation includes:

* population initialization
* roulette-wheel selection
* single-point crossover
* mutation with individual gene perturbation
* elitism
* fitness function = ROC AUC on validation set

Population size: **10**
Generations: **10**
Total evaluated models ≈ **100**.

---

### **3.2 Random Search**

For a fair comparison, Random Search evaluates **100 random combinations**:

```python
random.shuffle(hyperparameter_combinations)
hyperparameter_combinations[:100]
```

Fitness function identical to GA.

---

## **4. Results**

### **4.1 Best Hyperparameters (Genetic Algorithm)**

```
criterion          : entropy
splitter           : best
max_depth          : 6
max_leaf_nodes     : 19
min_samples_split  : 5
min_samples_leaf   : 1
max_features       : None
Best Validation AUC: 0.9202
```

### **Test Performance**

```
Train AUC: 0.9219
Test  AUC: 0.9147
```

**Classification Report (Test):**

| Class | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| 0     | 0.913     | 0.950  | 0.931    |
| 1     | 0.766     | 0.643  | 0.699    |

---

### **4.2 Best Hyperparameters (Random Search)**

```
criterion          : entropy
splitter           : best
max_depth          : 16
max_leaf_nodes     : 19
min_samples_split  : 20
min_samples_leaf   : 4
max_features       : None
Best Validation AUC: 0.9193
```

### **Test Performance**

```
Train AUC: 0.9229
Test  AUC: 0.9126
```

**Classification Report (Test):**

| Class | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| 0     | 0.902     | 0.964  | 0.932    |
| 1     | 0.809     | 0.590  | 0.682    |

---

## **5. Discussion and Key Observations**

1. **Genetic Algorithm achieved slightly better validation AUC (0.9202)** compared to Random Search (0.9193).
2. Both models demonstrated **similar test AUC (≈0.914)**, confirming robustness.
3. The GA produced a **smaller and more interpretable tree** (`max_depth = 6`), whereas Random Search tended to select deeper structures (`max_depth = 16`).
4. The GA model had **higher recall for the minority class (1)**, which is important in imbalanced classification.
5. Given equal computational budgets (~100 models), **GA provides more efficient exploration** of the hyperparameter space.

---

## **6. Conclusion**

The study demonstrates that a well-designed Genetic Algorithm can serve as an efficient hyperparameter optimization technique for tree-based classifiers, achieving performance comparable to Random Search but producing simpler and more balanced models.
Under constrained computation, GA provides a competitive alternative to classical randomized approaches.

---

## **7. Repository Structure**

```
├── ml_utils.py               # ROC AUC plotting and evaluation functions
├── genetic_search.py         # Genetic Algorithm implementation
├── random_search.py          # Random Search pipeline
├── process_bank_churn.py     # Data preprocessing
├── notebooks/
│   └── experiments.ipynb     # Full workflow, experiments, plots
└── README.md                 # Project documentation
```

---

## **8. Requirements**

```
numpy
pandas
scikit-learn
matplotlib
seaborn
```

---

## **9. Usage**

Example of running GA:

```python
best_params, best_score = genetic_algorithm_improved(
    X_train, X_val, train_targets, val_targets, params_dt
)
```

Example of evaluating final model:

```python
final_model = DecisionTreeClassifier(random_state=42, **best_params)
final_model.fit(X_train, train_targets)
auroc_train_and_val(final_model, X_train, X_test, train_targets, test_targets)
```

---

Якщо хочете, можу:

✅ оформити README у Markdown-файл з форматуванням таблиць
✅ додати математичні формули (LaTeX)
✅ зробити графіки або схему роботи GA
✅ написати секцію “Limitations & Future Work”
✅ оформити варіант README українською мовою

Скажіть, чи потрібно доповнити.

