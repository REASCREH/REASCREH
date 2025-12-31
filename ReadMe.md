# 👋 Hi, I'm **Qamar Usman**
**Machine Learning Engineer | LLMs, Transformers, RAG | Deep Learning & Computer Vision | Time Series | Kaggle Competition Expert (Top 0.4%) | Data Science | Medical Healthcare AI**

---

## 💫 About Me  
I am a research-driven **Machine Learning Engineer** with a strong foundation in **Mathematics**, specializing in:

- Large Language Models (LLMs)  
- Retrieval-Augmented Generation (RAG)  
- Time Series Forecasting  
- Computer Vision  
- Medical & Healthcare AI  
- Deep Learning and Transformer-based architectures  

My work combines **rigorous ML engineering**, **research experimentation**, and **practical deployment** of real-world AI systems.  
I currently work at **VFIXALL**, building end-to-end production ML pipelines.

---

## 🧪 **Research & Project Highlights**
# 🔬 Medical & Healthcare AI

## 🧠 Child Mind Institute: Problematic Internet Use Prediction
**Kaggle Silver Medal (Top 3%) | Rank: 76 / 3,559 Teams**

### 📌 Overview
Developed a machine learning solution for the **Child Mind Institute** to detect early signs of problematic internet usage in youth. This research uses accessible **physical activity and fitness data** as proxies for mental health indicators, bypassing traditional clinical barriers to enable early intervention for depression and anxiety.

### 🛠️ Technical Approach
* **Model:** LightGBM Regressor with a 7-Fold Stratified K-Fold validation strategy.
* **Optimization:** Utilized the **Nelder-Mead method** for precise threshold tuning to maximize the Quadratic Weighted Kappa (QWK) score.
* **Performance:** Achieved a **Final QWK of 0.463**, proving high generalization on unseen fitness data.

### 🔗 Resources
* **Kaggle Notebook:** [View Full Implementation & Research](https://www.kaggle.com/code/qamarmath/fork-of-handling-overfitting-val-qwk-0-457-4c34cf)
# 🧬 Structural Bioinformatics & Deep Learning

## 🧪 Stanford RNA 3D Folding Challenge
**Kaggle Silver Medal (Top 4%) | Rank: 57 / 1,516 Teams**

### 📌 Project Overview
Successfully solved one of biology’s "grand challenges": predicting the **3D atomic coordinates of RNA molecules** from their primary sequences. This project aims to accelerate RNA-based medicine, including cancer immunotherapies and CRISPR gene editing, by illuminating the "dark matter of biology"—the folds and functions of natural RNA.



### 🛠️ Technical Architecture
Developed a dual-stage neural network pipeline for high-fidelity structure generation:
* **RNA Language Model (RNA2nd):** An 18-layer encoder transformer that captures hidden sequence patterns across up to 2,400 nucleotides.
* **Structure Prediction (MSA2XYZ):** A multi-cycle refinement model that converts sequence embeddings into global 3D coordinates for key atoms (P, C4', N1/N9).
* **Ensemble Strategy:** Integrated up to 20 different model versions to ensure robust structural diversity and accuracy.

### ⚙️ Optimization & Physics
* **Segmented Prediction:** Implemented overlapping chunking for long sequences (>480 nt) using mathematical transformations for stitching.
* **Energy-Based Refinement:** Utilized **OpenMM** to calculate bond, angle, and stacking energies, ensuring the predicted folds are thermodynamically stable.
* **Evaluation:** Optimized for the **TM-score** metric, focusing on global topology rather than local errors.

### 🔗 Resources
* **Kaggle Notebook:** [View Silver Medal Implementation](https://www.kaggle.com/code/qamarmath/rna-3d-structure-prediction-project)
  
# 🔬 Medical & Healthcare AI

## 🧠 HMS: Harmful Brain Activity Classification
**Top 11% | Rank: 312 / 2,767 Teams**

### 📌 Project Overview
Developed a deep learning pipeline to automate the detection and classification of seizures and other harmful brain activity patterns from **Electroencephalography (EEG) signals**. This research aims to remove the manual review bottleneck for critically ill patients, enabling faster neurocritical care and more accurate drug development for epilepsy.

### 🛠️ Technical Approach
* **Model Architecture:** Implemented a **ResNet18d** backbone (via the `timm` library) modified to handle single-channel spectrogram data.
* **Signal Processing:** * Converted raw EEG/spectrogram data into normalized logarithmic representations.
    * Applied custom image transformations (512x512 resizing) for standardized deep learning input.
* **Loss Function:** Utilized **Kullback-Leibler (KL) Divergence** to handle the "soft labels" provided by expert neurologist votes, accounting for cases where clinical experts disagree.
* **Training Strategy:** 5-Fold Cross-Validation with a **Cosine Annealing Learning Rate Scheduler** over 9 epochs to ensure model convergence and stability.

### 📊 Key Results
* **Best Fold Test Loss:** 0.56 (KL Divergence).
* **Classification Targets:** Seizure (SZ), LPD, GPD, LRDA, GRDA, and "Other".
* **Impact:** The model effectively differentiates between "idealized" patterns and complex "edge cases" where expert agreement is split.

### 🔗 Resources
* **Kaggle Competition:** [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/c/hms-harmful-brain-activity-classification)
* **Dataset Context:** Annotated by specialized neurologists for neurocritical care applications.
# 🔬 Medical & Healthcare AI

## 🏥 CIBMTR: Equity in post-HCT Survival Predictions
**Top 10% | Rank: 341 / 3,325 Teams**

### 📌 Project Overview
Developed advanced predictive models for allogeneic Hematopoietic Cell Transplantation (HCT) survival rates for the **Center for International Blood and Marrow Transplant Research (CIBMTR)**. The primary focus was ensuring **equitable outcomes** by reducing predictive bias across different racial groups, socioeconomic statuses, and geographies to rebuild trust in the healthcare system.

### 🛠️ Technical Approach
* **Model Ensemble:** Engineered a robust ensemble system combining **XGBoost**, **CatBoost**, and **LightGBM** to capture diverse feature interactions while maintaining stability.
* **Fairness Metric:** Optimized for the **Stratified Concordance Index (C-index)**, a specialized metric calculated as the `Mean - Standard Deviation` of C-indices across racial groups. This penalized the model if it performed significantly better for one race than another.
* **Data Processing:** Managed complex synthetic datasets that mirror real-world clinical disparities, ensuring privacy-compliant survival analysis.

### 📊 Performance Summary
The Ensemble model achieved the highest stability and generalization, effectively balancing performance across diverse patient demographics.

| Model | Val RMSE | Val MSE | Status |
| :--- | :--- | :--- | :--- |
| **LGBM** | 0.2790 | 0.0778 | Slight Overfitting |
| **XGBoost** | 0.2765 | 0.0764 | High Generalization |
| **CatBoost** | 0.2765 | 0.0764 | High Generalization |
| **Ensemble (Final)** | **0.2757** | **0.0760** | **Best Stability** |



### 🔗 Resources
* **GitHub Repository:** [Survival-Prediction-1](https://github.com/REASCREH/Survival-Prediction-1)
* **Kaggle Competition:** [CIBMTR - Equity in post-HCT Survival Predictions](https://www.kaggle.com/c/cibmtr-equity-in-post-hct-survival-predictions)
* # 🔬 Medical & Healthcare AI

## 🛡️ Skin Cancer Classification: EfficientNet-B0 Implementation
**State-of-the-Art Diagnostic Performance | 96.59% AUROC**

### 📌 Project Overview
Developed a high-precision deep learning system to classify skin lesions as **Malignant** or **Benign**. By synthesizing two major medical imaging repositories, this project addresses the critical need for early skin cancer detection, providing a robust tool that balances high sensitivity for malignancy with a high specificity to reduce unnecessary biopsies.

### 🛠️ Technical Architecture
* **Core Model:** **EfficientNet-B0** (pre-trained on ImageNet) utilizing compound scaling for optimal efficiency and accuracy.
* **Fine-Tuning Strategy:** Employed Transfer Learning by freezing early layers and fine-tuning the last three blocks alongside a custom-built classifier head.
* **Robust Pipeline:**
    * **Imbalance Handling:** Utilized a **weighted BCE loss (pos_weight=1.14)** and Stratified K-Fold validation to mitigate medical data skew.
    * **Augmentation:** Leveraged the `albumentations` library for geometric and color-space regularization (Flips, Rotations, Brightness/Contrast).
    * **Mixed Precision:** Trained on Tesla T4 GPUs using FP16 to reduce VRAM usage and speed up convergence.

### 📊 Performance Metrics (Test Set: 2,660 Images)
| Metric | Value | Clinical Significance |
| :--- | :--- | :--- |
| **AUROC** | **96.59%** | Exceptional class discrimination power |
| **Accuracy** | **90.19%** | High overall diagnostic reliability |
| **Recall** | **88.54%** | Minimizes missed malignant cases (Type II Error) |
| **Specificity** | **91.76%** | Reduces false alarms/unnecessary biopsies (Type I Error) |

### ⚕️ Clinical Implications
With a specificity of **91.76%**, this model acts as a powerful decision-support tool, potentially saving costs and patient anxiety by accurately ruling out benign cases while maintaining high sensitivity for life-threatening melanoma.

### 🔗 Resources
* **GitHub Repository:** [EfficientNet-B0-Skin-Cancer](https://github.com/REASCREH/EfficientNet-B0-Achieve-0.965-AUROC-in-Skin-cancer)
* **Live Demo:** [Streamlit Web Application](https://github.com/REASCREH/EfficientNet-B0-Achieve-0.965-AUROC-in-Skin-cancer/blob/main/mainapp.py)
* **Training Research:** [Kaggle Documentation](https://www.kaggle.com/code/qamarmath/efficientnet-b0-achieve-0.965-auroc-in-skin-cancer)
* # 🔬 Medical & Healthcare AI

## 🫁 Pneumonia Detection via Custom CNN
**High-Precision Diagnostic Tool | 94.01% Test Accuracy**

### 📌 Project Overview
Developed an automated diagnostic system to identify Pneumonia in chest X-ray images. This project addresses clinical bottlenecks by merging three major Kaggle datasets into a high-volume, diverse training corpus. The resulting model provides a rapid, secondary screening tool to assist radiologists in resource-limited environments.

### 🛠️ Technical Architecture
* **Custom CNN Design:** Engineered a deep sequential architecture with 5 Convolutional blocks, featuring **Batch Normalization** and **Dropout (up to 0.3)** to ensure stable learning and prevent overfitting.
* **Data Engineering:** * **Multi-Source Fusion:** Synthesized data from three distinct repositories, cleaning and isolating Pneumonia vs. Normal cases (excluding COVID-19/TB for specific task focus).
    * **Preprocessing:** Standardized grayscale pipeline at 150x150 resolution with ImageNet-style normalization.
    * **Augmentation:** Utilized `ImageDataGenerator` for real-time geometric shifts, simulating variations in patient positioning.
* **Training Dynamics:** Optimized using **RMSprop** and a **ReduceLROnPlateau** scheduler to fine-tune weights as validation accuracy plateaued.

### 📊 Performance Summary (Test Set: 2,420 Images)
| Metric | Value | Clinical Significance |
| :--- | :--- | :--- |
| **Accuracy** | **94.01%** | Highly reliable screening performance |
| **Precision (Pneumonia)** | **0.96** | Very low false-alarm rate |
| **Recall (Pneumonia)** | **0.95** | Critical for ensuring infected cases are not missed |
| **F1-Score** | **0.94** | Balanced performance across both classes |

### 🚀 Production Deployment
* **Backend:** Integrated the model into a **FastAPI** application for high-performance inference.
* **Web Interface:** Includes a user-friendly frontend for instant X-ray uploads and prediction results.
* **Reproducibility:** Fully documented pipeline for retraining or hyperparameter modification via Kaggle environments.

### 🔗 Resources
* **GitHub Repository:** [Pneumonia-Detection-CNN](https://github.com/REASCREH/Pneumonia-Detection-via-CNN-94-Test-Accuracy)
* **Training Research:** [Kaggle Notebook](https://www.kaggle.com/code/qamarmath/pneumonia-detection-via-cnn-94-test-accuracy)
* **Inference API:** Served via `uvicorn app:app`
* # 🔬 Medical & Healthcare AI

## 🏥 Pediatric Sepsis Early Detection
**PHEMS Hackathon | Advanced Predictive Modeling**

### 📌 Project Overview
Sepsis in children is a rapid, life-threatening emergency. This project involved building a machine learning algorithm to predict sepsis onset **6 hours prior to clinical diagnosis**. Utilizing a massive dataset of **331,639 time points** from 2,649 patients, the model provides healthcare providers with the lead time necessary for life-saving interventions like fluid resuscitation and antibiotics.

### 🛠️ Technical Approach
* **Handling Extreme Imbalance:** Managed a severe class imbalance (only 2.07% sepsis cases) through a strategic **Undersampling Strategy**, forcing the model to learn rare sepsis-specific signatures without bias toward the majority class.
* **Feature Engineering Pipeline:**
    * **Drug Exposure (TF-IDF):** Treated daily medication logs as "documents" to create 200 informative features, capturing rare antibiotic usage as a signal for high-risk infections.
    * **Temporal Dynamics:** Extracted diurnal (hour of day) and seasonal cycles to account for hospital workflow variations and infection trends.
* **Model Selection:** Deployed **XGBoost** for its robust handling of tabular clinical codes and native regularization, ensuring high performance on sensitive medical data.
* **Validation Strategy:** Employed **Stratified Group K-Fold Cross-Validation** to ensure patient independence—guaranteeing that the model generalizes to new patients it has never seen.

### 📊 Performance Results
| Metric | Value | Clinical Interpretation |
| :--- | :--- | :--- |
| **PR-AUC** | **0.9675** | Exceptional performance on highly imbalanced clinical data |
| **Accuracy** | **91 - 96%** | High overall reliability across multiple folds |
| **F1-Score** | **0.91 - 96%** | Balanced precision/recall; minimizes dangerous false negatives |



### ⚕️ Clinical Impact
The high **PR-AUC (0.9675)** indicates that the model is extremely effective at ranking high-risk patients. By balancing precision and recall, the tool minimizes "alert fatigue" for clinicians while ensuring that almost no sepsis cases go undetected during the critical 6-hour pre-diagnosis window.

### 🔗 Resources
* **GitHub Repository:** [Early-Sepsis-Detection-Model](https://github.com/REASCREH/Early-Sepsis-Detection-Model)
* **Hackathon Platform:** [PHEMS Hackathon: Pediatric Sepsis Prediction](https://www.kaggle.com/competitions/phems-hackathon-early-sepsis-prediction)

* # 🤖 Large Language Models & RAG (Retrieval-Augmented Generation)
# 🤖 Large Language Models (LLMs) & Data Privacy

## 🛡️ PII Detection in Student Writing (NER)
**Automated Data Anonymization for Educational Science**

### 📌 Project Overview
The massive growth of educational data offers huge potential for learning science, but student privacy (PII) remains a major barrier. This project developed a **Named Entity Recognition (NER)** system using LLMs to detect and remove sensitive information—such as student names, emails, and phone numbers—while distinguishing them from non-sensitive data like cited authors.

### 🛠️ Technical Architecture
* **Model Ensemble Strategy:** Integrated a "Piiranha" ensemble of three specialized **DeBERTa-v3** models (`cola`, `cuerpo`, and `cabeza`).
* **Weighted Inference:** Applied a **Softmax Weighted Average** approach to combine predictions, significantly boosting the model's reliability across diverse writing styles.
* **Token-Level Classification:**
    * **Custom Tokenization:** Developed a robust mapping system to align model sub-tokens back to original document words, handling complex trailing whitespaces and line breaks.
    * **Heuristic Refinement:** Implemented a high-confidence threshold (**0.99**) for the "O" (Outside) class to prioritize recall—ensuring that sensitive data is rarely missed.
* **Optimization:** Fine-tuned pre-trained transformers using the `Trainer` API with **Mixed Precision (FP16)** and **Stratified K-Fold** validation.

### 📊 Performance & Evaluation
* **Metric:** Optimized for **Micro F5-Score**, which weights **Recall 5x more heavily than Precision**. This ensures the highest level of student safety by minimizing False Negatives.
* **Entities Detected:** Student Names (B/I), Emails, ID Numbers, URLs, Phone Numbers, and Street Addresses.

### ⚙️ Implementation Roadmap
1. **Preprocessing:** Converted raw JSON essays into `Dataset` objects with character-to-token offset mapping.
2. **Parallel Tokenization:** Used multi-processing to handle large-scale inference across 3,500+ token documents.
3. **Post-Processing:** Filtered "O" predictions and converted predicted indices into human-readable B-I-O triplets for final CSV reporting.

### 🔗 Resources
* **Kaggle Notebook (Insemble):** [Ensemble Prediction & Exploitation](https://www.kaggle.com/code/qamarmath/ensemble-model-prediction-with-comprehensive-explo)
* **Training Research:** [BERT Fine-Tuning for Robust NER](https://www.kaggle.com/code/qamarmath/exploratory-data-analysis-for-robust-named-entit)
* **Dataset:** Vanderbilt University / The Learning Agency Lab
# 📐 Math Misconception Classification
**High-Precision Educational Diagnostic Tool | MAP@3: 0.9428**

## 📌 Project Overview
Developed a diagnostic pipeline to identify why students make specific mathematical errors. Using the Ettin-Encoder-400M, the system classifies natural language student explanations into **65 distinct pedagogical misconception categories**, enabling automated and personalized feedback at scale.

## 🛠️ Technical Architecture
- **Specialized Foundation**: Utilized `jhu-clsp/ettin-encoder-400m`, a model pre-trained specifically on mathematical and scientific text to handle technical logic more effectively than general LLMs.
- **Contextual Prompting**: Structured inputs by concatenating:
- - **Feature Engineering**: Automated the extraction of the correct answer key from training data to generate a binary `is_correct` logic feature.
- **Optimization**: Fine-tuned for 3 epochs using FP16 Mixed Precision on dual P100 GPUs with a max token length of 256.

## 📊 Performance & Metrics
The model was optimized for **Mean Average Precision at 3 (MAP@3)**, ensuring the correct pedagogical reason is prioritized in the top 3 suggestions provided to educators.

| Training Step | Training Loss | Validation Loss | MAP@3 (Accuracy) |
| :------------ | :------------ | :-------------- | :--------------- |
| 1000          | 0.8310        | 0.4457          | 0.9183           |
| 3000          | 0.6015        | 0.3522          | 0.9376           |
| **5400 (Final)** | **0.3441**    | **0.4778**      | **0.9428**       |

## ⚙️ Implementation Pipeline
1. **EDA**: Analyzed token distributions to set a 256-token limit, capturing 100% of student data without truncation.
2. **Fine-Tuning**: Implemented a classification head with 65 output nodes using the HuggingFace Trainer API.
3. **Inference**: Developed a robust prediction loop that decodes multi-class logits into human-readable pedagogical labels for final deployment.

## 🔗 Resources
- **Kaggle Competition**: [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/c/math-misconceptions)
- **Foundation Model**: [Ettin-Encoder-400M on HuggingFace](https://huggingface.co/jhu-clsp/ettin-encoder-400m)
- # Chat with PDFs or Websites 📚💬
**AI-Powered Document Analysis Tool** – Upload PDFs or input website URLs to query content using Gemini, Gemma, or OpenAI models. Features semantic search, chat history, and local processing for privacy. Built with Streamlit, LangChain, and FAISS.  

**Transform document interaction** with intelligent Q&A from multiple sources. Extract insights, analyze research, or review content through natural conversation. Maintains full chat history and supports various embedding models for accurate, context-aware responses.

**[🔗 Live Demo](https://chatwebpdf.streamlit.app/)** | **[📂 GitHub Repository](https://github.com/REASCREH/llm)**
# Chat with Your Data 📊💬  
**AI-Powered Data Analysis Tool** – Interact with CSV, Excel, and SQL data using natural language queries. Upload datasets, ask questions in plain English, and get instant insights powered by Google Generative AI. Built with Streamlit and LangChain.  

**Transform data exploration** with conversational analytics—no SQL or coding required. Export full chat histories as PDFs for easy documentation and sharing. Perfect for analysts, researchers, and teams needing quick, intuitive data insights.  

**[🔗 Live Demo](https://chatwithdatacsv.streamlit.app/)** | **[📂 GitHub Repository](https://github.com/REASCREH/csv-chat)**

#⏰  Time Series Forecasting & Data Analysis

# Comprehensive E-commerce Data Analysis and Customer Segmentation 📊🔍
**End-to-End EDA, RFM, Cohort & Deep Learning** – Analyze transaction data, segment customers using RFM modeling, conduct cohort analysis, and build predictive models. Features time-series decomposition and customer lifetime value prediction using Keras/TensorFlow.

**Uncover business insights** through multi-dimensional analysis of marketing, sales, and customer behavior data. Includes detailed statistical analysis, advanced visualizations with Plotly, and neural network implementation for predictive analytics.

**[📂 GitHub Repository](https://github.com/REASCREH/Comprehensive-E-commerce-Data-Analysis-and-Customer-Segmentation-EDA-RFM-Cohort)**
# Rohlik Orders Forecasting Challenge 🛒📈  
**Time Series Forecasting for E-commerce Operations** – Predict daily order volumes for online grocery service Rohlik with 3.37% MAPE accuracy using XGBoost. Features advanced feature engineering, cyclical encoding, and TF-IDF holiday analysis.

**Solve operational planning challenges** by accurately forecasting demand across 7 European warehouses. Includes comprehensive EDA, feature importance analysis, and a Streamlit web app for interactive predictions. Helps optimize inventory, staffing, and logistics planning.

**[🔗 Live App](https://rohlik-orders-forecasting-challenge-htxwnjy5vm3sjudwkhsz5t.streamlit.app/)** | **[📂 GitHub Repository](https://github.com/REASCREH/Rohlik-Orders-Forecasting-Challenge)** | **[📊 Kaggle Notebook](https://www.kaggle.com/code/qamarmath/fork-of-rohlik-orders-forecasting-xgboost-with-da)**

## 🎯 Problems Solved
- **Demand Forecasting**: Accurate prediction of daily orders across multiple warehouses
- **Seasonality Management**: Capture weekly, monthly, and holiday patterns in grocery ordering
- **Operational Efficiency**: Optimize inventory, staffing, and delivery logistics
- **Business Planning**: Provide reliable forecasts for financial and strategic planning
- 
# Automated Machine Learning Classification and Regression Application 🤖⚡  
**Streamlit AutoML Platform** – Automated end-to-end ML pipeline for classification and regression tasks. Supports XGBoost, LightGBM, CatBoost with automatic preprocessing, hyperparameter tuning, and performance evaluation.

**Simplify complex ML workflows** with automated EDA, missing value handling, categorical encoding, and Bayesian optimization. Perfect for data scientists, analysts, and students seeking rapid model development and deployment without coding overhead.

**[🔗 Live App](https://automated-ml.streamlit.app/)** | **[📂 GitHub Repository](https://github.com/REASCREH/Automated-ml)**

## 🎯 Problems Solved
- **Workflow Automation**: Eliminate repetitive data preprocessing and model tuning tasks
- **Accessibility**: Make machine learning accessible to non-technical users and beginners
- **Time Efficiency**: Reduce model development time from hours to minutes
- **Best Practices**: Enforce proper ML workflows with automated validation and metrics
- **Reproducibility**: Ensure consistent results with documented preprocessing steps
- **Hyperparameter Optimization**: Automatically find optimal model parameters using Bayesian methods
# Data Visualization and EDA (Exploratory Data Analysis) 📊🔍  
**Automated Data Analysis Platform** – Perform comprehensive EDA, generate interactive visualizations, and create professional PDF reports from CSV, Excel, and SQL files. Features time series decomposition, missing value analysis, and statistical summaries.

**Transform raw data into actionable insights** with an intuitive no-code interface. Perfect for analysts, researchers, and educators seeking rapid data exploration and presentation-ready reports without programming overhead.

**[🔗 Live App](https://visualizationeda.streamlit.app/)** | **[📂 GitHub Repository](https://github.com/REASCREH/Visualization-)**

## 🎯 Problems Solved
- **Accessibility**: Enable non-technical users to perform sophisticated data analysis
- **Time Efficiency**: Automate repetitive EDA tasks and visualization generation
- **Reporting**: Generate comprehensive PDF reports for stakeholders and presentations
- **Consistency**: Standardize data exploration workflows across teams and projects
- **Interactive Exploration**: Facilitate data discovery through multiple visualization types
- **Data Quality**: Quickly identify missing values, outliers, and data quality issues

## 🛠️ Tech Stack  

**Languages & Frameworks:**  
`Python` · `TensorFlow` · `PyTorch` · `Keras` · `Transformers` · `FastAPI`  
`Scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost`

**ML Domains:**  
LLMs · RAG · NLP · Computer Vision · Deep Learning  
Time Series Forecasting · Medical AI · AutoML · EDA

**Tools:**  
Git · GitHub Actions · Streamlit · Docker · MLflow · Anaconda  
NumPy · Pandas · Matplotlib · Plotly · SciPy  

