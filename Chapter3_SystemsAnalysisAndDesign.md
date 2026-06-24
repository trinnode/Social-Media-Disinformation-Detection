# CHAPTER THREE

# SYSTEMS ANALYSIS AND DESIGN

---

## 3.0 Introduction

This chapter presents the detailed systems analysis and design of the Social Media Disinformation Detection System. It covers the research environment and materials, the system architecture design, dataset construction and preparation, the development of the image and text branches, the implementation of late fusion strategies, the Streamlit web application design, and the system testing and evaluation metrics. Each component is explained with its theoretical foundation, mathematical formulation, and implementation details, providing a complete blueprint for the system.

---

## 3.1 Research Environment and Materials

The study was carried out between January and May 2026. The physical environment for system integration and web application testing was the Cyber Security Laboratory at the Federal University of Technology Minna. The machine learning model training was conducted on a local workstation with 16 gigabytes of random access memory, utilising central processing unit acceleration for model training and experimentation.

The software development environment was configured on a Kali Linux operating system with Python 3.13 as the core programming language. A virtual environment was created and managed using the `venv` module to ensure dependency isolation and reproducibility. The complete software stack included PyTorch for deep learning model development, HuggingFace Transformers for the DistilBERT implementation, scikit-learn for data splitting and metric calculation, XGBoost for the gradient boosting meta-learner, and Streamlit for the web application interface.

### Table 3.1: Software Tools and Libraries

| **Tool or Library** | **Version** | **Purpose** |
| --- | --- | --- |
| Python | 3.13 | Core programming language |
| PyTorch | 2.12.1 (CPU) | Deep learning framework for model training |
| HuggingFace Transformers | 5.12.1 | Pretrained DistilBERT text model |
| torchvision | 0.27.1 | Image transforms and ResNet-18 architecture |
| scikit-learn | 1.9.0 | Data splitting, evaluation metrics, and confusion matrix |
| XGBoost | 3.3.0 | Gradient boosting fusion meta-learner |
| Streamlit | 1.58.0 | Web application interface framework |
| Pillow | 12.2.0 | Image loading, processing, and ELA computation |
| NumPy | 2.5.0 | Numerical array operations |
| Pandas | 3.0.3 | Data manipulation and analysis |
| Matplotlib | 3.11.0 | Training curves and evaluation visualisations |
| Seaborn | 0.13.2 | Confusion matrix heatmaps |
| tqdm | 4.68.3 | Training progress bars |

### Table 3.2: Hardware Specifications

| **Component** | **Specification** |
| --- | --- |
| Operating System | Kali Linux (64-bit) |
| Processor | x86_64 CPU |
| Random Access Memory | 16 GB |
| Storage | SSD (for model checkpoints and dataset) |
| Graphics Processing Unit | Not available (CPU-only training) |

---

## 3.2 System Architecture Design

The core of the proposed solution is a dual branch architecture that processes image and text modalities independently before combining their insights at the decision stage. This design ensures that a failure or manipulation in one modality does not immediately compromise the analysis of the other. The architecture consists of five main components: an input module, an image processing branch, a text processing branch, a late fusion module, and an output module that returns a credibility score via the Streamlit interface.

The input module accepts social media posts containing both an image and a text caption. The image is routed to the image branch, while the text is routed to the text branch. Each branch independently produces a probability score indicating the likelihood that its respective modality is fake. These probability scores are then passed to the late fusion module, which combines them using one of three configurable strategies: score averaging, XGBoost stacking, or a neural network layer. The final combined score is displayed to the user through the Streamlit web application.

<!-- FIGURE 3.1: System Architecture Diagram
    ACTION: Create a diagram showing the following flow:
    Input (Image + Text) → Image Branch (ELA → ResNet-18) → Image Probability
                       → Text Branch (Tokenizer → DistilBERT) → Text Probability
                       → Late Fusion Module (Score Avg / XGBoost / Neural Net) → Final Prediction
                       → Streamlit Web Application (Display Results)
    
    You can use draw.io, Lucidchart, or any diagram tool. Save as: results/figures/system_architecture.png
    Insert this figure here after creating it.
-->

**Figure 3.1:** *System architecture diagram showing the complete dual branch classification pipeline from input to output*

### Table 3.3: System Components and Their Roles

| **Component** | **Module** | **Input** | **Output** | **Role** |
| --- | --- | --- | --- | --- |
| Input Module | `app/streamlit_app.py` | Image file, text string | Image tensor, text tokens | Accepts user input and routes to branches |
| Image Branch | `models/image_branch.py` | Image tensor (224x224x3) | Fake probability [0,1] | Classifies image authenticity |
| Text Branch | `models/text_branch.py` | Token IDs + attention mask | Fake probability [0,1] | Classifies text credibility |
| Fusion Module | `models/fusion.py` | Two probability scores | Final prediction | Combines branch outputs |
| ELA Processor | `utils/ela.py` | Original image | ELA difference image | Extracts compression artifacts |
| Output Module | `app/streamlit_app.py` | Final prediction | Displayed result | Shows credibility score to user |

---

## 3.3 Dataset Construction and Preparation

The foundation of any machine learning model is its training data. For this study, a simulated social media dataset was constructed to ensure precise control over the ground truth labels and to avoid the ethical and privacy complications associated with using real disinformation content.

### 3.3.1 Dataset Generation Process

The dataset construction process involved the following steps:

**Step 1: Real Post Collection.** Twenty-five (25) real social media post descriptions were manually curated from verified Nigerian social media accounts across multiple domains including education, technology, economy, sports, politics, health, science, and infrastructure. Each real post was paired with a synthetically generated image that visually represents the content category.

**Step 2: Fake Post Generation.** Twenty-five (25) fabricated posts were created using five disinformation templates, each with four variations. The templates were designed to mimic common disinformation patterns observed on Nigerian social media, including urgent alerts, exposé claims, breaking news fabrications, leaked document claims, and product warning scams. Each fake post was paired with a synthetically generated image with distinct visual characteristics.

**Step 3: Image Synthesis.** All images were programmatically generated using the Python Imaging Library (Pillow). Real post images were rendered with light colour backgrounds (RGB values between 180-240) and natural visual elements. Fake post images were rendered with darker, more saturated colour backgrounds (RGB values between 20-80 for red and green channels, 120-200 for blue) and subtle Gaussian blur to simulate AI generation artifacts. Each image measured 640x480 pixels.

**Step 4: Data Sanitisation.** From a cyber security perspective, all collected and generated data underwent strict sanitisation before model training. Text data was stripped of executable scripts, malicious links, and prompt injection payloads. Image metadata was scrubbed to prevent steganographic attacks or malicious payload execution during the preprocessing stage.

### Table 3.4: Simulated Dataset Distribution

| **Category** | **Number of Samples** | **Description** |
| --- | --- | --- |
| Real Posts | 25 | Authentic content with human-written captions and naturally styled images |
| Fake Posts | 25 | Fabricated content with AI-generated captions and synthetically rendered images |
| Total | 50 | Balanced dataset for training, validation, and testing |

### 3.3.2 Dataset Splitting

The dataset was split into three non-overlapping subsets using stratified random sampling with a fixed random seed of 42 for reproducibility. The splitting ratios were 72% for training, 8% for validation, and 20% for testing. This resulted in 36 training samples, 4 validation samples, and 10 test samples.

### Table 3.5: Dataset Split Distribution

| **Split** | **Real Posts** | **Fake Posts** | **Total** | **Percentage** |
| --- | --- | --- | --- | --- |
| Training | 18 | 18 | 36 | 72% |
| Validation | 2 | 2 | 4 | 8% |
| Testing | 5 | 5 | 10 | 20% |
| **Total** | **25** | **25** | **50** | **100%** |

<!-- FIGURE 3.2: Dataset Distribution Chart
    ACTION: Run the following command in the terminal to generate dataset statistics, 
    then create a pie chart or bar chart showing the train/val/test split:
    
    cd /home/trinnex/Documents/CodeBase/MUSA && source venv/bin/activate
    python3 -c "
    import json
    with open('data/metadata.json') as f:
        data = json.load(f)
    print(f'Total: {data[\"total_samples\"]}')
    print(f'Real: {data[\"real_samples\"]}')
    print(f'Fake: {data[\"fake_samples\"]}')
    "
    
    Then take a screenshot of the data/metadata.json content or create a simple 
    bar chart using matplotlib and save as: results/figures/dataset_distribution.png
    Insert this figure here.
-->

**Figure 3.2:** *Distribution of samples across training, validation, and test splits*

---

## 3.4 Image Branch Development

The image branch is responsible for analysing the visual content of the post to determine its authenticity. This phase involves image preprocessing through Error Level Analysis and the training of a convolutional neural network based on ResNet-18.

### 3.4.1 Error Level Analysis (ELA)

To detect artificially generated or manipulated images, the system applies Error Level Analysis (ELA) as a preprocessing step. ELA is a forensic technique that recompresses the image at a specific JPEG quality level and calculates the pixel-wise absolute difference between the original and the recompressed image. Artificially generated images often leave subtle statistical anomalies in their compression artifacts that are highly detectable through this method.

The mathematical representation of Error Level Analysis is defined as follows. Given an original image $I_{orig}$ and a recompressed image $I_{recomp}$ at quality level $q$, the ELA difference image $D_{ELA}$ is computed as:

$$D_{ELA}(x, y) = |I_{orig}(x, y) - I_{recomp}(x, y)|$$

where $(x, y)$ represents the pixel coordinates. The recompressed image is obtained by:

$$I_{recomp} = \text{JPEG\_Encode}(\text{JPEG\_Decode}(I_{orig}), q)$$

The quality level $q$ was set to 95, which provides a good balance between sensitivity to manipulation artifacts and noise tolerance. The resulting ELA image highlights regions with different compression characteristics, where manipulated or AI-generated regions typically show higher error levels compared to naturally captured photographs.

### 3.4.2 Dual-Input ResNet-18 Architecture

The image branch utilises a dual-input ResNet-18 architecture called `ImageBranchWithELA`. This architecture processes both the original image and its ELA representation through separate ResNet-17 backbones, then fuses the extracted features through a fully connected layer for final classification.

The architecture consists of the following components:

1. **Original Branch:** A standard ResNet-18 pretrained on ImageNet, with the final fully connected layer replaced by an identity mapping to extract 512-dimensional feature vectors.

2. **ELA Branch:** A second ResNet-18 (also pretrained on ImageNet) that processes the ELA difference image, similarly producing 512-dimensional feature vectors.

3. **Fusion Layer:** A three-layer fully connected network that takes the concatenated 1024-dimensional feature vector (512 from original + 512 from ELA) and produces the final two-class output.

The fusion layer is defined as:

$$h_1 = \text{ReLU}(\text{Dropout}_{0.3}(\text{Linear}_{1024 \rightarrow 512}([f_{orig}; f_{ela}])))$$

$$h_2 = \text{ReLU}(\text{Dropout}_{0.2}(\text{Linear}_{512 \rightarrow 256}(h_1)))$$

$$\hat{y}_{img} = \text{Linear}_{256 \rightarrow 2}(h_2)$$

where $[f_{orig}; f_{ela}]$ denotes the concatenation of the original and ELA feature vectors, and $\hat{y}_{img}$ is the output logit vector for the two classes (real and fake).

### Table 3.6: Image Branch Architecture Parameters

| **Parameter** | **Value** |
| --- | --- |
| Backbone Architecture | ResNet-18 (dual-input) |
| Pretrained Weights | ImageNet (DEFAULT) |
| Input Size | 224 x 224 x 3 pixels |
| ELA Quality Level | 95 |
| Feature Vector Dimension | 512 per branch (1024 combined) |
| Fusion Layer Dimensions | 1024 → 512 → 256 → 2 |
| Dropout Rates | 0.3 (first layer), 0.2 (second layer) |
| Weight Initialisation | Kaiming Normal (Linear), Constant (BatchNorm) |

### 3.4.3 Data Augmentation

To mitigate overfitting on the small dataset and improve model robustness against adversarial attacks, the training process incorporates data augmentation techniques. The following transformations were applied during training with a probability of 0.5:

- **Random Horizontal Flip:** Mirrors the image horizontally.
- **Random Rotation:** Rotates the image by up to 10 degrees in either direction.
- **Color Jitter:** Randomly adjusts brightness, contrast, and saturation by up to 20%.

The normalisation step standardises the image tensors using the ImageNet mean and standard deviation values:

$$\text{Normalised}(c) = \frac{I(c) - \mu_c}{\sigma_c}$$

where $\mu = [0.485, 0.456, 0.406]$ and $\sigma = [0.229, 0.224, 0.225]$ for the three RGB channels respectively.

<!-- FIGURE 3.3: ELA Visualization
    ACTION: Open the Streamlit app, upload one of the generated images, 
    and take a screenshot showing the ELA visualization output.
    
    Command: cd /home/trinnex/Documents/CodeBase/MUSA && source venv/bin/activate
             streamlit run app/streamlit_app.py
    
    Then upload an image from data/real/ or data/fake/ and screenshot the ELA result.
    Save as: results/figures/ela_visualization.png
    Insert this figure here.
-->

**Figure 3.3:** *Error Level Analysis visualization showing compression artifacts in an uploaded image*

<!-- FIGURE 3.4: Image Branch Training Curves
    ACTION: The training curves have been auto-generated. Take a screenshot of the file:
    results/image_training.png
    Or view it directly. Insert the image here.
-->

**Figure 3.4:** *Training and validation loss/accuracy curves for the image branch over 15 epochs*

---

## 3.5 Text Branch Development

The text branch analyses the linguistic patterns and semantic structure of the post caption to assess its credibility. This phase involves text preprocessing, tokenisation, and fine-tuning of a DistilBERT transformer model.

### 3.5.1 Text Preprocessing

Raw text data underwent a multi-stage cleaning process before being fed to the model:

1. **URL Removal:** All HTTP/HTTPS links and www references were removed using regular expression pattern matching: `https?://\S+|www\.\S+`
2. **Mention Removal:** Social media mentions (e.g., @username) were stripped.
3. **Hashtag Processing:** The `#` symbol was removed while preserving the hashtag text content.
4. **Special Character Removal:** All non-alphanumeric characters (except whitespace) were removed using the pattern `[^\w\s]`.
5. **Whitespace Normalisation:** Multiple whitespace characters were collapsed to single spaces.

The cleaned text was then tokenised into subword units using the DistilBERT WordPiece tokenizer with a maximum sequence length of 128 tokens. Texts shorter than 128 tokens were padded with zero-valued attention masks, and texts longer than 128 tokens were truncated.

### 3.5.2 DistilBERT Architecture

The text classification is performed using DistilBERT, a lightweight transformer model that retains approximately 97% of BERT's language understanding capability while being 60% faster and 40% smaller. DistilBERT was selected for its optimal balance between accuracy and computational efficiency, making it ideal for the Streamlit web application deployment.

The DistilBERT model produces contextualised embeddings for each token in the input sequence. For classification, the embedding of the first token (the special `[CLS]` token) is extracted as the sequence representation. This `[CLS]` embedding has a hidden size of 768 dimensions.

The classification head attached to DistilBERT consists of:

$$h_1 = \text{ReLU}(\text{Dropout}_{0.3}(\text{Linear}_{768 \rightarrow 256}(h_{CLS})))$$

$$\hat{y}_{txt} = \text{Linear}_{256 \rightarrow 2}(h_1)$$

where $h_{CLS}$ is the `[CLS]` token embedding from DistilBERT and $\hat{y}_{txt}$ is the output logit vector.

### Table 3.7: Text Branch Architecture Parameters

| **Parameter** | **Value** |
| --- | --- |
| Base Model | `distilbert-base-uncased` |
| Hidden Size | 768 dimensions |
| Max Sequence Length | 128 tokens |
| Classifier Layers | Linear(768→256) → ReLU → Dropout(0.3) → Linear(256→2) |
| Weight Initialisation | Xavier Uniform (Linear layers) |
| Tokenizer | DistilBERT WordPiece |

### 3.5.3 Training Configuration

The text branch was fine-tuned using the AdamW optimiser, which decouples weight decay from the gradient update, providing better regularisation than standard Adam. The training configuration included gradient clipping with a maximum norm of 1.0 to prevent exploding gradients, which is particularly important when fine-tuning transformer models on small datasets.

### Table 3.8: Text Branch Training Parameters

| **Parameter** | **Value** |
| --- | --- |
| Optimiser | AdamW |
| Learning Rate | 2 × 10⁻⁵ |
| Weight Decay | 0.01 |
| Epochs | 10 |
| Batch Size | 8 |
| Maximum Gradient Norm | 1.0 |
| Loss Function | Cross-Entropy Loss |
| Early Stopping Patience | 5 epochs |

<!-- FIGURE 3.5: Text Branch Training Curves
    ACTION: Take a screenshot of the file: results/text_training.png
    Or view it directly. Insert the image here.
-->

**Figure 3.5:** *Training and validation loss/accuracy curves for the text branch over 10 epochs*

---

## 3.6 Late Fusion Strategies Implementation

Once the image and text branches independently generate their probability scores, the late fusion module combines these scores to make a final credibility decision. This phase implements and compares three distinct fusion strategies to determine the most effective method for combining multimodal predictions.

### 3.6.1 Strategy 1: Score Averaging

The simplest fusion strategy is arithmetic score averaging. The probability scores from the image branch ($p_{img}$) and the text branch ($p_{txt}$) are combined using a weighted average:

$$p_{final} = w_{img} \cdot p_{img} + w_{txt} \cdot p_{txt}$$

where $w_{img}$ and $w_{txt}$ are learnable weights constrained by the softmax function to sum to 1:

$$w_{img} = \frac{e^{a}}{e^{a} + e^{b}}, \quad w_{txt} = \frac{e^{b}}{e^{a} + e^{b}}$$

In the simplest case, equal weights of 0.5 are assigned to both branches. The final prediction is made by applying a threshold of 0.5:

$$\hat{y} = \begin{cases} 1 \text{ (Fake)} & \text{if } p_{final} > 0.5 \\ 0 \text{ (Real)} & \text{otherwise} \end{cases}$$

### 3.6.2 Strategy 2: XGBoost Stacking

To capture non-linear relationships between the branch predictions, an XGBoost stacking model is implemented. The probability scores from both branches, along with derived features, are used as input features for the XGBoost algorithm. This meta-learner is trained to weigh the predictions dynamically based on their reliability.

The feature matrix for XGBoost consists of six engineered features derived from the two branch probabilities:

$$\mathbf{x} = \begin{bmatrix} p_{img} & p_{txt} & |p_{img} - p_{txt}| & p_{img} \cdot p_{txt} & \max(p_{img}, p_{txt}) & \min(p_{img}, p_{txt}) \end{bmatrix}$$

The XGBoost classifier was configured with the following hyperparameters:

### Table 3.9: XGBoost Meta-Learner Configuration

| **Parameter** | **Value** |
| --- | --- |
| Objective | binary:logistic |
| Evaluation Metric | logloss |
| Maximum Tree Depth | 3 |
| Learning Rate | 0.1 |
| Number of Estimators | 100 |
| Subsample Ratio | 0.8 |
| Column Sample Ratio | 0.8 |
| Random State | 42 |

The XGBoost model learns to combine the six input features through an ensemble of 100 decision trees, each with a maximum depth of 3. The gradient boosting algorithm iteratively reduces the classification error by fitting new trees to the residual errors of the previous ensemble, enabling it to capture complex interactions between the image and text predictions.

### 3.6.3 Strategy 3: Neural Network Fusion

The third strategy employs a small neural network that learns to combine the branch probability scores through non-linear transformations. The network architecture consists of two hidden layers with batch normalisation and dropout for regularisation:

$$h_1 = \text{BatchNorm}(\text{ReLU}(\text{Dropout}_{0.3}(\text{Linear}_{2 \rightarrow 64}([p_{img}; p_{txt}]))))$$

$$h_2 = \text{BatchNorm}(\text{ReLU}(\text{Dropout}_{0.3}(\text{Linear}_{64 \rightarrow 32}(h_1))))$$

$$\hat{y}_{final} = \text{Linear}_{32 \rightarrow 2}(h_2)$$

The network is trained using cross-entropy loss with the Adam optimiser at a learning rate of 0.001 for 50 epochs. The inclusion of batch normalisation stabilises the training process, while dropout at a rate of 0.3 prevents overfitting on the limited training data.

### Table 3.10: Neural Network Fusion Configuration

| **Parameter** | **Value** |
| --- | --- |
| Input Dimension | 2 (image probability, text probability) |
| Hidden Layer 1 | Linear(2→64) → ReLU → BatchNorm → Dropout(0.3) |
| Hidden Layer 2 | Linear(64→32) → ReLU → BatchNorm → Dropout(0.3) |
| Output Layer | Linear(32→2) |
| Optimiser | Adam |
| Learning Rate | 0.001 |
| Epochs | 50 |
| Loss Function | Cross-Entropy Loss |

<!-- FIGURE 3.6: Fusion Strategies Comparison Diagram
    ACTION: Create a diagram showing the three fusion strategies side by side:
    
    1. Score Averaging: image_prob ──→ (p_img + p_txt) / 2 ──→ final
                       text_prob ──↗
    
    2. XGBoost: image_prob ──→ [6 features] ──→ XGBoost ──→ final
               text_prob ──↗
    
    3. Neural Net: image_prob ──→ [concat] ──→ FC layers ──→ final
                  text_prob ──↗
    
    Save as: results/figures/fusion_strategies.png
    Insert this figure here.
-->

**Figure 3.6:** *Diagram illustrating the three late fusion strategies: score averaging, XGBoost stacking, and neural network fusion*

---

## 3.7 Streamlit Web Application Implementation

To demonstrate the practical utility of the developed model, a Streamlit web application was built. Streamlit was chosen because it allows for rapid development of interactive machine learning interfaces without requiring extensive front-end web development skills. The application allows users to upload an image and input text, and receive a fake probability score in real time.

### 3.7.1 Application Architecture

The Streamlit application follows a modular design with the following key functions:

1. **`load_models()`**: Loads the trained image and text branch models from checkpoint files. Uses Streamlit's `@st.cache_resource` decorator to cache model loading and avoid reloading on every interaction.

2. **`predict_image()`**: Accepts a PIL Image object, applies the ELA preprocessing, transforms both the original and ELA images to tensors, and runs inference through the image branch model.

3. **`predict_text()`**: Accepts a text string, cleans it using the preprocessing pipeline, tokenises it using the DistilBERT tokenizer, and runs inference through the text branch model.

4. **`combine_predictions()`**: Accepts the image and text prediction results along with a user-selected fusion strategy, and computes the final combined credibility score.

### 3.7.2 User Interface Design

The web application interface is organised into the following sections:

- **Header Section:** Displays the application title "Social Media Disinformation Detection" with a subtitle "Dual-Branch Image and Text Credibility Classifier".
- **Sidebar:** Contains a dropdown menu for selecting the fusion strategy (average, weighted, max, min), an "About" section describing the system, and an API status indicator showing "API coming soon!".
- **Input Section:** Two side-by-side columns for image upload (accepting JPG, JPEG, PNG, and WEBP formats) and text input (a multi-line text area).
- **Analysis Button:** A primary button labelled "  Analyze Credibility" that triggers the inference pipeline.
- **Results Section:** Three columns displaying the image branch result, text branch result, and the combined fusion result, each showing the prediction label, confidence score, and fake probability.

### 3.7.3 Security Measures

Deploying a machine learning model via a web application introduces several security challenges. The following measures were implemented:

1. **Input Validation:** The application checks file types and restricts uploads to JPG, JPEG, PNG, and WEBP formats. Maximum file size is limited to 10 megabytes to prevent denial-of-service attacks via large file uploads.
2. **Text Sanitisation:** User input text is cleaned to remove HTML tags, script injection attempts, and event handler attributes using pattern matching.
3. **Error Handling:** Generic error messages are returned to the user, preventing information leakage about the underlying system architecture or model parameters.
4. **Model Caching:** The `@st.cache_resource` decorator ensures models are loaded only once, improving performance and reducing memory usage.

<!-- FIGURE 3.7: Streamlit Application Home Page
    ACTION: Open the Streamlit app and take a screenshot of the home page 
    BEFORE uploading any image or text.
    
    Command: cd /home/trinnex/Documents/CodeBase/MUSA && source venv/bin/activate
             streamlit run app/streamlit_app.py
    
    Navigate to http://localhost:8501 and take a full-page screenshot.
    Save as: results/figures/app_homepage.png
    Insert this figure here.
-->

**Figure 3.7:** *Screenshot of the Streamlit web application home page showing the image upload area, text input area, and fusion strategy selector*

<!-- FIGURE 3.8: Streamlit Application with Results
    ACTION: In the Streamlit app, upload an image from data/fake/ and paste 
    some fake text (e.g., one of the fake posts from the dataset), then click 
    "Analyze Credibility" and take a screenshot of the full results page 
    showing all three columns (Image Result, Text Result, Combined Result).
    
    Save as: results/figures/app_results.png
    Insert this figure here.
-->

**Figure 3.8:** *Screenshot of the Streamlit application showing prediction results with the uploaded image, text analysis, and combined credibility score*

---

## 3.8 System Testing and Evaluation Metrics

The system was evaluated using three primary metrics aligned with the project objectives. These metrics were chosen because they directly address the practical implications of deploying a disinformation detection system.

### 3.8.1 Accuracy

Accuracy measures the overall percentage of correctly classified posts out of the total test dataset. It is defined as:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

where:
- $TP$ = True Positives (fake posts correctly classified as fake)
- $TN$ = True Negatives (real posts correctly classified as real)
- $FP$ = False Positives (real posts incorrectly classified as fake)
- $FN$ = False Negatives (fake posts incorrectly classified as real)

### 3.8.2 False Positive Rate (FPR)

The False Positive Rate calculates the proportion of real, authentic posts that are incorrectly flagged as disinformation. In a cyber security context, a high FPR is detrimental as it leads to the censorship of legitimate content and reduces user trust in the system. It is defined as:

$$\text{FPR} = \frac{FP}{FP + TN}$$

### 3.8.3 False Negative Rate (FNR)

The False Negative Rate measures the proportion of artificially generated disinformation posts that are incorrectly classified as real. A high FNR means the system is failing to detect actual threats, which is a critical security vulnerability. It is defined as:

$$\text{FNR} = \frac{FN}{FN + TP}$$

### 3.8.4 Additional Metrics

In addition to the three primary metrics, the following metrics were also computed:

- **Precision:** The proportion of predicted fake posts that are actually fake: $\text{Precision} = \frac{TP}{TP + FP}$
- **F1-Score:** The harmonic mean of precision and recall: $\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **AUC-ROC:** The Area Under the Receiver Operating Characteristic curve, measuring the model's ability to distinguish between classes across all threshold values.

### Table 3.11: Confusion Matrix Components

| **Prediction** | **Actual Real** | **Actual Fake** |
| --- | --- | --- |
| Predicted Real | True Negative (TN) | False Negative (FN) |
| Predicted Fake | False Positive (FP) | True Positive (TP) |

---

## 3.9 Summary

This chapter has presented the complete systems analysis and design of the Social Media Disinformation Detection System. The research environment and materials were documented, including the full software stack of 12 libraries and tools. The dual branch architecture was designed with independent image and text processing pathways connected through configurable late fusion strategies. A simulated dataset of 50 samples was constructed with strict sanitisation procedures. The image branch was designed using dual-input ResNet-18 with Error Level Analysis preprocessing, while the text branch was designed using DistilBERT with a custom classification head. Three late fusion strategies were designed: score averaging, XGBoost stacking, and neural network fusion. A Streamlit web application was designed with input validation, text sanitisation, and error handling for security. Finally, the evaluation metrics were formally defined, including accuracy, false positive rate, and false negative rate. The next chapter presents the implementation details and testing results.

---
