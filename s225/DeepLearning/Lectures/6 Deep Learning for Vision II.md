
***

# Deep Learning: Week 6 Study Sheet
## Network Architectures, Visualization, and Robustness

**Main Topics Covered:**
1.  Historical Context & The Rise of Deep Learning
2.  Traditional ML vs. Deep Learning Pipeline
3.  Evolution of CNN Architectures
4.  Visualization and Interpretability
5.  Adversarial Attacks and Robustness

---

### 1. Historical Context & The Rise of Deep Learning

*   **Key Milestones:**
    *   **Backpropagation:** Proposed in 1986.
    *   **CNN:** Proposed by Yann LeCun in 1995 (LeNet-5).
    *   Pre-2012, neural networks were not widely believed in due to limited data and compute.
*   **The Catalysts for Change (circa 2012):**
    *   **Data:** The introduction of large-scale datasets like **ImageNet** (~1 million images) by Fei-Fei Li's lab provided the necessary fuel for deep models.
    *   **Hardware:** The shift from CPUs to **GPUs** (Graphical Processing Units). Initially designed for gaming, their massively parallel architecture was perfectly suited for the matrix operations in neural network training. NVIDIA's pivot to scientific computing (AI) was a key enabler.
    *   **Algorithm:** Efficient training of deep networks via backpropagation on GPUs.
*   **The "Old Way" (Pre-Deep Learning):**
    *   **Computer Vision** was dominated by **handcrafted features** (e.g., HOG - Histogram of Oriented Gradients, SIFT - Scale-Invariant Feature Transform).
    *   A **disconnected two-step pipeline:**
        1.  **Feature Extraction:** Manually design algorithms to extract relevant features from images.
        2.  **Classification:** Feed these feature vectors into a separate, shallow model like an **SVM (Support Vector Machine)** or **Gradient Boosting** machine.
    *   **Limitation:** No joint optimization. The feature extractor is not tuned to improve the classifier's performance, and vice-versa.

---

### 2. Traditional ML vs. Deep Learning Pipeline

*   **Deep Learning Pipeline (End-to-End Learning):**
    *   **Feature Extraction:** This is done by the **convolutional layers** of a CNN. The network *learns* the optimal features directly from the raw pixel data.
    *   **Classification:** This is done by the **fully connected (FC) layers** (or MLP) at the end of the network.
    *   **Key Advantage: Joint Optimization.** The entire network—both convolutional and FC layers—is trained together using backpropagation and a single loss function (e.g., Cross-Entropy). This allows the feature extractor to learn representations that are optimally suited for the classification task, all within a unified "manifold" or optimization space.

---

### 3. Evolution of CNN Architectures

The goal has been to build deeper, more powerful, and more efficient networks.

*   **AlexNet (2012):** The breakthrough model that won ImageNet. Used larger convolutional filters (e.g., 11x11, 7x7) and proved the effectiveness of GPUs for training.
*   **VGGNet (2014):** Introduced the idea of building networks with stacks of small **3x3 convolutional filters**. A 3x3 conv layer has fewer parameters than a larger filter and allows for building very deep networks.
*   **GoogleNet / Inception (2014):** Introduced the **Inception module**. Key ideas:
    *   **Multi-Scale Processing:** Instead of choosing one filter size, the module uses multiple parallel paths with 1x1, 3x3, and 5x5 convolutions, and pooling. This allows the network to capture patterns at different scales simultaneously.
    *   **1x1 Convolutions:** Used for **dimensionality reduction** (cheaply reducing the number of feature maps) before expensive 3x3 and 5x5 convs, making the module computationally efficient.
    *   **Bypass Connections:** Early inspiration for residual connections.
*   **ResNet (2015):** Solved the **vanishing gradient** problem in very deep networks (>100 layers) using **Skip Connections** (or **Residual Connections**).

    *   **The Problem:** In very deep plain networks, gradients become extremely small as they are backpropagated, preventing early layers from learning effectively.
    *   **The Solution: Residual Block.**
        *   **Formula:** `Output = F(x) + x`
        *   `x` is the input to the block (the skip connection).
        *   `F(x)` is the output of the main path (typically two 3x3 conv layers with Batch Norm and ReLU).
        *   The network learns the *residual* `F(x) = Output - x`, which is often an easier function to learn.
    *   **Why it works:** The skip connection provides an unimpeded path for the gradient to flow directly backwards, mitigating the vanishing gradient problem. It also allows the network to be an "identity mapper" by easily setting `F(x)` to zero if needed.

*   **Modern Trends:**
    *   **Global Average Pooling (GAP):** Replaces large FC layers at the end. Instead of flattening the feature map, it takes the average of each feature map, resulting in a single vector. Drastically reduces parameters and helps prevent overfitting.
    *   **Increasing Depth & Width:** Modern architectures progressively reduce spatial size (height/width) while increasing the number of feature maps (channels), learning richer representations.

---

### 4. Visualization and Interpretability

Understanding *what* a network has learned and *why* it makes a certain prediction.

*   **Visualizing Filters/Kernels:**
    *   The first-layer filters often learn low-level features like edge detectors, color contrasts, and Gabor filters.
    *   Higher-layer filters combine these to represent more complex, abstract patterns (e.g., eyes, textures, object parts).
*   **Visualizing Feature Maps:** Showing the output (activation) of intermediate layers for a given input image reveals what patterns the layer is detecting and where they are located.
*   **Class Activation Mapping (CAM) & Grad-CAM:**
    *   **Goal:** Create a heatmap that highlights the regions of the input image that were most **important** for the network's prediction.
    *   **How (Grad-CAM):**
        1.  Take the final convolutional feature maps.
        2.  Compute the gradient of the class score (e.g., "cat") with respect to these feature maps. This tells us how important each feature map is for the class "cat".
        3.  Perform a weighted combination of the feature maps using these gradient-based importance weights.
        4.  Apply a ReLU to only keep features that have a positive influence on the class.
        5.  Upsample the result to overlay the heatmap on the original image.
    *   **Limitation:** The heatmap can be low-resolution (as it's based on conv features) and may not capture very fine-grained details.

---

### 5. Adversarial Attacks and Robustness

A critical vulnerability of deep learning models.

*   **What is an Adversarial Attack?**
    *   An attacker adds a tiny, carefully crafted perturbation (noise) `η` to an input image `x` to create an **adversarial example** `x_adv = x + η`.
    *   To a human, `x` and `x_adv` look identical.
    *   However, the model's prediction for `x_adv` is completely different and wrong (e.g., a "panda" is classified as a "gibbon").
*   **Why does this happen?**
    *   Models learn decision boundaries in a high-dimensional space. The data we train on is only a sparse sampling of this space.
    *   An attacker can find directions where the decision boundary is very close to a data point. A tiny step across this boundary in a direction humans are insensitive to can cause a misclassification.
*   **Real-World Implications:**
    *   **Autonomous Driving:** A stop sign could be perturbed to be misclassified as a speed limit sign.
    *   **Security Systems:** Bypassing facial recognition or authentication systems.
    *   **Generative AI (LLMs):** "Jailbreaking" prompts can make models bypass safety filters and generate harmful content. "Hallucination" can be seen as a form of internal adversarial drift.
*   **Types of Attacks:**
    *   **White-Box:** Attacker has full knowledge of the model architecture and parameters.
    *   **Black-Box:** Attacker can only query the model and see its outputs.
*   **Attack Methods:**
    *   **Fast Gradient Sign Method (FGSM):** A simple and fast one-step attack that calculates the perturbation as `η = ε * sign(∇ₓ J(θ, x, y))`, where `ε` is a small scaling factor, and `∇ₓ J` is the gradient of the loss w.r.t. the input image.
    *   **Projected Gradient Descent (PGD):** A stronger, iterative version of FGSM. It takes multiple smaller steps and projects the adversarial example back into a valid range (e.g., ensuring pixel values are valid) after each step.
*   **Defenses:**
    *   **Adversarial Training:** The most common defense. Training the model on a mixture of clean examples and adversarial examples. This teaches the model to be robust to such perturbations.
    *   **Formal Verification:** Mathematically proving a model is robust within a certain region around an input.
    *   **Detection:** Building a separate model to detect if an input is adversarial.
*   **The Broader Issue - AI Safety:**
    *   This field is closely tied to **AI alignment** – ensuring AI systems act in accordance with human values and intentions.
    *   Companies like **Anthropic** (makers of Claude) explicitly focus on building "safe" and "constitutional" AI, prioritizing value alignment alongside capability.

***

### Summary of Key Concepts:

| Concept | Description | Key Takeaway |
| :--- | :--- | :--- |
| **End-to-End Learning** | Jointly optimizing feature extraction and classification layers. | Superior to disconnected traditional ML pipelines; enabled by backpropagation. |
| **ResNet / Skip Connections** | `Output = F(x) + x` | Solves vanishing gradients, enabling training of networks that are hundreds of layers deep. |
| **1x1 Convolutions** | Filters that operate on the channel dimension only. | Used for cheap dimensionality reduction and feature pooling (e.g., in Inception modules). |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping. | Technique to create a heatmap showing which parts of an image were most important for a prediction. |
| **Adversarial Example** | `x_adv = x + η` | A minimally perturbed input that causes a model to make a confident error. Highlights model brittleness. |
| **Adversarial Training** | Training on a mix of clean and adversarial examples. | A primary defense method to improve model robustness against attacks. |