# Deep Learning: Week 3 Study Sheet - The Visual System & CNNs

## Table of Contents
1.  [Biological Inspiration: The Human Visual System](#1-biological-inspiration-the-human-visual-system)
2.  [Fukushima's Neocognitron (1982)](#2-fukushimas-neocognitron-1982)
3.  [The Convolutional Neural Network (CNN)](#3-the-convolutional-neural-network-cnn)
4.  [Core CNN Operations: Convolution](#4-core-cnn-operations-convolution)
5.  [Core CNN Operations: Pooling](#5-core-cnn-operations-pooling)
6.  [Putting It All Together: A Simple CNN Architecture](#6-putting-it-all-together-a-simple-cnn-architecture)
7.  [Training Tricks & Concepts](#7-training-tricks--concepts)
8.  [Types of Data Shift](#8-types-of-data-shift)

---

## 1. Biological Inspiration: The Human Visual System

The human visual processing pipeline is divided into three conceptual zones:

*   **Image Processing Zone (The Eye):**
    *   **Function:** Captures the 3D world but acts as a filter. It does not process all information; it focuses on important features like sharp edges, boundaries, and areas of change (attention zones).
    *   **Analogy in AI:** The input layer of a neural network that receives raw pixel data.

*   **Computer Vision Zone (Early Brain Processing):**
    *   **Function:** Extracts meaning from the filtered visual data. It starts to join the edges and features detected by the eyes to form shapes and patterns.
    *   **Analogy in AI:** The hidden layers of a CNN that perform feature extraction (convolution).

*   **Deep Learning Zone (Decision-making in the Brain):**
    *   **Function:** Makes high-level decisions based on the processed information (e.g., recognizing the pattern as the letter "A").
    *   **Analogy in AI:** The final fully connected layers of a CNN that perform classification.

### The Two-Cell Theory

This theory explains how the brain achieves **invariance**—recognizing an object regardless of its position (shift) or size.

*   **Simple Cells:** Extract important, local features from a pattern (e.g., the specific angles and lines that make up the letter 'A'). They are sensitive to exact position and orientation.
*   **Complex Cells:** Aggregate inputs from simple cells. They are responsible for **tolerance**, making the recognition **invariant** to small shifts (translation) and size variations. They ignore minor variances and focus on the presence of the feature.

**Goal of CNNs:** Artificially simulate this simple-complex cell hierarchy to achieve shift and size invariance.

---

## 2. Fukushima's Neocognitron (1982)

A pioneering neural network architecture inspired by the visual system.

*   **Key Idea:** Uses layers of "simple" and "complex" cells to progressively extract features and build tolerance.
*   **Similarity Measure:** It measured the similarity between a test pattern and a stored training pattern.
    *   Vectors were extracted from "interest points" (key features) of both patterns.
    *   The **dot product** between these vectors was calculated. The dot product is high if vectors are similar (point in the same direction) and low if they are dissimilar.
    *   A **threshold function** was applied: if the similarity (dot product) was above a threshold `θ`, the pattern was recognized; if below, it was ignored.
*   **Limitations:**
    1.  Not trained end-to-end with gradient descent (the threshold `θ` was set manually).
    2.  Limited invariance capabilities.
    3.  Did not scale well to complex, high-dimensional images like real RGB photos.

---

## 3. The Convolutional Neural Network (CNN)

The modern evolution of the neocognitron, designed to be trained end-to-end.

*   **Proposed by:** Yann LeCun et al.
*   **Key Innovation:** Using **backpropagation** to learn the optimal filters (kernels) automatically from data, instead of manually setting thresholds.
*   **Mechanism for Invariance:**
    *   **Convolution Operation:** Achieves **shift invariance**. A filter detecting an edge will detect it anywhere in the image.
    *   **Pooling Operation:** Achieves **size invariance** and provides a form of translational invariance by progressively reducing the spatial size of the feature maps.

---

## 4. Core CNN Operations: Convolution

### Concept
The convolution operation simulates a neuron (or an "eye") looking at a small local region of the input data (the "panorama"), calculating a weighted sum, and then sliding (shifting) to the next region.

### Terminology
*   **Input / Data:** The raw image or feature map from the previous layer. Shape: `(Channels, Height, Width)`.
*   **Kernel / Filter:** A small matrix of weights that detects a specific feature (e.g., an edge). The network learns these. Shape: `(Output_Channels, Input_Channels, Kernel_Height, Kernel_Width)`.
*   **Feature Map / Activation Map:** The output of applying a kernel to the input. Each kernel produces one feature map.
*   **Stride:** The number of pixels the kernel shifts each time. A stride of `1` moves one pixel at a time; a stride of `2` skips one pixel, reducing the output size.
*   **Padding:** Adding pixels (usually zeros) around the border of the input image. This controls the spatial size of the output feature map.

### Types of Padding
*   **Valid Padding:** No padding. The kernel only operates on valid positions, which reduces the output size. `Output Size = (W - K + 1) / S`
*   **Same Padding:** Padding is added so that the output size is the **same** as the input size.
*   **Zero Padding:** Padding with zeros.
*   **Reflection Padding:** Padding using a mirror image of the input at the borders.
*   **Replication Padding:** Padding by repeating the last value at the borders.
*   **Circular Padding:** Padding such that the image is treated as a periodic signal (the left border is padded with pixels from the right border, and vice versa).

### Mathematics & Hand Calculation

The core operation is a **dot product** (element-wise multiplication and sum) between the kernel and a local patch of the input.

**Formula for 2D Convolution:**
`(I * K)[i, j] = ∑_{m} ∑_{n} I[i+m, j+n] • K[m, n]`
Where `I` is the input, `K` is the kernel, `i, j` are the coordinates in the output, and `m, n` iterate over the kernel dimensions.

**Example: 1D Convolution**
*   **Data:** `[0, 1, 2, 3, 4]`
*   **Kernel:** `[0.5, 0.5]` (a simple averaging filter)
*   **Stride:** `1`
*   **Padding:** `0` (Valid)

**Calculation:**
1.  Position 1: `[0, 1] • [0.5, 0.5] = (0*0.5) + (1*0.5) = 0.5`
2.  Position 2: `[1, 2] • [0.5, 0.5] = (1*0.5) + (2*0.5) = 1.5`
3.  Position 3: `[2, 3] • [0.5, 0.5] = 2.5`
4.  Position 4: `[3, 4] • [0.5, 0.5] = 3.5`

**Resulting Feature Map:** `[0.5, 1.5, 2.5, 3.5]`

**Demonstrating Shift Invariance:**
If we shift the input data to the right by adding a `0` at the beginning: `[0, 0, 1, 2, 3, 4]`, the resulting feature map is `[0, 0.5, 1.5, 2.5, 3.5]`. The non-zero values are the same, just shifted—this is shift invariance.

### Output Size Calculation
A crucial formula for determining the size of the feature map after a convolution layer.

**Formula:**
`W_out = floor( (W_in + 2P - K) / S ) + 1`
`H_out = floor( (H_in + 2P - K) / S ) + 1`

Where:
*   `W_in`, `H_in`: Input width and height.
*   `W_out`, `H_out`: Output width and height.
*   `K`: Kernel size (assumed square, e.g., 3 for a 3x3 kernel).
*   `P`: Padding size.
*   `S`: Stride.

**Example:**
*   Input: `32x32` image
*   Kernel: `3x3` (`K=3`)
*   Padding: `1` (`P=1`)
*   Stride: `1` (`S=1`)
*   `W_out = (32 + 2*1 - 3) / 1 + 1 = (32 + 2 - 3) + 1 = 32`
	= $\frac{(input + 2*padding) - kernel}{stride+padding}$
The output size is `32x32` (same as input, thanks to padding).

---

## 5. Core CNN Operations: Pooling

### Concept
Pooling reduces the spatial dimensions (width & height) of the feature maps. This achieves three things:
1.  **Size Invariance:** Makes the network less sensitive to the exact size of features.
2.  **Dimensionality Reduction:** Decreases computational cost and number of parameters.
3.  **Prevents Overfitting:** Provides a form of translation invariance by summarizing a region into a single value.

### Types of Pooling
*   **Max Pooling:** Takes the maximum value from a region. Most common, as it preserves the most salient features.
*   **Average Pooling:** Takes the average value from a region.

**Example: Max Pooling with 2x2 window and stride 2**
Input Patch: [ [5, 8], -> Max Pooling -> Output: [ [8, 4],  
[3, 4] ] [ [2, 9] ]  
Next Patch: [ [1, 4],  
[2, 9] ]

The `4x4` input is reduced to a `2x2` output.

---

## 6. Putting It All Together: A Simple CNN Architecture

A typical CNN has a pattern: **Convolution -> Activation (ReLU) -> Pooling**, repeated multiple times, followed by **Flattening -> Fully Connected (Dense) Layers -> Output**.

**Example: CIFAR-10 Classifier (Simplified)**
*   **Input:** `3x32x32` (3-channel RGB image of 32x32 pixels).
*   **Conv Layer 1:**
    *   Uses `32` different `3x3` kernels.
    *   Padding=`1`, Stride=`1`
    *   **Output Shape:** `32 x 32 x 32` (32 feature maps, each 32x32).
*   **Pooling Layer 1:**
    *   Max Pooling with `2x2` window, stride=`2`.
    *   **Output Shape:** `32 x 16 x 16` (spatial dimensions halved).
*   **(Repeat Convolution and Pooling blocks...)**
*   **Flatten Layer:**
    *   Converts the 3D feature map `(Channels, Height, Width)` into a 1D vector. `(32 * 16 * 16) = 8192` elements.
*   **Fully Connected Layers:**
    *   Standard neural network layers that take the flattened vector and perform classification.
    *   Final layer has 10 nodes (for CIFAR-10's 10 classes) with a softmax activation.

---

## 7. Training Tricks & Concepts

### 1. Normalization / Standardization
*   **Why?** Features (input pixels or learned features) can have different scales (e.g., age 0-100 vs. weight 10-200). This can slow down or destabilize training.
*   **Normalization:** Scaling features to a range, typically `[0, 1]`.
*   **Standardization:** Transforming features to have a **mean of 0** and a **standard deviation of 1**. `X_new = (X - μ) / σ`. This is very common in deep learning.
*   **Batch Normalization:** A powerful technique that standardizes the *activations* of a layer across a *mini-batch* during training. It helps networks train faster and be more stable.

### 2. Dropout
*   **What?** Randomly "dropping out" (setting to zero) a percentage of neurons during each training step.
*   **Why?** Prevents overfitting. It stops neurons from becoming overly reliant on specific upstream neurons, forcing the network to learn more robust features. It's like preventing a student from memorizing answers by randomly changing the questions slightly.
*   **How?** Applied usually on fully connected layers. During testing, all neurons are active, but their outputs are scaled down by the dropout probability.

---

## 8. Types of Data Shift

Understanding how real-world data can change is crucial for deploying robust models.

*   **Covariate Shift:** The distribution of the input data `P(X)` changes between training and testing, but the conditional distribution `P(Y|X)` (the relationship between input and label) remains the same.
    *   **Example:** Training on high-quality, sunny day photos of houses, testing on blurry, cloudy day photos. A house is still a house.
*   **Label Shift / Prior Probability Shift:** The distribution of the *labels* `P(Y)` changes, but `P(X|Y)` remains the same.
    *   **Example:** Training a disease classifier on a hospital population where disease prevalence is 50%. Deploying it in the general population where prevalence is 1%.
*   **Concept Shift:** The very meaning of the label `Y` for a given input `X` changes. `P(Y|X)` changes.
    *   **Example:** The definition of "spam" email evolves over time.
*   **Domain Shift:** A broad term encompassing covariate, label, and concept shift when moving from one "domain" (e.g., one MRI machine) to another.




---

#### **2. Image Representation: The Input**

*   **Pixel:** The fundamental building block of an image.
*   **Grayscale Image:** A 2D tensor (matrix) of shape `(Height, Width)`. Each value is a scalar between 0 (black) and 255 (white).
*   **Color Image:** A 3D tensor of shape `(Channels, Height, Width)`. Typically 3 channels: Red, Green, and Blue (RGB). Each channel is a 2D matrix.
*   **Batch Processing:** In practice, we process multiple images simultaneously. A batch is a **4D tensor**: `(Batch_size, Channels, Height, Width)`.

# 3. The Core Building Blocks of a CNN

A standard CNN has three main types of layers:

## 1. Convolutional Layer (The Feature Extractor)
*   **Purpose:** To detect local patterns (features) from the input using learnable filters (kernels).
*   **Operation:**
    *   A filter (a small tensor, e.g., 3x3, 5x5) slides (convolves) across the input image.
    *   At each location, it performs an **element-wise multiplication** between the filter and the image patch, then **sums all the products** to produce a single value in the output **feature map**.
    *   Multiple filters are used to detect different features (e.g., edges, textures, colors), creating a stack of feature maps (an output volume).
*   **Key Hyperparameters:**
    *   **Kernel/Filter Size (f):** The spatial dimensions of the filter (e.g., 3, 5).
    *   **Stride (s):** The number of pixels the filter moves each step. A stride of 2 downsamples the feature map by half.
    *   **Padding (p):** Adding zeros around the border of the input image. Used to control the spatial size of the output (often to preserve it).
*   **Output Size Calculation (CRITICAL FORMULA):**
    *   For an input of size `(H_i, W_i)`, the output feature map size `(H_o, W_o)` is:
        `H_o = floor( (H_i + 2p - f_h) / s ) + 1`
        `W_o = floor( (W_i + 2p - f_w) / s ) + 1`
	    *   Example: $$\begin{matrix}Input\ \ 7x7, \ kernel\ \ 3x3, \ stride=2, \\ padding=1 -> Output = floor((7+2-3)/2)+1 = floor(6/2)+1 = 3+1 = 4x4\end{matrix}$$

## 2. Pooling Layer (The Downsampler)
*   **Purpose:** To achieve *spatial invariance* (making the network less sensitive to the exact position of a feature) and to **reduce computational complexity** by progressively reducing the spatial size of the representation.
*   **Operation:** Operates independently on each feature map (channel).
    *   **Max Pooling:** Outputs the maximum value in each window. Most common, as it preserves the most salient features.
    *   **Average Pooling:** Outputs the average value in each window.
*   **Common Hyperparameters:**
    *   **Pool size = (2,2), Stride = (2,2), Padding = 0.** This downsamples the input by a factor of 2 (e.g., 224x224 -> 112x112).

## 3. Fully-Connected (FC) Layer (The Classifier)
*   **Purpose:** To perform high-level reasoning and classification based on the features extracted by the convolutional and pooling layers.
*   **Operation:** Typically placed at the end of the network. The 3D feature volume from the last layer is **flattened** into a 1D vector and fed into one or more traditional neural network layers.
*   The final FC layer has as many neurons as there are classes, often followed by a softmax activation to output class probabilities.

# 4. Advanced Layers for Stable & Effective Training

## 1. Batch Normalization (BatchNorm) Layer
*   **Problem it Solves: Internal Covariate Shift** - The change in the distribution of layer inputs during training, which slows down learning.
*   **Operation (Training):**
    1.  For a mini-batch, standardize the values for each channel: $z_{hat} = \frac{(z - μ_B)} {sqrt(σ_B² + ϵ)}$
        *   `μ_B`: mean of the mini-batch for that channel.
        *   `σ_B²`: variance of the mini-batch for that channel.
        *   `ϵ`: small constant for numerical stability.
    2.  *Scale* and *shift* the normalized value using two *learnable parameters*: $z_{BN} = γ * z_{hat} + β$
*   **Benefits:** Allows higher learning rates, reduces overfitting, makes training much more stable and faster.
*   **Testing:** Uses running averages of mean and variance computed during training, not the batch statistics.

## 2. Dropout Layer
*   **Problem it Solves: Overfitting.**
*   **Operation (Training):** Randomly "drops" (sets to zero) a fraction (`p`) of the neurons in a layer during each training step. This prevents neurons from co-adapting too much and forces the network to learn robust features.
*   **Testing:** All neurons are active, but their outputs are multiplied by the dropout probability `(1 - p)` to scale the output correctly.

# 5. Classic CNN Architectures (Know the Evolution)

*   **LeNet-5 (Pioneer):** One of the first successful CNNs (Conv -> Pool -> Conv -> Pool -> FC -> FC -> Output). Small and simple for tasks like digit recognition.
*   **AlexNet (Breakthrough):** Similar philosophy to LeNet but **deeper** and larger. Key innovations: Use of **ReLU** activation (faster training), **dropout** for regularization, and training on GPUs. Won ImageNet 2012.
*   **VGG (Simplicity & Depth):** Key idea: use **small 3x3 filters** stacked in deep layers. A VGG block consists of multiple 3x3 conv layers followed by one max-pooling layer. This design builds more complex features with fewer parameters than a single large filter (e.g., 5x5 or 7x7). Very uniform and deep architecture.

# 6. Putting It All Together: A Typical CNN Data Flow

## 1.  Input: Image tensor 
`(Batch_size, 3, 32, 32)`
1.  **Feature Learning Stage:**
    *   **Conv Layer 1:** Applies `K1` filters. Output: `(Batch_size, K1, H1, W1)`
    *   **(Optional) BatchNorm + Activation (ReLU)**
    *   **Pooling Layer 1:** Downsamples. Output: `(Batch_size, K1, H1/2, W1/2)`
    *   **(Optional) Dropout**
    *   Repeat this pattern, increasing the number of filters (`K2, K3...`) and decreasing spatial size.
2.  **Classification Stage:**
    *   **Flatten:** Convert final 3D feature map `(Batch_size, K_final, H_final, W_final)` to a 1D vector `(Batch_size, K_final * H_final * W_final)`.
    *   **Fully-Connected Layers:** One or more layers for classification.
    *   **Output Layer:** FC layer with softmax activation for class probabilities.

# 7. Key PyTorch Functions to Remember

*   `torch.nn.functional.conv2d(...)`: Performs a 2D convolution.
*   `torch.nn.functional.max_pool2d(...) / avg_pool2d(...)`: Performs pooling.
*   `torch.nn.BatchNorm2d`: Batch normalization layer.
*   `torch.nn.Dropout`: Dropout layer.
*   `torch.nn.Flatten`: Flattens the input.

---
**How to Use This Sheet:**
1.  **Understand the Concepts:** Don't just memorize the formulas. Understand *why* each layer is used (e.g., "Conv layers extract features, pooling layers make them invariant").
2.  **Practice Calculations:** Be able to calculate the output size of any convolutional or pooling layer given `(H_i, W_i, f, s, p)`.
3.  **Compare Architectures:** Be able to explain the key differences between LeNet, AlexNet, and VGG.
4.  **Explain Advanced Layers:** Be prepared to explain what BatchNorm and Dropout do and why they are important.