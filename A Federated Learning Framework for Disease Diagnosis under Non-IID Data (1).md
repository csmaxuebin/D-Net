# This code is the source code implementation for the paper "A Federated Learning Framework for Disease Diagnosis under Non-IID Data."

## Abstract
![输入图片说明](https://github.com/csmaxuebin/D-Net/blob/main/tp/%7B09B7C755-4D59-45db-B5B8-9488C462A79F%7D.png)
Federated learning has become one of the common frameworks for disease diagnosis because of its characteristics of protecting the privacy of local data and overcoming data islands. Recently, many disease diagnosis methods based on federated learning have been proposed, but most of them do not aim at the characteristics of non-independent and identically distributed (non-IID). Therefore, this paper proposes a disease diagnosis framework based on federated learning to improve the accuracy of disease prediction under non-IID. This framework has two main innovations: data sharing strategy method and image segmentation model. In the data distribution, it can improve the prediction accuracy when the data are not independent and identically distributed. In the stage of image segmentation, this paper proposes D-Net, which can realize the accurate segmentation of the image under the condition of low parameters. Finally, the experimental results on the datasets show that this framework can improve the accuracy of disease diagnosis.


# Experimental Environment

```
- absl-py==0.9.0
- alabaster==0.7.12
- anaconda-client==1.7.2
- anaconda-navigator==1.9.12
- numpy==1.18.1
- numpy-base==1.18.1
- numpydoc==0.9.2
- olefile==0.46
- pytorch==1.5.0
- python==3.7.6
- python-dateutil==2.8.1
- qt==5.9.7
- torchvision==0.6.0
- wheel==0.34.2
- xlrd==1.2.0
- zlib==1.2.11
- ...
```

## Datasets

`2D COVID-19, MNIST`


## Experimental Setup

- **Federated Learning Configuration:**
  - **Number of Clients:** The study involves multiple client nodes, each possessing local datasets that are not identically and independently distributed (non-IID).
  - **Rounds of Training:** The training process is conducted over several rounds, emphasizing iterative updates and learning improvements through client-server interactions.

- **Data Distribution and Management:**
  - **Non-IID Nature of Data:** The data used in the experiments are inherently non-IID, reflecting the diverse and varied nature of real-world medical datasets.
  - **Data Sharing Strategy:** This includes mechanisms to enhance prediction accuracy by effectively managing data distribution among the clients to mitigate the challenges posed by non-IID data.

- **Model and Algorithms:**
  - **Disease Diagnosis Framework:** Utilizes a federated learning approach tailored for the specifics of non-IID data, aiming to improve diagnostic accuracies.
  - **Image Segmentation Model (D-Net):** Introduced as part of the framework to handle image-based diagnostic processes, enabling precise segmentation with fewer parameters.

- **Privacy and Security:**
  - **Local Differential Privacy (LDP):** Implemented to ensure the privacy of data at the client level, crucial for sensitive medical data handling.

- **Evaluation Metrics:**
  - **Model Performance:** Assessed through accuracy measures and loss metrics to evaluate the effectiveness of the federated learning model in predicting diseases under the constraints of non-IID data.
  - **Privacy Protection:** Ensured by the adherence to LDP protocols and the federated nature of the learning process, which keeps the data decentralized.

- **Implementation Details:**
  - **Software and Tools:** The experiments are simulated or carried out using specific software tools tailored for federated learning applications, though specific software names are not mentioned in the abstract provided.

## Python Files
1. **baseline_i3d.py**:
   - Contains implementation or usage details of the Inflated 3D ConvNet (I3D) model, which is commonly used for action recognition in video data.

2. **common.py**:
   - This script contain common utility functions or configurations that are shared across multiple other scripts in the project, such as path setups, common parameters, or helper functions.

3. **Fed.py**:

   - This script pertains to federated learning processes, possibly defining the federated learning models, training procedures, or aggregation methods.

4. **head_helper.py**:

   - Contain utility functions related to the "head" of neural networks, which often deal with the final layers of models where decisions or predictions are made.

5. **init_weight.py**:

   - This file include functions or classes for initializing the weights of neural networks, which is crucial for training deep learning models effectively.

6. **layer.py**:

   - Defines custom layers for neural network architectures that might be used in other models within the project.

7. **model_resnet_i3d.py**:

   - This script combines ResNet, a popular convolutional network architecture, with I3D for improved performance in tasks like video action recognition.

8. **newUnet.py**:

   - Contain a modified or newly implemented version of the U-Net architecture, which is widely used for image segmentation tasks.

9. **nonlocal_helper.py**:

   - This script implement non-local operations, which capture long-range dependencies in images or videos, as part of neural network models.

10. **R2Unet.py**:

    - Involves an implementation of a recurrent residual U-Net (R2U-Net) for medical image segmentation, enhancing the original U-Net architecture with recurrent and residual connections.





##  Experimental Results
The resultssection of the document showcases the effectiveness of a proposed federated learning framework tailored for disease diagnosis under non-IID data. It emphasizes two main innovations: a data sharing strategy and an image segmentation model, D-Net. These components significantly enhance prediction accuracy and model efficiency in handling non-IID data conditions typical in medical scenarios. The framework was rigorously tested on various real-world datasets, demonstrating improvements in accuracy and computational efficiency over existing methods. Quantitative metrics such as model accuracy, loss, and computational resources underscored the practical applicability and robustness of the framework. Overall, the experimental outcomes validate the framework's potential in real-world medical applications, highlighting its capability to manage diverse data distributions securely and efficiently without compromising privacy.
![输入图片说明](/imgs/2024-06-17/hE5bcJzqc4OvfYBX.png)


```
## Update log

```
- {24.06.13} Uploaded overall framework code and readme file
