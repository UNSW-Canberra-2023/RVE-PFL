# RVE-PFL
### RVE-PFL: Robust Variational Encoder-based Personalised Federated Learning  against Model Inversion Attacks  

Abstract—Federated learning (FL) enables distributed joint training of machine learning (ML) models without the need to share local data. FL is, however, not immune to privacy threats such as model inversion (MI) attacks. The conventional FL paradigm often uses privacy-preserving techniques, and this could lead to a considerable loss in the model’s utility and consequently compromised by MI attackers. Seeking to address this limitation, this paper proposes a robust variational encoder-based personalized FL (RVE-PFL) approach that mitigates MI attacks, preserves model utility, and ensures data privacy. RVE-PFL comprises an innovative personalized variational encoder architecture and a trustworthy threat model-integrated FL method to autonomously preserve data privacy and mitigate MI attacks. The proposed architecture seamlessly trains heterogeneous data at every client, while the proposed approach aggregates data at the server side and effectively discriminates against adversarial settings (i.e., MI); thus, achieving robustness and trustworthiness in real-time. RVE-PFL is evaluated on three benchmark datasets, namely: MNIST, Fashion-MNIST, and Cifar-10. The experimental results revealed that RVE-PFL achieves a high accuracy level while preserving data and tuning adversarial settings. It outperforms Noising before Model Aggregation FL (NbAFL) with significant accuracy improvements of 8%, 20%, and 59% on MNIST, Fashion-MNIST, and Cifar-10, respectively. These findings reinforce the effectiveness of RVE-PFL in protecting against MI attacks while maintaining the model’s utility.

# How to run
## Run RVE-PFL
1) Open the options.py file and customize the options for the selected dataset (mnist, fmnist, or cifar), #global rounds, #local rounds, ...etc. 
2) Run RVE-PFL.py.
3) You will get the training and test performance metrics and store them in a .csv file within the project directory. Additionally, both the personalized models and the global classifier models will be saved for subsequent use in the Model Inversion Attack Analysis.
## Attack Analysis

# Requirements
- python=3.9.17
- tensorboard-data-server==0.7.2
- tensorboard-plugin-wit==1.8.1
- tensorboardX==2.6.1
- tensorflow==2.15.0
- torch==2.2.0
- torchdata==0.7.1
- torchvision==0.17.0
- adversarial-robustness-toolbox==1.15.0

# Citation
In case you use this code, please cite:   https://ieeexplore.ieee.org/abstract/document/10443672
W. Issa, N. Moustafa, B. Turnbull and K. -K. R. Choo, "RVE-PFL: Robust Variational Encoder-based Personalised Federated Learning against Model Inversion Attacks," in IEEE Transactions on Information Forensics and Security, doi: 10.1109/TIFS.2024.3368879. keywords: {Training;Data models;Servers;Privacy;Internet of Things;Federated learning;Feature extraction;Federated Learning (FL);Variational Autoencoder (VAE);Model Inversion (MI) attack;Differential Privacy (DP)},

