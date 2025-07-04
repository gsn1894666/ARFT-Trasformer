from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import math
import data_loader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from visualization import plot_training_metrics, plot_evaluation_metrics
import torch.nn as nn
from pytorch_tabnet.tab_network import TabNetNoEmbeddings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mmd import mmd_rbf_noaccelerate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training parameters
batch_size = 32
iteration = 1000
lr = 0.001
momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4
embedding_dim = 10


cuda = not no_cuda and torch.cuda.is_available()  # no_cuda
device = torch.device('cuda:0' if cuda else 'cpu')

# Dataset paths
root_path = "./dataset/"
src_name = "linux.csv"
tgt_name = "mysql.csv"

cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


# Patch the forward method to handle device mismatch
original_forward = TabNetNoEmbeddings.forward

def patched_forward(self, x):
    if x.device.type == 'cuda':

        if hasattr(self.encoder, 'group_attention_matrix'):
            if self.encoder.group_attention_matrix.device.type != 'cuda':
                self.encoder.group_attention_matrix = self.encoder.group_attention_matrix.to(x.device)

        for param in self.initial_bn.parameters():
            if param.device.type != 'cuda':
                param.data = param.data.to(x.device)

        if self.initial_bn.running_mean.device.type != 'cuda':
            self.initial_bn.running_mean = self.initial_bn.running_mean.to(x.device)
        if self.initial_bn.running_var.device.type != 'cuda':
            self.initial_bn.running_var = self.initial_bn.running_var.to(x.device)
    return original_forward(self, x)

TabNetNoEmbeddings.forward = patched_forward

def calculate_class_weights(loader):
    labels = [data_item[1].item() for data_item in loader.dataset]
    label_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(label_counts) * label_counts)
    k = 1.5
    ratio = (label_counts[0] / label_counts[1]) ** k
    weights[1] = weights[0] * ratio
    return torch.tensor(weights, dtype=torch.float32)

def visualize_features(model, src_loader, tgt_loader, iteration):
    model.eval()
    src_features, tgt_features = [], []
    with torch.no_grad():
        for src_data,  _ in src_loader:
            if cuda:
                src_data = src_data.cuda()
            src_feats, _ = model(src_data, return_features=True)  # only features
            src_features.append(src_feats.cpu().numpy())

        for tgt_data, _ in tgt_loader:
            if cuda:
                tgt_data = tgt_data.cuda(),
            tgt_feats, _ = model(tgt_data, return_features=True)  # only features
            tgt_features.append(tgt_feats.cpu().numpy())

    # transform NumPy
    src_features = np.concatenate(src_features, axis=0)
    tgt_features = np.concatenate(tgt_features, axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    all_features = np.vstack([src_features, tgt_features])
    embedded = tsne.fit_transform(all_features)
    # plot
    plt.scatter(embedded[:len(src_features), 0], embedded[:len(src_features), 1], c='blue', label='Source', alpha=0.6, edgecolors='k')
    plt.scatter(embedded[len(src_features):, 0], embedded[len(src_features):, 1], c='red', label='Target', alpha=0.6, edgecolors='k')
    plt.legend()
    plt.title(f"Feature Alignment at Iteration {iteration}")
    plt.show()



class TabNetClassifier(nn.Module):
    def __init__(self, num_classes=2, feature_dim=None, tabnet_output_dim=512, batch_size=256):
        super(TabNetClassifier, self).__init__()
        self.input_dim = feature_dim
        self.tabnet = TabNetNoEmbeddings(
            input_dim=self.input_dim,
            output_dim=tabnet_output_dim,
            n_d=16, n_a=16, n_steps=5, gamma=1.5,
            n_independent=2, n_shared=2,
            virtual_batch_size=batch_size, momentum=0.02, mask_type="sparsemax"
        )
        self.dropout = nn.Dropout(0.3)
        self.cls_fc = nn.Linear(tabnet_output_dim, num_classes)

    def forward(self, x, return_features=False):
        features, _ = self.tabnet(x)
        features = self.dropout(features)
        pred = self.cls_fc(features)
        if return_features:
            return pred, features  # Return classification results + extracted features
        return pred


def pretrain_tabnet(model, loader, criterion, epochs=50, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    best_balance = -float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_labels = []
        for data, label in loader:
            if cuda:
                data, label = data.cuda(), label.cuda()
            label = label.long()
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred_labels = pred.argmax(dim=1).cpu().numpy()
            true_labels = label.cpu().numpy()
            all_preds.extend(pred_labels)
            all_labels.extend(true_labels)

        avg_loss = total_loss / len(loader)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=1)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=1)
        balance = 1 - (((1 - recall) ** 2 + (1 - precision) ** 2) ** 0.5) / (2 ** 0.5)

        print(f"Pretrain Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}, "
              f"Precision: {precision:.4f}, Balance: {balance:.4f}")

        if balance > best_balance:
            best_balance = balance
            patience_counter = 0
            torch.save(model.state_dict(), 'best_pretrain_model.pth')
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
    model.load_state_dict(torch.load('best_pretrain_model.pth'))

def plot_loss_curve(iteration_history, total_loss_history):
    plt.plot(iteration_history, total_loss_history, label='Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

def train(model):
    class_weights = calculate_class_weights(src_loader)
    if cuda:
        class_weights = class_weights.cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)
    correct = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_decay)

    lambda_mmd = 0.5  # MMD weight
    # data record
    total_loss_history = []
    test_loss_history = []
    test_iterations = []
    iteration_history = []
    # Average value recorded every 100 times
    avg_loss_history = []
    precision_history = []
    recall_history = []
    f1_history = []
    balance_history = []
    pf_history = []
    metric_iterations = []

    print("✅ Start Training...")
    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / iteration), 0.75)
        if (i - 1) % 100 == 0:
            print(f'🔹 learning rate: {LEARNING_RATE:.4f}')

        try:
            src_data, src_label = next(src_iter)
        except StopIteration:
            src_iter = iter(src_loader)
            src_data, src_label = next(src_iter)

        try:
            tgt_data, _ = next(tgt_iter)  # no labels are required
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            tgt_data, _ = next(tgt_iter)

        src_data = src_data.to(device)
        src_label = src_label.to(device)
        tgt_data = tgt_data.to(device)

        # Adjust batch size
        min_batch_size = min(src_data.size(0), tgt_data.size(0))
        src_data = src_data[:min_batch_size]
        src_label = src_label[:min_batch_size]
        tgt_data = tgt_data[:min_batch_size]

        if i % log_interval == 0:  # Adjust printing conditions
            print(f"Train iteration {i}: src_data device: {src_data.device}, size: {src_data.size(0)}")
            print(f"Train iteration {i}: src_label device: {src_label.device}, size: {src_label.size(0)}")
            print(f"Train iteration {i}: tgt_data device: {tgt_data.device}, size: {tgt_data.size(0)}")

        src_label = src_label.long()
        optimizer.zero_grad()

        # **feedforward**
        src_pred, src_features = model(src_data, return_features=True)
        tgt_pred, tgt_features = model(tgt_data, return_features=True)

        # **cross entropy loss**
        ce_loss = criterion(src_pred, src_label)

        # **MMD loss**
        mmd_loss = mmd_rbf_noaccelerate(src_features, tgt_features)

        # **total loss**
        total_loss = ce_loss + lambda_mmd * mmd_loss

        total_loss.backward()
        optimizer.step()
        total_loss_history.append(total_loss.item())
        iteration_history.append(i)

        if i % (log_interval * 1) == 0:
            print(f'🟢 Train iter: {i} [{100. * i / iteration:.0f}%] '
                  f'Loss: {total_loss.item():.6f}')

        # Calculate the average value every 100 times
        if i % 100 == 0:
            start_idx = max(0, i - 100)
            avg_loss = np.mean(total_loss_history[start_idx:i])
            avg_loss_history.append(avg_loss)
            metric_iterations.append(i)

            # Test and record evaluation metrics
            t_correct, _, precision_list, recall_list, f1_list, balance_list, pf_list = test(model)
            precision_history.extend(precision_list)
            recall_history.extend(recall_list)
            f1_history.extend(f1_list)
            balance_history.extend(balance_list)
            pf_history.extend(pf_list)

            t_correct, test_loss, _, _, _, _, _ = test(model)
            test_loss_history.append(test_loss)
            test_iterations.append(i)
            if t_correct > correct:
                correct = t_correct
            print(f'🟠 src: {src_name} to tgt: {tgt_name} '
                  f'max correct: {correct} '
                  f'max accuracy: {100. * correct / tgt_dataset_len:.2f}%\n')
            src_accuracy = test_source(model)
            print(f'🟡 Source domain accuracy: {src_accuracy * 100:.2f}%')

        #Visualize the domain distribution every 500 iterations
        if i % 500 == 0:
            visualize_features(model, src_loader, tgt_loader, iteration)

    plot_evaluation_metrics(metric_iterations, precision_history, recall_history, f1_history, balance_history, pf_history)
    plot_loss_curve(metric_iterations, avg_loss_history,test_iterations, test_loss_history)


def test_source(model):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for src_data, src_label in src_loader:
            if cuda:
                src_data, src_label = src_data.cuda(), src_label.cuda()
            src_label = src_label.long()
            src_pred = model(src_data)
            pred = src_pred.argmax(dim=1)
            correct += pred.eq(src_label).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(src_label.cpu().numpy())

    accuracy = correct / src_dataset_len
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=1)
    balance = 1 - (((1 - recall) ** 2 + (1 - precision) ** 2) ** 0.5) / (2 ** 0.5)
    print("Source domain prediction distribution:", np.bincount(all_preds))
    print(f"Target domain: {precision:.4f}, Recall: {recall:.4f}, Balance: {balance:.4f}, F1: {f1:.4f}")
    return accuracy

def test(model):
    model.eval()
    correct = 0
    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for tgt_data, tgt_label in tgt_loader:
            if cuda:
                tgt_data, tgt_label = tgt_data.cuda(), tgt_label.cuda()
            tgt_label = tgt_label.long()
            tgt_pred = model(tgt_data)

            # **caculate test loss**
            loss = F.cross_entropy(tgt_pred, tgt_label)
            test_loss += loss.item()

            probs = torch.softmax(tgt_pred, dim=1)[:, 1]
            pred = (probs > 0.5).long()
            correct += pred.eq(tgt_label).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(tgt_label.cpu().numpy())

    avg_test_loss = test_loss / len(tgt_loader)  # average test_loss
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=1)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    pf = fp / (fp + tn) if (fp + tn) > 0 else 0
    balance = 1 - (((1 - recall) ** 2 + (1 - precision) ** 2) ** 0.5) / (2 ** 0.5)

    accuracy = 100. * correct / tgt_dataset_len
    print("Predicted label distribution:", np.bincount(all_preds))
    print(f'\n{tgt_name} set: Accuracy: {correct}/{tgt_dataset_len} ({accuracy:.2f}%) '
          f'Recall: {recall:.6f}, Precision: {precision:.6f}, '
          f'F1: {f1:.6f}, Balance: {balance:.6f}, PF: {pf:.6f}\n')
    return correct, avg_test_loss, [precision], [recall], [f1], [balance], [pf]


if __name__ == '__main__':
    print("Starting the script...")
    print(f"CUDA available: {cuda}, Device: {torch.cuda.current_device()}")

    # Load data
    src_loader = data_loader.load_training_with_oversampling(root_path, src_name, batch_size, kwargs)
    tgt_loader = data_loader.load_testing(root_path, tgt_name, batch_size, kwargs)
    
    # obtain feature number
    src_sample_data = src_loader.dataset[0][0]  # Feature data of the first sample
    feature_dim = src_sample_data.shape[0]
    print(f"Dynamically set feature_dim: {feature_dim}")

    src_dataset_len = len(src_loader.dataset)
    tgt_dataset_len = len(tgt_loader.dataset)

    model = TabNetClassifier(
        num_classes=2,
        feature_dim=feature_dim,
        tabnet_output_dim=512,
        batch_size=batch_size
    ).to(device)
    print(model)

    if cuda:
        model = model.cuda()
        for name, param in model.named_parameters():
            if param.device.type != 'cuda':
                print(f"Warning: Parameter {name} is on {param.device}, moving to cuda:0")
                param.data = param.data.to('cuda:0')
        print(f"Model moved to: {next(model.parameters()).device}")

    class_weights = calculate_class_weights(src_loader)
    if cuda:
        class_weights = class_weights.cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)


    src_train_dataset = src_loader.dataset
    tgt_test_dataset = tgt_loader.dataset
    print(f"Number of training set samples: {len(src_train_dataset)}")
    print(f"Number of testing set samples: {len(tgt_test_dataset)}")

    train(model)