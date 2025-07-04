import matplotlib.pyplot as plt

def plot_training_metrics(iterations, train_loss, soft_loss, mmd_loss, accuracy):
    """绘制训练损失和准确率曲线"""
    plt.figure(figsize=(12, 5))

    # 🔹 训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(iterations, train_loss, label="Total Loss", color='blue')
    plt.plot(iterations, soft_loss, label="Classification Loss", color='green')
    plt.plot(iterations, mmd_loss, label="MMD Loss", color='red')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Iterations")
    plt.legend()

    # 🔹 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(iterations, accuracy, label="Accuracy", color='purple')
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy Over Iterations")
    plt.legend()

    plt.show()

def plot_evaluation_metrics(iterations, precision, recall, f1, balance,pf):
    """绘制评估指标曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, precision, label="Precision", color='blue')
    plt.plot(iterations, recall, label="Recall", color='green')
    plt.plot(iterations, f1, label="F1-Score", color='red')
    plt.plot(iterations, balance, label="Balance", color='purple')
    plt.plot(iterations, pf, label="pf", color='orange')
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics Over Iterations")
    plt.legend()

    plt.show()

def plot_loss_curve(iterations, losses):
    """绘制训练过程中的总损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, label="Total Loss", color='blue', linewidth=1.5)
    plt.plot(test_iterations, test_loss_history, label='Test Loss', color='red', marker='o', linestyle='dashed')

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Total Loss Over Iterations")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

