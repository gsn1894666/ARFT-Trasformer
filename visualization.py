import matplotlib.pyplot as plt

def plot_evaluation_metrics(iterations, precision, recall, f1, balance,pf):
    """绘制评估指标曲线"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 1, 1)
    plt.plot(iterations, precision, label="Precision", color='blue')
    plt.plot(iterations, recall, label="Recall", color='green')
    plt.plot(iterations, f1, label="F1-Score", color='red')
    plt.plot(iterations, balance, label="Balance", color='purple')
    plt.plot(iterations, pf, label="pf", color='orange')
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    from DAN import src_name, tgt_name
    plt.title(f"Evaluation Metrics Over Iterations\n(Source: {src_name}, Target: {tgt_name})")

    plt.legend()

    plt.show()

def plot_train_loss(iteration_points, train_loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_points, train_loss_history, label='Training Loss (Avg every 100 iters)', color='blue', marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss (Avg every 100 iters)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_test_loss(iteration_points, test_loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_points, test_loss_history, label='Test Loss', color='orange', marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.legend()
    plt.grid(True)
    plt.show()