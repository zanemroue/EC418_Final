from planner import Planner, save_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from utils import load_data
import dense_transforms
from torch.utils.data import random_split
import time
import matplotlib.pyplot as plt


def train(args):
    from os import path
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print("Starting training...")
    print(f"Using device: {device}")
    model = model.to(device)
    
    if args.continue_training:
        model_path = path.join(path.dirname(path.abspath(__file__)), 'planner.th')
        if path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print("Loaded saved model for continued training.")
        else:
            print("No saved model found. Starting fresh training.")

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    
    #Load and split data into training and validation sets
    full_dataset = load_data('drive_data', transform=transform, num_workers=args.num_workers)
    if len(full_dataset) == 0:
        raise ValueError("Loaded dataset is empty. Please check the dataset path and contents.")
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=args.num_workers)
    
    global_step = 0
    start_time = time.time()

    # To store losses for plotting
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(args.num_epoch):
        epoch_start_time = time.time()
        model.train()
        train_losses = []
        print(f"\nEpoch {epoch+1}/{args.num_epoch} - Training Started")
        
        for batch_idx, (img, label) in enumerate(train_loader, 1):
            img, label = img.to(device), label.to(device)
            
            # Debugging: Print the shape of img and label
            if batch_idx == 1:
                print(f"Batch {batch_idx}: img shape = {img.shape}, label shape = {label.shape}")
            
            try:
                pred = model(img)
                loss_val = loss_fn(pred, label)
            except Exception as e:
                print(f"Error during forward pass: {e}")
                continue  # Skip this batch
            
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            if train_logger is not None:
                train_logger.add_scalar('Loss/Train', loss_val.item(), global_step)
                if global_step % 100 == 0:
                    log(train_logger, img, label, pred, global_step)
            
            train_losses.append(loss_val.item())
            global_step += 1
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{args.num_epoch}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss_val.item():.4f}")
        
        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        train_loss_history.append(avg_train_loss)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} Completed - Avg Train Loss: {avg_train_loss:.4f} - Duration: {epoch_duration:.2f}s")
        
        # Validation
        model.eval()
        val_losses = []
        print(f"Epoch {epoch+1} - Validation Started")
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)
                try:
                    pred = model(img)
                    loss_val = loss_fn(pred, label)
                    val_losses.append(loss_val.item())
                except Exception as e:
                    print(f"Error during validation pass: {e}")
                    continue  # Skip this batch
        
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        val_loss_history.append(avg_val_loss)
        print(f"Epoch {epoch+1} Completed - Avg Validation Loss: {avg_val_loss:.4f}")
        
        if valid_logger is not None:
            valid_logger.add_scalar('Loss/Validation', avg_val_loss, epoch)
        
        # Save model after each epoch
        save_model(model)
        print(f"Model saved after epoch {epoch+1}")
    
    total_duration = time.time() - start_time
    print(f"\nTraining Completed - Total Duration: {total_duration/60:.2f} minutes")
    save_model(model)

    # Plot training and validation loss
    plot_loss(train_loss_history, val_loss_history)


def plot_loss(train_loss, val_loss):
    """
    Plot the training and validation loss.
    """
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predicted aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    import numpy as np
    
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    img_np = TF.to_pil_image(img[0].cpu())
    ax.imshow(img_np)
    WH2 = np.array([img.shape[-1], img.shape[-2]]) / 2
    label_np = label[0].cpu().detach().numpy()
    pred_np = pred[0].cpu().detach().numpy()
    ax.add_artist(plt.Circle(WH2 * (label_np + 1), 5, ec='g', fill=False, lw=2))
    ax.add_artist(plt.Circle(WH2 * (pred_np + 1), 5, ec='r', fill=False, lw=2))
    logger.add_figure('Prediction', fig, global_step)
    plt.close(fig)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Planner Model')
    
    parser.add_argument('--log_dir', type=str, default=None, help='Directory for TensorBoard logs')
    parser.add_argument('-n', '--num_epoch', type=int, default=50, help='Number of training epochs')
    parser.add_argument('-w', '--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-c', '--continue_training', action='store_true', help='Continue training from saved model')
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])', help='Data transformations')
    
    args = parser.parse_args()
    train(args)
