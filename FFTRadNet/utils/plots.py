
import numpy as np
import matplotlib.pyplot as plt
import os
#import cv2
from sklearn.cluster import DBSCAN

## plots for visualization model outputs

def matrix_plot(base_dir, predictions, labels, trial_num, datamode, epoch, batch):
    directory = os.path.join(base_dir, 'plot')
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    #prediction = predictions[0, 0, :, :].detach().cpu().numpy().copy()
    prediction = predictions[:, 0, :, :].detach().cpu().numpy().copy()
    #print("predictions: ", predictions.shape)
    #target_prediction = (prediction > 0.5).float()
    #label = labels[0, 0, :, :].detach().cpu().numpy().copy()
    label = labels[:, 0, :, :].detach().cpu().numpy().copy()

    for p in range(prediction.shape[0]):
        pred = prediction[p, :, :]
        lab = label[p, :, :]

        m1 = axs[0].imshow(pred, cmap='magma', interpolation='none')
        axs[0].set_title('prediction')
        axs[0].set_ylim(0, pred.shape[0])
        axs[0].set_xlim(0, pred.shape[1])
        axs[0].set_xlabel('azimuth')
        axs[0].set_ylabel('range')

        fig.colorbar(m1, ax=axs[0])

        # Plot the second matrix
        m2 = axs[1].imshow(lab, cmap='magma', interpolation='none', vmin=0.0, vmax=1.0)
        axs[1].set_title('label')
        axs[1].set_ylim(0, lab.shape[0])
        axs[1].set_xlim(0, lab.shape[1])
        axs[1].set_xlabel('azimuth')
        axs[1].set_ylabel('range')

        fig.colorbar(m2, ax=axs[1])

        # Save the plot with an incrementally named file
        # Check if the directory exists, if not, create it
        if not os.path.exists( directory):
            os.makedirs(directory)

        filepath = os.path.join(directory,f'matrix_plot_trial{trial_num}_epoch{epoch}_batch{batch}_num{p}_{datamode}.png')
        plt.savefig(filepath)
        #print(f'Plot saved to {filepath}')

        # Close the plot to free up memory
        plt.close()     

def plot_prediction_with_histogram(prediction):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting the prediction using imshow
    m1 = axs[0].imshow(prediction, cmap='magma', interpolation='none')
    axs[0].set_title('Prediction')
    axs[0].set_ylim(0, prediction.shape[0])
    axs[0].set_xlim(0, prediction.shape[1])
    axs[0].set_xlabel('Azimuth')
    axs[0].set_ylabel('Range')
    fig.colorbar(m1, ax=axs[0])

    # Plotting the histogram of the prediction's values
    axs[1].hist(prediction.ravel(), bins=50, color='purple')
    axs[1].set_title('Histogram of Prediction Values')
    axs[1].set_xlabel('Prediction Value')
    axs[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def count_targets(positions):
    # Convert positions to numpy array
    positions_array = np.array(positions)

    # Use DBSCAN clustering algorithm
    # eps is the maximum distance between two points to be considered in the same cluster
    # min_samples is the minimum number of points to form a cluster (set to 1 to include all points)
    db = DBSCAN(eps=1.5, min_samples=1).fit(positions_array)

    # Number of clusters (targets)
    num_targets = len(set(db.labels_))


    # The number of unique blocks is the number of targets
    return num_targets
    
### plot detection (classification)
def detection_plot(base_dir, predictions, labels, trial_num, datamode, epoch, batch):
    #print("predictions: ", predictions.shape) #torch.Size([4, 3, 128, 224])
    #prediction = predictions[0, 0, :, :].detach().cpu().numpy().copy()
    prediction = predictions[:, 0, :, :].detach().cpu().numpy().copy()  # predictions[:, 0, :, :]: map[0, :, :]
    #print("predictions: ", predictions.shape)
   

    #target_prediction = (prediction > 0.5).float()
    #label = labels[0, 0, :, :].detach().cpu().numpy().copy()
    label = labels[:, 0, :, :].detach().cpu().numpy().copy()
    # Specify the directory to save the plot
    directory = os.path.join(base_dir, 'plot') #'./plot_0706_16rx_seqdata/'
    # Iterate through each matrix

    list_target_sum = []

    for p in range(prediction.shape[0]):
        pred = prediction[p, :, :]
        lab = label[p, :, :]
        #target_num = 0
        # Create a figure
        plt.figure(figsize=(6, 6))

        target_position = []

        # Plot pre: Red points
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                #if pre[i, j] == 1:
                if pred[i, j] >= 0.2:
                    #target_num += 1
                    #print("predict target!!! probability: ", pred[i, j], "at position: ", i, j )
                    plt.scatter(j, i, color='red', s=1, label='prediction' if i == 0 and j == 0 else "")
                    target_position.append([i, j])
        # Check if the list is empty
        if not target_position:
            num_target = 0  # Return 0 if the list is empty
        else:
            num_target = count_targets(target_position)  # Call the count_targets function
        
        list_target_sum.append(num_target)
        #print("number of targets after combined: ", list_target_sum)

        # Plot lab: Blue points
        for i in range(lab.shape[0]):
            for j in range(lab.shape[1]):
                if lab[i, j] == 1:
                    plt.scatter(j, i, color='blue', s=1, label='ground truth' if i == 0 and j == 0 else "")

        if num_target != 3:
            # Set plot limits and labels
            plt.xlim(0, pred.shape[1])
            plt.ylim(0, pred.shape[0])
            #plt.gca().invert_yaxis()  # To match matrix indexing
            plt.xlabel('angle Index')
            plt.ylabel('range Index')
            plt.title('Comparison of prediction and labels')
            
            # Save the plot with an incrementally named file
            # Check if the directory exists, if not, create it
            if not os.path.exists(directory):
                os.makedirs(directory)

            filepath = os.path.join(directory, f'plot_trial{trial_num}_epoch{epoch}_batch{batch}_ind{p}_{datamode}.png')
            plt.savefig(filepath)
            #print(f'Plot saved to {filepath}')

            # Close the plot to free up memory
            plt.close()        

    return list_target_sum



## plot histograms and also create video
def plot_histograms(predictions, epoch, histogram, batch, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    prediction = predictions[0, 0, :, :]
    plt.figure(figsize=(8, 6))
    plt.hist(prediction.ravel(), bins=50, color='purple')
    plt.title(f'Histogram of Prediction Values - Epoch {epoch}')
    plt.xlabel('Prediction Value')
    plt.ylabel('Frequency')
    plt.ylim(0, 1000)
    filepath = os.path.join(output_dir, f'histogram_{histogram}_epoch{epoch}_batch{batch}.png')
    plt.savefig(filepath)
    print(f'Plot saved to {filepath}')
    plt.close()

# def create_video_from_images(image_dir, output_video, fps=2):
#     #output_video = "histograms_video.mp4"
#     images = sorted([img for img in os.listdir(image_dir) if img.endswith(".png")])
#     frame = cv2.imread(os.path.join(image_dir, images[0]))
#     height, width, layers = frame.shape

#     video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

#     for image in images:
#         video.write(cv2.imread(os.path.join(image_dir, image)))

#     cv2.destroyAllWindows()
#     video.release()
