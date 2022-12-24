# put your image generator here
fig = plt.figure(figsize=(10,7))
rows = 3
columns = 3

random_samples = torch.randn((9,4),device=device)
with torch.no_grad():
  samples = decoder(random_samples).cpu()
  for i in range(random_samples.shape[0]):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(samples[i][0],cmap="gist_gray")

plt.show()
# put your clustering accuracy calculation here

dataset_x = np.empty((48000,4))
dataset_y = np.empty((48000,))
i = 0

for X,y in train_loader:
  image_noisy = add_noise(X,noise_factor)
  image_noisy = image_noisy.to(device)
  encoded_output = encoder(image_noisy.to(device))
  dataset_x[i:i+batch_size,:] = encoded_output.cpu().detach().numpy()
  dataset_y[i:i+batch_size] = y
  i+=batch_size

kmeans = KMeans(n_clusters=10,random_state=1)
cluster_labels = kmeans.fit_predict(dataset_x)

dp = np.zeros((10,10))
dataset_y = dataset_y.astype(int)
for i in range(len(cluster_labels)):
  dp[dataset_y[i]][cluster_labels[i]] += 1

row,col = linear_sum_assignment(dp,maximize=True)

index_reassignment = {}

for i,j in zip(row,col):
  index_reassignment[j] = i

labels = np.copy(cluster_labels)
for i in range(0,len(cluster_labels)):
  labels[i] = index_reassignment[labels[i]]

print("Accuracy Score after index reassignement",accuracy_score(labels,dataset_y))