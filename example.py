from KitNET.Results import resultAccuracy
from Kitsune import Kitsune
import numpy as np
import time

# Load Mirai pcap (a recording of the Mirai botnet malware being activated)
# The first 70,000 observations are clean...
# print("Unzipping Sample Capture...")
# import zipfile
# with zipfile.ZipFile("mirai.zip","r") as zip_ref:
#   zip_ref.extractall()


# File location
path = r'C:\Users\roiaz\PycharmProjects\AnomalyDetection\SSDP_Flood_2610000_2620000.pcapng.tsv'  # the pcap, pcapng, or tsv file to process.
labels_path = r'C:\Users\roiaz\PycharmProjects\AnomalyDetection\SSDP_Flood_labels.csv'
first_packet = 2610000
last_packet = 2620000

skip_rows = range(0, first_packet - 1)
num_of_rows = last_packet - first_packet + 2
res_acc = resultAccuracy(labels_path=labels_path, skip=skip_rows, num_of_rows=num_of_rows, threshold=10)
packet_limit = np.Inf  # the number of packets to process

# KitNET params:
maxAE = 10  # maximum size for any autoencoder in the ensemble layer
# FMgrace = 5000  # the number of instances taken to learn the feature mapping (the ensemble's architecture)
# ADgrace = 50000  # the number of instances used to train the anomaly detector (ensemble itself)
FMgrace = 100  # the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 1000  # the number of instances used to train the anomaly detector (ensemble itself)

# Build Kitsune
K = Kitsune(path, packet_limit, maxAE, FMgrace, ADgrace)

print("Running Kitsune:")
RMSEs = []
prediction_success_list = []
i = 0
start = time.time()
# Here we process (train/execute) each individual packet.
# In this way, each observation is discarded after performing process() method.
while True:
    i += 1
    if i % 1000 == 0:
        print(i)
    rmse = K.proc_next_packet()
    if rmse == -1:
        break
    RMSEs.append(rmse)
    prediction_success_list.append(int(res_acc.add(rmse=rmse, index=i)))
stop = time.time()
print("Complete. Time elapsed: " + str(stop - start))

# Here we demonstrate how one can fit the RMSE scores to a log-normal distribution (useful for finding/setting a cutoff threshold \phi)
from scipy.stats import norm

benignSample = np.log(RMSEs[FMgrace + ADgrace + 1:100000])
# benignSample = np.log(RMSEs[FMgrace+ADgrace+1])
# print(10*"$")
# print(benignSample)

logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))

# plot the RMSE anomaly scores
print("Plotting results")
from matplotlib import pyplot as plt

plt.figure(figsize=(10, 5))
fig = plt.scatter(range(FMgrace + ADgrace + 1, len(RMSEs)), RMSEs[FMgrace + ADgrace + 1:], s=0.1,
                  c=logProbs[FMgrace + ADgrace + 1:], cmap='RdYlGn')
success_rate = res_acc.accuracyRate()
print(f'success_rate is {success_rate:.3f}.')
plt.yscale("log")
plt.title("Anomaly Scores from Kitsune's Execution Phase")
plt.ylabel("RMSE (log scaled)")
plt.xlabel("Packet num")
figbar = plt.colorbar()
figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
plt.show()


fig2 = plt.scatter(range(0, len(prediction_success_list)), prediction_success_list,  s=0.1)
plt.title("Accuracy Rate")
plt.ylabel("Accuracy")
plt.xlabel("Packet num")
#figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
plt.show()