## Initial creation
```sh
gcloud compute --project "dhpollack944" instances create "spokenlanguages-1" --zone "us-west1-a" --machine-type "n1-highcpu-8" --subnet "default" --no-restart-on-failure --maintenance-policy "TERMINATE" --preemptible --service-account "292078068994-compute@developer.gserviceaccount.com" --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring.write","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --tags "jupyter" --image "debian-8-jessie-v20170523" --image-project "debian-cloud" --boot-disk-size "30" --boot-disk-type "pd-standard" --boot-disk-device-name "spokenlanguages-1" --metadata-from-file startup-script=startup.sh
```

## Create Image
```sh
gcloud compute instances stop --zone "us-west1-a" spokenlanguages-1

gcloud compute images create slv2 --source-disk spokenlanguages-1 --source-disk-zone us-west1-a --family debian-8
```

## Create from image
```sh
gcloud compute --project "dhpollack944" instances create "spokenlanguages-1" --zone "us-west1-a" --machine-type "n1-standard-8" --subnet "default" --no-restart-on-failure --maintenance-policy "TERMINATE" --preemptible --service-account "292078068994-compute@developer.gserviceaccount.com" --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring.write","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --tags "jupyter" --image "slv2" --image-project "dhpollack944" --boot-disk-size "30" --boot-disk-type "pd-standard" --boot-disk-device-name "spokenlanguages-1" --metadata-from-file startup-script=startup.sh
```

## Start/Stop/SSH/Delete
```sh
gcloud compute instances start --zone "us-west1-a" spokenlanguages-1

gcloud compute instances stop --zone "us-west1-a" spokenlanguages-1

gcloud compute ssh --zone "us-west1-a" --ssh-key-file="/home/david/.ssh/google-cloud-lid" david@spokenlanguages-1

gcloud compute instances delete --zone "us-west1-a" spokenlanguages-1
```

## Locations

/var/log/syslog - output of startup script

## Calculate Convolutional layer output dimensions

```python
def conv_dim(vol, f, s, p, k):
    return((k, np.floor((vol[1]-f[0]+2*p[0])/s[0] + 1), np.floor((vol[2]-f[1]+2*p[1])/s[1] + 1)))

def pool_dim(vol, s):
    return((vol[0], np.floor(vol[1]/s[0]), np.floor(vol[2]/s[1])))
```

## Concepts

- Short-time Fourier Transform (spectrograms), Wavelet Transform (scaleogram)
- Resnet, LSTM, GRU, etc
- Connectionist Temporal Classification
