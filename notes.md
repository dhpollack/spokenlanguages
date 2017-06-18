## Initial creation
gcloud compute --project "dhpollack944" instances create "spokenlanguages-1" --zone "us-west1-a" --machine-type "n1-standard-2" --subnet "default" --no-restart-on-failure --maintenance-policy "TERMINATE" --preemptible --service-account "292078068994-compute@developer.gserviceaccount.com" --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring.write","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --tags "http-server","https-server" --image "debian-8-jessie-v20170523" --image-project "debian-cloud" --boot-disk-size "30" --boot-disk-type "pd-standard" --boot-disk-device-name "spokenlanguages-1" --metadata-from-file startup-script=startup.sh

## Create Image
gcloud compute instances stop --zone "us-west1-a" spokenlanguages-1

gcloud compute images create spokenlangv1 --source-disk spokenlangauges-1 --source-disk-zone us-west1-a --family debian-8

## Create from image
gcloud compute --project "dhpollack944" instances create "spokenlanguages-1" --zone "us-west1-a" --machine-type "n1-standard-2" --subnet "default" --no-restart-on-failure --maintenance-policy "TERMINATE" --preemptible --service-account "292078068994-compute@developer.gserviceaccount.com" --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring.write","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --tags "http-server","https-server" --image "spokenlangv1" --image-project "dhpollack944" --boot-disk-size "30" --boot-disk-type "pd-standard" --boot-disk-device-name "spokenlanguages-1" --metadata-from-file startup-script=startup.sh

## Start/Stop/SSH/Delete

gcloud compute instances start --zone "us-west1-a" spokenlanguages-1

gcloud compute instances stop --zone "us-west1-a" spokenlanguages-1

gcloud compute ssh --zone "us-west1-a" --ssh-key-file="/home/david/.ssh/google-cloud-lid" david@spokenlanguages-1

gcloud compute instances delete --zone "us-west1-a" spokenlanguages-1


## Locations

/var/log/syslog - output of startup script
