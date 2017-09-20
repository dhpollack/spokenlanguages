#!/bin/bash

apt-get update
apt-get install -y gcc build-essential cmake make bzip2 ffmpeg unzip git wget dos2unix sox libsox-dev libsox-fmt-all curl vim nano ca-certificates libjpeg-dev libpng-dev apt-transport-https

UNAME="david"
USRDIR="/home/"$UNAME

if [ -f $USRDIR/.setup_complete ]; then
    echo "Already Setup!"
else
    # install python3.6 with miniconda
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh
    chmod +x /tmp/miniconda3.sh
    /tmp/miniconda3.sh -b -p $USRDIR/miniconda3
    rm /tmp/miniconda3.sh
    source $USRDIR/miniconda3/bin/activate
    source activate root
    conda install -y conda-build
    conda install -y scipy scikit-learn jupyter pillow matplotlib h5py numpy ipython nb_conda mkl ipykernel pyyaml
    pip install librosa
    touch $USRDIR/.setup_complete
    chown -Rf $UNAME:$UNAME $USRDIR
    #https://stackoverflow.com/questions/21065922/how-to-open-a-specific-port-such-as-9090-in-google-compute-engi
fi
