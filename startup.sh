#!/bin/bash

apt-get update
apt-get install -y gcc make bzip2 ffmpeg unzip git wget dos2unix

UNAME="david"
USRDIR="/home/"$UNAME
SPKDIR="/spokenlanguages"

if [ -f $USRDIR/.setup_complete ]; then
    echo "Already Setup!"
else
    git clone https://github.com/dhpollack/spokenlanguages.git $USRDIR$SPKDIR
    #below steps done in git repo already
    #dos2unix $USRDIR$SPKDIR/data/trainingset.csv
    #dos2unix $USRDIR$SPKDIR/data/testingset.csv
    #sed -e 's/$/,English/' -i $USRDIR$SPKDIR/data/testingset.csv
    #sed -e 's/$/,English/' -i $USRDIR$SPKDIR/data/trainingset.csv
    mkdir -p $USRDIR$SPKDIR/data/train $USRDIR$SPKDIR/data/test $USRDIR$SPKDIR/models $USRDIR$SPKDIR/output/states
    wget http://www.topcoder.com/contest/problem/SpokenLanguages/S1.zip -O $USRDIR$SPKDIR/data/S1.zip
    wget http://www.topcoder.com/contest/problem/SpokenLanguages/S2.zip -O $USRDIR$SPKDIR/data/S2.zip
    # install python3.6 with miniconda
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh
    chmod 755 /tmp/miniconda3.sh
    /tmp/miniconda3.sh -b -p $USRDIR/miniconda3
    rm /tmp/miniconda3.sh
    source $USRDIR/miniconda3/bin/activate
    conda create -y -n hu
    source activate hu
    conda install -y pytorch torchvision -c soumith
    #conda install pytorch torchvision cuda80 -c soumith
    conda install -y scipy scikit-learn jupyter matplotlib h5py
    pip install librosa pydub
    unzip $USRDIR$SPKDIR/data/S1.zip -d $USRDIR$SPKDIR/data/trainingset
    unzip $USRDIR$SPKDIR/data/S2.zip -d $USRDIR$SPKDIR/data/testingset
    rm $USRDIR$SPKDIR/data/S*.zip
    touch $USRDIR/.setup_complete
    chown -Rf $UNAME:$UNAME $USRDIR
    #https://stackoverflow.com/questions/21065922/how-to-open-a-specific-port-such-as-9090-in-google-compute-engi
fi
