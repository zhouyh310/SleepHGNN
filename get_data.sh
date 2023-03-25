mkdir -p data/raw_data
mkdir -p data/raw_label

for i in `seq 1 10`  
do
    echo Downloding subject ${i}...
    wget -nc -P data/raw_data/ http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupIII-Extractedchannels/subject${i}.mat
    wget -nc -P data/raw_label/ http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupIII/${i}.rar
    unrar e data/raw_label/${i}.rar data/raw_label/ -x *.txt
done

rm data/raw_label/*.rar