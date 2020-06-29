
#!/bin/sh

for entry in `ls .../path/of/16k/wavefiles/*.wav`; do

    fname=`basename $entry .wav`
    echo $fname
    sox $entry -r 8000 -b 16 -c 1 .../path/to/store/8k/files/$fname.wav 

done
exit
