cd /work/icybee/icyChessZero/scripts
sh kill_all.sh
sleep 10s
sh multithread_start.sh -p /usr/local/bin/python3  -g 1 -t 20 -n no
sleep 10s
/usr/local/bin/python3 model_update.py -g 0
sleep 10s
sh validate.sh -p /usr/local/bin/python3 -g 0 -t 20
sleep 10s
sh multithread_start.sh -p /usr/local/bin/python3  -g 0 -t 20
