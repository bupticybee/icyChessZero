#/bin/bash
date 
for i in `seq 1 16`
do
{
	/usr/local/bin/python3 self_play.py	
} &
done
wait

