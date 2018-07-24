@echo off
for /L %%x in (1, 1,%1) do (
	echo "start /min python self_play.py -g %2 -s %3"
	start /min python self_play.py -g %2 -s %3
)