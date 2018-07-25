@echo off
for /L %%x in (1, 1,%1) do (
	echo "start /min %3 icyChess_selfplay.py -g %2"
	start /min %3 icyChess_selfplay.py -g %2
)