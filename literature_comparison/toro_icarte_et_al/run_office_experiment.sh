#!/bin/bash
#!/bin/bash
cd ./src
for i in `seq 0 59`;
do

	# Single task
	python run.py --alg=qlearning --env=Office-single-Toro-Icarte --num_timesteps=1e5 --gamma=0.9 --log_path=./data/ql/office-single-Toro-Icarte/experiment/$i
	python run.py --alg=qlearning --env=Office-single-Strategy --num_timesteps=1e5 --gamma=0.9 --log_path=./data/ql/office-single-Strategy/experiment/$i
done
