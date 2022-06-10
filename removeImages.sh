#!/bin/bash

#Leaves only "I - 1" number of images for each card
I=5

for d in ./Images/* ;
do
	find $d -type f -print0 | sort -zR | tail -zn +$I | xargs -0 rm
done
