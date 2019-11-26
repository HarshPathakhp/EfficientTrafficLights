#!/bin/sh

rm -rf Results

scp -P 19697 -r nihesh@0.tcp.ngrok.io:/home/nihesh/Documents/TrafficLights/Results ./
