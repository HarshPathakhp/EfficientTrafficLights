#!/bin/sh

rm -rf Results

scp -P 11185 -r nihesh@0.tcp.ngrok.io:/home/nihesh/Documents/SteelDefectDetection/Results ./
