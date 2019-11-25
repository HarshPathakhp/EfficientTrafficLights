#!/bin/sh

rm -rf Results

scp -P 11185 nihesh@0.tcp.ngrok.io -r nihesh:/home/nihesh/Documents/SteelDefectDetection/Results ./
