#!/bin/bash
set -ex
mkdir pretrain
gdown --id 1RC1YsnxlgOsa24A7_BiHimgMJlK-8rYR -O pretrain/albert.tar
gdown --id 16AvDelZ7U0a0sbsi1rdFEV2cm_0pPlLJ -O pretrain/char.tar
gdown --id 1o__V6nJ6hh0NWe7NrcXND0nUEJl8kZaP -O pretrain/electra.tar
gdown --id 1wPcw_9gor7RqekxdbMxEP50L_PppsmnM -O pretrain/bert.tar
for file in pretrain/*.tar
do 
    tar -xf "$file"  -C pretrain
done
rm pretrain/*.tar