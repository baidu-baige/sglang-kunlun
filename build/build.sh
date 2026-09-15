#!/usr/bin/env bash

# This script is used to build the project.

set -e

workspace=`pwd`
echo $workspace

rm -rf output && mkdir output

#ls | grep -v output |awk '{print $1}'|xargs -i{} cp -r {} output/
tar -zcvf ../sglang_kunlun.tar.gz ../sglang-kunlun/
mv ../sglang_kunlun.tar.gz ./output/
