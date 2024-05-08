#!/usr/bin/env bash

while getopts ":n" opt; do
  case ${opt} in
    n )
      newline="-n"
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

echo $newline" pip : \c"
which pip

echo $newline" pip3 : \c"
which pip3

echo $newline" python : \c"
which python

echo $newline" python3 : \c"
which python3

echo $newline" conda env list : \c"
conda env list

python3 -V
