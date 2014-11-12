#!/usr/bin/env bash
# Run unit tests and format checks.
#
# Example usage: ./run_tests.sh
#
# Don't edit this file. You can use it to check the correctness
# of your implementation as you work. I recommend running it
# early and often to correct style errors and to check which unit
# tests still need work.

# added by Thanh; remove before turnin?
PYTHONPATH="${PYTHONPATH}:/lusr/opt/opencv-2.4.9/lib/python2.7/dist-packages"
export PYTHONPATH

# Check that source code conforms to PEP8.
if pep8 --show-source *.py; then
	echo "No style errors found. You're so hip!"
else
	echo "Some style errors detected."
fi
echo ""

