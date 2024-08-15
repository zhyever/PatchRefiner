#!/bin/bash

# Get the list of JOBIDs for the current user
jobids=$(squeue --me --noheader --format=%i)

# Iterate through each JOBID and cancel it
for jobid in $jobids; do
    scancel $jobid
    echo "Cancelled job $jobid"
done