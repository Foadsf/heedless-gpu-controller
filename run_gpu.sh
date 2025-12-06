#!/bin/bash

# Configuration
KERNEL_SLUG=$1

# Help text if no argument is provided
if [ -z "$KERNEL_SLUG" ]; then
    echo "Usage: ./run_gpu.sh <username/project-slug>"
    echo "Example: ./run_gpu.sh <KAGGLE_USER_NAME>/gpu-test"
    exit 1
fi

echo "🚀 Sending code to Kaggle GPU ($KERNEL_SLUG)..."
kaggle kernels push

echo "⏳ Waiting for remote execution..."
while true; do
    # specific grep to handle different status outputs cleanly
    STATUS=$(kaggle kernels status "$KERNEL_SLUG" | grep -o "KernelWorkerStatus.[A-Z]*")

    if [[ "$STATUS" == *"COMPLETE"* ]]; then
        echo "✅ Job Complete!"
        break
    elif [[ "$STATUS" == *"ERROR"* ]]; then
        echo "❌ Job Failed! Check web UI for details."
        exit 1
    fi

    # Simple spinner or status update
    echo -ne "Current Status: $STATUS\r"
    sleep 5
done

echo ""
echo "⬇️ Downloading logs..."
# -p . downloads to current directory, --quiet suppresses progress bar
kaggle kernels output "$KERNEL_SLUG" -p . --quiet

echo "📄 OUTPUT LOG (gpu-test.log):"
echo "---------------------------------"
# Check if log exists before cat
if [ -f "gpu-test.log" ]; then
    cat gpu-test.log
else
    # Fallback if the log name is different (often main.log or similar)
    cat *.log 2>/dev/null || echo "No log file found."
fi
echo "---------------------------------"
