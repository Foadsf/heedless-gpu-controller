#!/bin/bash

KERNEL_SLUG=$1
if [ -z "$KERNEL_SLUG" ]; then
    echo "Usage: $0 <your_username/kernel-slug>"
    exit 1
fi

echo "🚀 Pushing code (Version $(date +%s)) to $KERNEL_SLUG"
kaggle kernels push -p .

echo "✅ Push complete! Monitor here: https://www.kaggle.com/code/$KERNEL_SLUG"
echo "When complete, run: kaggle kernels output $KERNEL_SLUG -p outputs/"
