TOTAL_CALLS=15
NUM_EXPERIMENTS=5

for i in $( seq 0 $TOTAL_CALLS ); do
    echo "Run: $((i % NUM_EXPERIMENTS)) config: $((i / NUM_EXPERIMENTS))"
done
