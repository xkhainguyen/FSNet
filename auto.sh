while true; do
    codex resume \
    --yolo \
     019d8ec2-254a-7741-be75-33c5d9e335b1 \
    "have a look at program.md and continue the experiment loop" \
    2>&1 | tee -a agent.log
    sleep 5
done