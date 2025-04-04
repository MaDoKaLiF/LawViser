SESSION_NAME="session"
CMD1="python train_eval.py --config=configs/one.json --seed=10"

tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "$CMD1" C-m
tmux attach-session -t $SESSION_NAME
