#!/bin/bash
SESSION="ihm2"
SESSIONEXISTS=$(tmux list-sessions | grep $SESSION)
# Only create tmux SESSION if it doesn't already exist
if [ "$SESSIONEXISTS" = "" ]
then
    tmux new-session -d -s $SESSION

    tmux rename-window -t 0 'fox'
    tmux send-keys -t 'fox' 'mamba activate ihm2; . install/setup.sh; ros2 launch foxglove_bridge foxglove_bridge_launch.xml' C-m

    tmux new-window -t $SESSION:1 -n 'ros'
    tmux send-keys -t 'ros' 'mamba activate ihm2; . install/setup.sh' C-m
    tmux split-window -h
    tmux send-keys -t 'ros' 'mamba activate ihm2; . install/setup.sh' C-m

    tmux new-window -t $SESSION:2 -n 'debug'
    tmux send-keys -t 'debug' 'mamba activate ihm2; . install/setup.sh' C-m
fi

tmux attach-session -t $SESSION:1
