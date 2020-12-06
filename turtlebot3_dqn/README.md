## Steps to run the RoboNav

- Set the TURTLEBOT3_MODEL in system environment\
`export TURTLEBOT3_MODEL=burger`

- Launch Gazebo with turtlebot3 in Gazebo\ (set the stage number)
`roslaunch turtlebot3_gazebo turtlebot3_stage_1.launch`

- Launch the DQN training\ (set the stage number, method, mode and others)
`roslaunch turtlebot3_dqn turtlebot3_dqn_torch.launch stage:=3 method:='dueling' mode:='test' move_3:='true'`
