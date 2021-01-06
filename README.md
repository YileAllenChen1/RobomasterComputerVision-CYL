# RobomasterComputerVision-CYL

## Robomaster Competition
The RoboMaster program is a platform for robotic competitions and academic exchange founded by Da-Jiang Innovations (DJI) and specially designed for global technology enthusiasts. The RoboMaster University Series has been expanding its influence year by year, attracting more than 400 colleges and universities around the world to participate annually and nurturing the engineering talents of over 35,000 young learners. For more information please check [RoboMaster Website](https://www.robomaster.com/en-US)

During the competition, robots from two teams battle each other on the arena by shooting pellets at armor plates mounted on the robots. The rules and format are somewhat similar to League of Lengends (LOL). Computer vision is widely used on the robots to track and detect opponent robots and perform autoaiming and shooting. There is also a power rune that robots have to target to earn energy buffs during the competition, which also requires computer vision.

<p align="center">
  <img src="demo/arena.jpg" height="300" width="350" > <img src="demo/arena2.jpg" height="300" width="450" > 
</p>

## CV
Apply traditional OpenCV methods to perform object detection and tracking on Power Rune and robot armor plate. Check it out [here](https://github.com/YileAllenChen1/RobomasterComputerVision-CYL/tree/main/CV)

## Modeling
Build a custom object detection model to detection robot armor plates using Yolo V5 (In progress). Check it out [hear](https://github.com/YileAllenChen1/RobomasterComputerVision-CYL/tree/main/modeling)