# Docker
## Overview
Docker is a platform that allows you to develop, ship, and run applications in an open environment or a container. With Docker, you can manage your infrastructure using the same methodologies you use to manage your applications.

## Downloading Docker
https://www.docker.com/products/docker-desktop/

The link takes you to the official docker website to install Docker Desktop for the Apple Chip, Windows, and Linux.

### To get visualation working and To forward XQuartz from inside a docker container to a host running on macOS

1. Install XQuartz: https://www.xquartz.org/
2. Launch XQuartz. Under the XQuartz menu, select Preferences
3. Go to the security tab and ensure "Allow connections from network clients" is checked.
4. Run xhost + ${hostname} to allow connections to the macOS host 
5. Setup a HOSTNAME env var export HOSTNAME=`hostname`
Add the following to your docker-compose:

`% IP=$(/usr/sbin/ipconfig getifaddr en0)`

`echo $IP`

`% /opt/X11/bin/xhost + "$IP"`


`% docker run -it -e DISPLAY="${IP}:0" -v / tmp./.X11-unix:/tmp/.X11-unix hypercode_code`

`cd HyperTools/`

`mkdir build`

`cd build/`

`cmake ..`

`Make -j3`

## Running Docker From the Terminal
**In the terminal**

`IP=$(/usr/sbin/ipconfig getifaddr en0)`
`/opt/X11/bin/xhost + "$IP"`

*cd to the Hypertools folder in the terminal and then run*
`pwd`
*the location of the folder can be traced with the following input into the terminal to replace into the next line of code accordingly*

`docker run -it --rm -e DISPLAY="${IP}:0" -v /tmp/.X11-unix:/tmp/.X11-unix hypercode_code`









