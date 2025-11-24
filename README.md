# Fast Slam base repository
Built to work on Eufsim simulator.
Contains:
* Individual nodes for prediction step, 4 algorithms for Data association over a known map and JCBB based correction step accessible through:

```ros2 run fslam fslam_pred```

Runs prediction without wheel speed corrections. Fairly accurate but drifts.
Gyro bis correction needs 1 second of recorded data at the time of initialization. (Reduced from 5 post gyro based correction and yaw inflation)

```ros2 run fslam fslam_jcbb```

Runs data association currently over a known skidpad map. To be run after prediction starts

```ros2 run fslam plot``` (optional)

Run to live view the prediction & correction results relative to GT. #RUN ONLY AFTER VEHICLE MOVES (ISSUE WITH PLOTTER)

### Subscribe to the output top of JCBB node for correct pose (use while skidpad)


## Path Planning

The repository now includes a full path planning pipeline integrated from `fsd_path_planning`.

### Overview
The pipeline consists of:
1. **Cone Sorting**: Orders cones into left and right boundaries.
2. **Cone Matching**: Matches left and right cones to define the track.
3. **Path Calculation**: Generates a smooth centerline path using splines and prepares it for MPC.

### Running the Node
To run the path planner node:

```bash
ros2 run pathp path_planner_node --ros-args -p mission_type:=trackdrive
```

**Parameters**:
- `mission_type`: `trackdrive` (default), `skidpad`, `acceleration`.
- `cones_topic`: Topic for cone input (default: `/ground_truth/cones`).
- `odom_topic`: Topic for odometry input (default: `/ground_truth/odom`).
- `path_topic`: Topic for output path (default: `/path_planner/path`).

**Example with custom topics**:
```bash
ros2 run pathp path_planner_node --ros-args -p cones_topic:=/my/cones -p odom_topic:=/my/odom
```

### Visualization
The node publishes a `nav_msgs/Path` message to `/path_planner/path`. You can visualize this in Rviz2 by adding a Path display and subscribing to this topic.

### Legacy Nodes
- `path_plot`: Live Delaunay plots (visualization only).
  ```bash
  ros2 run pathp path_plot
  ```



* Added a ros bag for controls to run each time. Works on small_track map

* Mapping Codes yet to be finished. So far map exists if driven correctly and on track. Unnecessary turns blow the dynamic gating algorithm

WILL UPDATE MAPPING AS IT PROGRESSES.
