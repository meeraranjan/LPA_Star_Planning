# LPA* Navigation Channel Path Planning

## Project Overview

This project implements a dynamic path planning system for the RobotX Navigation Channel task, where an autonomous boat must navigate through a channel marked by red and green buoys. The system uses Lifelong Planning A* (LPA*), an incremental variant of A* that efficiently handles dynamic environments by reusing previous computations when the environment changes.

The planner identifies gate pairs from buoy positions, computes optimal paths through the navigation channel, and provides local goal points for the boat to follow in real-time.

## Demo
![Navigation Channel Demo](https://youtube.com/shorts/vyo19bY0nTs?feature=share)

## Key Components

### 1. Local Goal Selector (`local_goal_selector.py`)

The Local Goal Selector is the main coordinator that manages gate detection, tracking, and goal selection.

**Initialization:**
- Sets up `GateManager` - tracks gate progression and used buoys
- Sets up `GateBuilder` - computes gates from buoy positions
- Subscribes to boat pose and red/green buoy data
- Publishes local goals and gate midpoints for navigation

**Main Callbacks:**
- **Buoy Data Parsers**: Processes red and green buoy position data
- **Boat Pose Updates**: Maintains current boat position and heading
- **Gate Recomputation**: 
  - Filters out already-used buoys via `GateManager`
  - Generates new gate sequences from `GateBuilder`
  - Calls `buoys2gates` to match red-green buoy pairs
  - Computes gate midpoints and headings
  - Updates `GateManager` with new gate information

**Main Loop:**
- Publishes all gate midpoints for visualization
- Checks if current gate has been passed using `GateManager.check_and_advance()`
- Computes goal position/heading via `compute_goal`
- Retrieves current gate from `GateManager` and returns midpoint & heading towards that gate

### 2. LPA* Node (`lpa_star_node.py`)

Implements the Lifelong Planning A* algorithm for dynamic replanning in the navigation channel.

**Core Functionality:**
- Maintains two distance estimates per cell: g-values (known cost from start) and rhs-values (one-step lookahead values)
- Uses priority queue to track locally inconsistent cells that need updating
- Efficiently updates only affected portions of the path when environment changes
- Reuses information from previous searches to drastically reduce computation compared to running A* from scratch

**Key Features:**
- First search behaves identically to A*
- Subsequent searches only update nodes affected by environment changes
- Handles dynamic obstacles by incrementally adjusting the path
- Provides real-time replanning capabilities for the moving boat

### 3. Helper Codes

#### **Gate Matching (`buoys2gates`):**
- Builds edges between red and green buoys within maximum matching distance
- Finds optimal buoy pairs using graph matching algorithms
- Returns gate matches as list of tuples: `(red_key, green_key, is_virtual)`
- Paired gates have `is_virtual=False`
- Prioritizes closest buoys and considers boat position to order gates by proximity

#### **Handling Unpaired Gates:**
In `GateBuilder.compute_preliminary_midpoints()`:
- Estimates heading direction from previous midpoint or toward buoy
- Offsets unpaired red buoys LEFT by `unpaired_red_offset`
- Creates virtual gates for unpaired buoys
- Estimates heading via `estimate_heading()`

#### **Used Buoy Determination:**
In `check_and_advance_logic()`:
1. Checks if boat has passed gate using `is_gate_passed()`:
   - Close proximity detection
   - Behind boat check (opposite vector, heading < 0)
2. If passed:
   - Adds gate red-key to used values
   - Adds gate green-key to used values
   - Stores gate in `passed_gate_ids`
   - Stores midpoint in `passed_midpoints` list
   - Advances current index to next gate
3. Filters used buoys from future gate computations

#### **Gate Midpoint Determination:**
Two-pass process in `GateBuilder.build_gates()`:

**Pass 1 - Preliminary Midpoints:**
- Paired gates: simple average of red and green buoy positions
- Virtual gates: buoy location + offset

**Pass 2 - `compute_gate_midpoints_and_heading()`:**
- Refines midpoints and headings using context from next & previous midpoints
- Ensures smooth path progression through the channel

## How to Run

Launch the complete navigation system using the demo launch file:

```bash
ros2 launch greenhorn_nav demo.launch.py
```

This will start:
- Local Goal Selector node
- LPA* planning node
- Required visualization and support nodes

The system will automatically:
1. Subscribe to buoy detection topics
2. Build and maintain gate sequence
3. Compute optimal paths through the channel
4. Publish local goal points for the boat controller

## Dependencies

- ROS2 (Robot Operating System)
- Python 3
- Standard ROS navigation stack packages

## Algorithm Background

**LPA* (Lifelong Planning A*)** is an incremental heuristic search algorithm that:
- Adapts to graph changes without full recalculation
- Updates only nodes affected by environmental changes
- Maintains efficiency in dynamic environments where obstacles or goals may shift
- Ideal for real-time robotic navigation in changing conditions

This makes it particularly well-suited for the RobotX navigation channel task where buoy positions may be updated as the boat moves and new sensor data becomes available.

