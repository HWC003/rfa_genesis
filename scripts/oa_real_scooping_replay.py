"""
Replay a soybean CSV through the two arm action controllers.

--preview-approach displays a virtual zero-to-start motion in MoveIt/RViz.
It does not send action goals or move hardware.
--execute moves from measured joint positions to the CSV start, then replays it.
"""
import argparse
import csv
from pathlib import Path
import time
from sensor_msgs.msg import JointState 

import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.qos import qos_profile_sensor_data
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from action_msgs.msg import GoalStatus

class ScoopingReplay(Node):
    def __init__(self):
        super().__init__('scooping_replay')

        self.left_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/left_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.right_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/right_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.current_positions = {}
        self.joint_received_at = {}

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos_profile_sensor_data
        )

    def joint_state_callback(self, msg):
        for name, position in zip(msg.name, msg.position):
            self.current_positions[name] = position
            self.joint_received_at[name] = time.monotonic()

    def read_current_positions(self, timeout_s=5.0):
        required_names = [
            f"openarm_{side}_joint{i}"
            for side in ["left", "right"]
            for i in range(1, 8)
        ]

        deadline = time.monotonic() + timeout_s

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

            if all(name in self.current_positions and
                   time.monotonic() - self.joint_received_at[name] < 0.5
                   for name in required_names):
                position = {
                    name: self.current_positions[name]
                    for name in required_names
                }

                if not all(math.isfinite(pos) for pos in position.values()):
                    raise ValueError("Received invalid joint positions")

                return position

        raise RuntimeError("Did not receive all 14 arm joints on /joint_states")

    def check_connections(self):
        left_ready = self.left_client.wait_for_server(timeout_sec=5.0)
        right_ready = self.right_client.wait_for_server(timeout_sec=5.0)

        self.get_logger().info(f'Left arm action server ready: {left_ready}')
        self.get_logger().info(f'Right arm action server ready: {right_ready}')
        if not left_ready or not right_ready:
            raise RuntimeError("Both arm action servers must be available")

    def load_trajectory(self, csv_path):
        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        if not rows:
            raise ValueError("The trajectory CSV is empty.")

        self.get_logger().info(f"Loaded trajectory: {csv_path.name}")
        self.get_logger().info(f"Number of points: {len(rows)}")
        self.get_logger().info(
            f"Final timestamp: {rows[-1]['time_from_start_s']} seconds"
        )

        return rows

    def build_goal(self, side, rows):
        goal = FollowJointTrajectory.Goal()

        joint_names=[
            f"openarm_{side}_joint{i}"
            for i in range(1, 8)
        ]
        goal.trajectory.joint_names = joint_names

        previous_time_ns = -1

        for row in rows:
            positions = [
                float(row[name])
                for name in joint_names
            ]
            time_s = float(row["time_from_start_s"])

            if not all(math.isfinite(pos) for pos in positions):
                raise ValueError(f"Non-finite joint positions found in row: {row}")

            if not math.isfinite(time_s) or time_s < 0:
                raise ValueError(f"Non-finite time_from_start_s found in row: {row}")

            time_ns = round(time_s * 1e9)

            if time_ns <= previous_time_ns:
                raise ValueError("Timestamps must increase")

            point = JointTrajectoryPoint()
            point.positions = positions 
            point.time_from_start = Duration(nanoseconds=time_ns).to_msg()
            goal.trajectory.points.append(point)
            previous_time_ns = time_ns

        return goal

    def build_zero_approach(self, replay_goal, duration_s=10.0):
        """Virtual zero-to-start approach used by the RViz preview."""
        zeros = dict.fromkeys(replay_goal.trajectory.joint_names, 0.0)
        return self.build_approach(replay_goal, zeros, duration_s)

    def build_approach(self, replay_goal, current_positions, duration_s=10.0):
        """Sample a quintic joint spline from measured angles to CSV start at 200 Hz.

        Both endpoint velocities and accelerations are zero. This is joint-space
        interpolation; it does not perform collision checking or motion planning.
        """
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Approach duration must be finite and positive")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(replay_goal.trajectory.joint_names)
        starts = [current_positions[name] for name in goal.trajectory.joint_names]
        targets = replay_goal.trajectory.points[0].positions
        changes = [target - start for start, target in zip(starts, targets)]
        steps = math.ceil(duration_s / 0.005)

        for step in range(steps + 1):
            u = step / steps
            blend = 10*u**3 - 15*u**4 + 6*u**5
            speed = (30*u**2 - 60*u**3 + 30*u**4) / duration_s
            acceleration = (60*u - 180*u**2 + 120*u**3) / duration_s**2

            point = JointTrajectoryPoint()
            point.positions = [start + blend * delta for start, delta in zip(starts, changes)]
            point.velocities = [speed * delta for delta in changes]
            point.accelerations = [acceleration * delta for delta in changes]
            point.time_from_start = Duration(
                nanoseconds=round(u * duration_s * 1_000_000_000)
            ).to_msg()
            goal.trajectory.points.append(point)

        return goal

    def execute_pair(self, left_goal, right_goal, label):
        """Schedule both arms together; wait for BOTH successful action results.

        Two controllers are not atomic. Rejection, timeout, interruption, or an
        arm failure triggers best-effort cancellation of all accepted goals.
        """
        goals = {"left": left_goal, "right": right_goal}
        clients = {"left": self.left_client, "right": self.right_client}
        handles, sends, cancellations = {}, {}, []
        stopping = False
        start_delay = 3.0
        start = self.get_clock().now() + Duration(seconds=start_delay)

        def on_response(side, future):
            if future.exception() is None:
                handle = future.result()
                if handle.accepted:
                    handles[side] = handle
                    if stopping:
                        cancellations.append(handle.cancel_goal_async())

        try:
            for side, goal in goals.items():
                goal.trajectory.header.stamp = start.to_msg()
                sends[side] = clients[side].send_goal_async(goal)
                sends[side].add_done_callback(lambda f, side=side: on_response(side, f))

            deadline = time.monotonic() + start_delay - 0.5
            while not all(f.done() for f in sends.values()):
                if not rclpy.ok() or time.monotonic() >= deadline:
                    raise RuntimeError(f"{label}: controllers did not accept in time")
                rclpy.spin_once(self, timeout_sec=0.05)
            for side, future in sends.items():
                handle = future.result()
                if not handle.accepted:
                    raise RuntimeError(f"{label}: {side} goal rejected")
                handles[side] = handle
            if self.get_clock().now().nanoseconds >= start.nanoseconds - 250_000_000:
                raise RuntimeError(f"{label}: insufficient time before scheduled start")

            self.get_logger().info(f"{label}: both goals accepted; waiting for completion")
            results = {side: handle.get_result_async() for side, handle in handles.items()}
            duration = max(
                g.trajectory.points[-1].time_from_start.sec +
                g.trajectory.points[-1].time_from_start.nanosec / 1e9
                for g in goals.values()
            )
            deadline = time.monotonic() + start_delay + duration + 5.0
            pending = set(results)
            while pending:
                if not rclpy.ok() or time.monotonic() >= deadline:
                    raise RuntimeError(f"{label}: execution timed out")
                rclpy.spin_once(self, timeout_sec=0.05)
                for side in list(pending):
                    if results[side].done():
                        response = results[side].result()
                        if (response.status != GoalStatus.STATUS_SUCCEEDED or
                                response.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL):
                            raise RuntimeError(
                                f"{label}: {side} failed: status={response.status}, "
                                f"code={response.result.error_code}, {response.result.error_string}"
                            )
                        pending.remove(side)
                        self.get_logger().info(f"{label}: {side} completed")
        except BaseException:
            stopping = True
            for handle in handles.values():
                cancellations.append(handle.cancel_goal_async())
            # Also process late acceptances so their goals receive cancellation.
            deadline = time.monotonic() + 2.0
            while rclpy.ok() and time.monotonic() < deadline:
                if (all(f.done() for f in sends.values()) and
                        all(f.done() for f in cancellations)):
                    break
                rclpy.spin_once(self, timeout_sec=0.05)
            self.get_logger().error(f"{label}: stopped; cancellation requested for accepted goals")
            raise

    def wait_for_start_pose(self, goals, tolerance=0.03, timeout_s=5.0):
        """Require fresh measured angles near the CSV start before replay."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            # Discard cached feedback and require new samples for this check.
            self.current_positions.clear()
            positions = self.read_current_positions(
                timeout_s=min(1.0, max(0.01, deadline - time.monotonic()))
            )
            error = max(
                abs(positions[name] - target)
                for goal in goals
                for name, target in zip(goal.trajectory.joint_names,
                                        goal.trajectory.points[0].positions)
            )
            if error <= tolerance:
                self.get_logger().info(f"Both arms at start pose; max error {error:.4f} rad")
                return
        raise RuntimeError("Arms did not reach the CSV start pose; replay not sent")

    def preview_approach(self, left_approach, right_approach):
        """Publish the two arms as ONE planned trajectory for RViz animation only."""
        from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
        from rclpy.qos import QoSProfile, DurabilityPolicy

        left = left_approach.trajectory
        right = right_approach.trajectory
        if len(left.points) != len(right.points):
            raise ValueError("Both arms must share the same approach timeline")

        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = left.joint_names + right.joint_names
        for lp, rp in zip(left.points, right.points):
            if lp.time_from_start != rp.time_from_start:
                raise ValueError("Left and right timestamps must match")
            point = JointTrajectoryPoint()
            point.positions = list(lp.positions) + list(rp.positions)
            point.velocities = list(lp.velocities) + list(rp.velocities)
            point.accelerations = list(lp.accelerations) + list(rp.accelerations)
            point.time_from_start = lp.time_from_start
            trajectory.joint_trajectory.points.append(point)

        display = DisplayTrajectory()
        display.trajectory_start.is_diff = True
        display.trajectory_start.joint_state.name = list(
            trajectory.joint_trajectory.joint_names
        )
        display.trajectory_start.joint_state.position = [0.0] * 14
        display.trajectory = [trajectory]

        self.preview_publisher = self.create_publisher(
            DisplayTrajectory, "/display_planned_path",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        # Retain the preview for late transient-local subscribers, and publish
        # again when a subscriber connects so volatile RViz subscribers see it too.
        self.preview_publisher.publish(display)
        previous_count = 0

        def on_subscriber_change():
            nonlocal previous_count
            count = self.preview_publisher.get_subscription_count()
            if count > previous_count:
                self.preview_publisher.publish(display)
                self.get_logger().info("RViz subscriber connected; preview published")
            previous_count = count

        self.preview_connection_timer = self.create_timer(0.5, on_subscriber_change)
        self.get_logger().info(
            "Preview ready on /display_planned_path; waiting for RViz. "
            "Use the MoveIt demo's MotionPlanning > Planned Path display. "
            "No action goals sent. Ctrl+C exits."
        )
        rclpy.spin(self)

def main():
    parser = argparse.ArgumentParser(
        description="Replay a soybean CSV through the two arm action controllers."
    )
    parser.add_argument(
        "trajectory",
        type=Path,
        help="Path to the trajectory CSV",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preview-approach", action="store_true",
        help="Animate zero-to-start in MoveIt/RViz only; sends no action goals",
    )
    mode.add_argument(
        "--execute", action="store_true",
        help="Send approach then CSV goals to the connected arm controllers",
    )
    parser.add_argument(
        "--approach-duration", type=float, default=10.0,
        help="Seconds for approach to the starting pose (default: 10)",
    )
    args = parser.parse_args()
    if not math.isfinite(args.approach_duration) or args.approach_duration <= 0:
        parser.error("--approach-duration must be finite and positive")

    rclpy.init()
    node = ScoopingReplay()

    try:
        if not args.preview_approach:
            node.check_connections()
        rows = node.load_trajectory(args.trajectory)

        left_goal = node.build_goal("left", rows)
        right_goal = node.build_goal("right", rows)

        node.get_logger().info(
            f"Left goal contains {len(left_goal.trajectory.points)} points."
        )
        node.get_logger().info(
            f"Right goal contains {len(right_goal.trajectory.points)} points."
        )

        if args.preview_approach:
            left_approach = node.build_zero_approach(left_goal, args.approach_duration)
            right_approach = node.build_zero_approach(right_goal, args.approach_duration)
            node.preview_approach(left_approach, right_approach)
            return

        current_positions = node.read_current_positions()

        for side, goal in (
            ("left", left_goal),
            ("right", right_goal),
        ):
            current = [
                current_positions[name]
                for name in goal.trajectory.joint_names
            ]

            target = goal.trajectory.points[0].positions

            largest_change = max(
                abs(target_angle - current_angle)
                for target_angle, current_angle in zip(target, current)
            )

            node.get_logger().info(
                f"{side}: largest joint change to starting pose "
                f"= {math.degrees(largest_change):.1f} degrees"
            )

        left_start = [
            float(rows[0][f"openarm_left_joint{i}"])
            for i in range(1, 8)
        ]

        right_start = [
            float(rows[0][f"openarm_right_joint{i}"])
            for i in range(1, 8)
        ]

        node.get_logger().info(f"Left starting angles: {left_start}")
        node.get_logger().info(f"Right starting angles: {right_start}")

        if args.execute:
            # Re-read immediately before building the approach, rather than
            # assuming the real robot is at the preview's virtual zero pose.
            node.current_positions.clear()
            current_positions = node.read_current_positions()
            left_approach = node.build_approach(left_goal, current_positions, args.approach_duration)
            right_approach = node.build_approach(right_goal, current_positions, args.approach_duration)
            for goal in (left_approach, right_approach):
                goal.goal_tolerance = [
                    JointTolerance(name=name, position=0.03)
                    for name in goal.trajectory.joint_names
                ]
                goal.goal_time_tolerance = Duration(seconds=2.0).to_msg()
            node.execute_pair(left_approach, right_approach, "Approach")
            node.wait_for_start_pose((left_goal, right_goal))
            node.execute_pair(left_goal, right_goal, "Replay")
            node.get_logger().info("Approach and replay completed")
        
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
